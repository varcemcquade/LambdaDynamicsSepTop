import MDAnalysis as mda
from MDAnalysis.topology import guessers
from MDAnalysis.analysis import rms, align, dssp
from MDAnalysis.lib.distances import distance_array, calc_angles, calc_dihedrals
import networkx as nx
from rdkit import Chem
import numpy as np
from scipy import stats
from scipy.signal.windows import cosine

force_const = 83.68
R = 8.31445985 * 0.001  # Gas constant in kJ/mol/K
T = 298.15


def select_ligand_atoms(complex_psf, complex_dcd, ligand_segid):
    """
    This function selects three ligand atoms for boresch restraints. Shortest distance between 
    each pair of atoms is calculated, and the longest path of this set is collected. Middle of longest shortest path
    is chosen to be center of mass atom. Closest ring atom to COM atom is the first true atom selection, L1. Subsequently, 
    L2 and L3 are chosen as the closest ring atoms to L1.

    :param complex_psf: 
        psf file of solvated protein-ligand complex
    :param complex_dcd:
        equilibration dcd file of solvated protein-ligand complex
    :param ligand_segid:
        editable segment id of ligand
        
    :return L1, L2, L3: 
        atom indices of atom selection heuristic
    """

    ligand_list = []

    # 0-based indexing starts
    u = mda.Universe(complex_psf, complex_dcd)

    # Make sure atoms have element attribute for RDKit conversion
    if not hasattr(u.atoms, "elements"):
        names = [str(n) for n in u.atoms.names]  # plain Python strings
        elems = [guessers.guess_atom_element(n) for n in names]  # per-atom
        u.add_TopologyAttr("elements", elems)

    ligand = u.select_atoms("segid %s" % ligand_segid)
    heavy_ligand = ligand.select_atoms("not name H* LP*")

    ligand_length = len(ligand.atoms)
    local = {atom.index: i for i, atom in enumerate(heavy_ligand)} # Dictionary for global index -> local index
    inv_local = {i: j for j, i in local.items()}
    idx = set(local.keys())
    ligand_graph = nx.Graph()
    ligand_graph.add_nodes_from(range(len(heavy_ligand.atoms)))

    for b in heavy_ligand.bonds:
        i = b.atoms[0].index  # global index of first atom
        j = b.atoms[1].index  # global index of second atom
        if i in idx and j in idx:
            ligand_graph.add_edge(local[i], local[j])

    # Find longest shortest path in ligand, get middle atom of that
    short_paths = dict(nx.shortest_path(ligand_graph))
    longest_paths = []
    longest_path_length = 0
    center = 0

    for i in short_paths.values():
        for key, value in i.items():
            if len(value) > longest_path_length:
                longest_path_length = len(value)
                longest_paths.clear()
                longest_paths.append(value)
            elif len(value) == longest_path_length:
                longest_paths.append(value)
        # there might be multiple longest path, just choose first one for now
        center = longest_paths[0][int(len(longest_paths[0]) / 2)]

    # Collect L1
    ligand_list.append(inv_local[center])

    aromatic_atoms = ligand.select_atoms("smarts a")

    # Collect L2 and L3
    for i in ligand_graph[center].keys(): # Loop through local neighbor indices
        index = inv_local[i] # Get global neighbor indices
        for j in aromatic_atoms.indices:
            if index == j:
                ligand_list.append(index)

    # If not enough aromatic neighbors, move to ordinary neighbors
    if len(ligand_list) < 3:
        for i in ligand_graph[center].keys():
            index = inv_local[i]
            ligand_list.append(index)

    if len(ligand_list) > 3:
        ligand_list = ligand_list[:3]

    # 0-based indexing stops
    for i in range(len(ligand_list)):
        ligand_list[i] += 1

    l1 = ligand_list[0]
    l2 = ligand_list[1]
    l3 = ligand_list[2]

    return l1, l2, l3

def select_protein_atoms(complex_psf, complex_equil_dcd, l1, rmsf_thresh=0.2, skip_start=20, skip_end=10, min_len_H=8,
    min_len_E=5, trim_H=3, trim_E=2,):
    """
        This function selects filters for C-alpha/C-beta, in the middle of a helix, RMSF < 0.1 nm,
        atom distance away from ligand COM (l1) 1nm < distance < 3nm.

        :param complex_psf:
            psf file of complex
        :param complex_equil_dcd:
            equilibration dcd file of solvated protein-ligand complex
        :param l1:
            Center of longest shortest path within ligand, returned by select_ligand_atoms (L1)
        :return protein_list:
            list of 0-based protein atom indexes
        """

    ## UNIVERSE SETUP
    u = mda.Universe(complex_psf, complex_equil_dcd)

    # RMSF setup
    avg = align.AverageStructure(u, u, select="protein and name CA", ref_frame=0).run()
    align.AlignTraj(u, avg.results.universe, select="protein and name CA", in_memory=True).run()
    candidates = u.select_atoms("protein and name CA CB")
    rmsf_vals = rms.RMSF(candidates).run().results.rmsf
    # Dictionary of atom_index: rmsf
    candidate_rmsf = dict(zip(candidates.indices, rmsf_vals))

    # DSSP on frame 0 of dcd, change to step4.crd later
    d = dssp.DSSP(u).run(start=0, stop=1)
    resids = d.results.resids.astype(int)
    sec = np.char.strip(np.asarray(d.results.dssp[0], dtype=str))
    N = len(resids)

    ## SELECT SECONDARY STRUCTURE
    nH = np.count_nonzero(sec == 'H')
    nE = np.count_nonzero(sec == 'E')

    if nH >= nE:
        targets = ['H']
    else:
        targets = ['H', 'E']

    core_pos = []

    for t in targets:
        mask = (sec == t).astype(int)
        # contiguous runs via edges in mask
        edges = np.diff(np.r_[0, mask, 0])
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]  # exclusive

        min_len = min_len_H if t == 'H' else min_len_E
        trim = trim_H if t == 'H' else trim_E

        for s, e in zip(starts, ends):
            if e - s < min_len:
                continue
            # respect skip windows
            s2 = max(s, skip_start)
            e2 = min(e, N - skip_end)
            if e2 - s2 < min_len:
                continue
            # trim ends of the run
            s3 = s2 + trim
            e3 = e2 - trim
            if e3 <= s3:
                continue
            core_pos.extend(range(s3, e3))

    if not core_pos:
        return u.atoms[:0]

    core_resids = resids[sorted(set(core_pos))]
    resid_str = " ".join(map(str, core_resids.tolist()))
    core_atoms = u.select_atoms(f"protein and (name CA CB) and (resid {resid_str})")
    if core_atoms.n_atoms == 0:
        return core_atoms

    candidates = []

    # Select stable atoms
    for i in core_atoms.indices:
        if candidate_rmsf[i] < rmsf_thresh:
            candidates.append(i)

    # L1 is 1-based index
    lig = u.atoms[l1 - 1]

    candidates_str = " ".join(map(str, candidates))
    candidates_atoms = u.select_atoms(f"index {candidates_str}")

    # Distance (A) from every protein atom to ligand center
    dA = distance_array(candidates_atoms.positions, lig.position, box=u.dimensions)

    protein_list = []

    for i, j in enumerate(dA):
        if 10.0 < j < 30.0:
            protein_list.append(candidates_atoms.indices[i])

    protein_list = [int(i) for i in protein_list]

    return protein_list

def check_angle(angle):
    # check if angle is <10kT from 0 or 180
    check1 = 0.5 * force_const * np.power((angle - 0.0) / 180.0 * np.pi, 2)
    check2 = 0.5 * force_const * np.power((angle - 180.0) / 180.0 * np.pi, 2)
    ang_check_1 = check1 / (R*T)
    ang_check_2 = check2 / (R*T)
    if ang_check_1 < 10.0 or ang_check_2  < 10.0:
        return False
    return True

def _is_collinear(positions, atoms, threshold=0.9):
    """Report whether any sequential vectors in a sequence of atoms are collinear.
    Parameters
    ----------
    positions : n_atoms x 3 simtk.unit.Quantity
        Reference positions to use for imposing restraints (units of length).
    atoms : iterable of int
        The indices of the atoms to test.
    threshold : float, optional, default=0.9
        Atoms are not collinear if their sequential vector separation dot
        products are less than ``threshold``.
    Returns
    -------
    result : bool
        Returns True if any sequential pair of vectors is collinear; False otherwise.

    Modification proposed by Eric Dybeck
    """
    result = False
    for i in range(len(atoms) - 2):
        v1 = positions[atoms[i + 1], :] - positions[atoms[i], :]
        v2 = positions[atoms[i + 2], :] - positions[atoms[i + 1], :]
        normalized_inner_product = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
        result = result or (np.abs(normalized_inner_product) > threshold) #INDENT AND ABSOLUTE VALUE ADDED HERE
    return result

def select_boresch_atoms(complex_psf, complex_equil_dcd, protein_list, l1, l2, l3):
    """Select possible protein atoms for Boresch-style restraints.
    Parameters
    ----------
    :param complex_psf:
        psf file of complex
    :param complex_equil_dcd:
        equilibration dcd file of solvated protein-ligand complex
    :param l1, l2, l3:
        Indices of selected ligand atoms
    :param protein_list:
        List of overlapping protein candidates for lig 1 and lig 2 (0-based)
    :return restrained_atoms:
        List of atom indices (1-based) to be used for restraints. Ordered p1, p2, p3, l1, l2, l3
    """

    u = mda.Universe(complex_psf, complex_equil_dcd)
    coords = np.array([u.atoms.positions.copy() for ts in u.trajectory]) # shaped (n_frames, n_atoms, 3) --  look up atoms w/ coords[:, idx, :]
    n_frames = coords.shape[0]
    protein_list_length = len(protein_list)
    all_protein_combinations = []

    print(f"Total of {protein_list_length} protein atoms.")

    for i, p1 in enumerate(protein_list):
        print(f"{i} of {protein_list_length} atoms complete.")
        a1, d1 = [], []

        p1_coords = coords[:, p1, :]
        l1_coords = coords[:, l1 - 1, :]
        l2_coords = coords[:, l2 - 1, :]
        l3_coords = coords[:, l3 - 1, :]

        for ts in range(n_frames):
            a1.append(np.degrees(calc_angles(p1_coords[ts], l1_coords[ts], l2_coords[ts])))
            d1.append(np.degrees(calc_dihedrals(p1_coords[ts], l1_coords[ts], l2_coords[ts], l3_coords[ts])))

        average_a1 = stats.circmean(a1, high=180, low=-180)
        var_a1 = stats.circvar(a1, high=180, low=-180)
        average_d1 = stats.circmean(d1, high=180, low=-180)
        var_d1 = stats.circvar(d1, high=180, low=-180)

        check_a1 = check_angle(average_a1)
        collinear = _is_collinear(u.atoms.positions, [int(p1), int(l1-1), int(l2-1), int(l3-1)])

        if not (check_a1 and not collinear and var_a1 < 100 and var_d1 < 300 and -150 < average_d1 < 150):
            continue

        # Collect valid p2s for this p1
        min_distance = 5 # Angstroms
        max_distance = (u.dimensions[0] / 2)
        valid_p2s = []

        for p2 in protein_list:
            if p2 == p1:
                continue

            a2, d2, distances = [], [], []
            p2_coords = coords[:, p2, :]

            for ts in range(n_frames):
                a2.append(np.degrees(calc_angles(p2_coords[ts], p1_coords[ts], l1_coords[ts])))
                d2.append(np.degrees(calc_dihedrals(p2_coords[ts], p1_coords[ts], l1_coords[ts], l2_coords[ts])))
                distances.append(distance_array(p1_coords[ts], p2_coords[ts]))

            average_a2 = stats.circmean(a2, high=180, low=-180)
            var_a2 = stats.circvar(a2, high=180, low=-180)
            average_d2 = stats.circmean(d2, high=180, low=-180)
            var_d2 = stats.circvar(d2, high=180, low=-180)

            check_a2 = check_angle(average_a2)
            collinear = _is_collinear(coords[0],[int(p2), int(p1), int(l1-2), int(l2-1)])

            if (
                check_a2
                and not collinear
                and var_a2 < 100
                and var_d2 < 300
                and -150 < average_d2 < 150
                and min_distance < np.mean(distances) < max_distance
            ):
                    valid_p2s.append(p2)

        # Collect all valid p3 for each p2
        for p2 in valid_p2s:
            p2_coords = coords[:, p2, :]
            p3_candidates = []
            distance_products = []

            for p3 in protein_list:
                if p3 in (p1, p2):
                    continue

                p3_coords = coords[:, p3, :]
                d3 = []

                for ts in range(n_frames):
                    d3.append(np.degrees(calc_dihedrals(p3_coords[ts], p2_coords[ts], p1_coords[ts], l1_coords[ts])))

                average_d3 = stats.circmean(d3, high=180, low=-180)
                var_d3 = stats.circvar(d3, high=180, low=-180)
                collinear = _is_collinear(u.atoms.positions, [int(p3), int(p2), int(p1), int(l1-2)])

                if not collinear and var_d3 < 300 and -150 < average_d3 < 180:
                    distances_p1_p3 = []
                    distances_p2_p3 = []
                    distances_l1_p3 = []

                    for ts in range(n_frames):
                        distances_p1_p3.append(distance_array(p1_coords[ts], p3_coords[ts]))
                        distances_p2_p3.append(distance_array(p2_coords[ts], p3_coords[ts]))
                        distances_l1_p3.append(distance_array(l1_coords[ts], p3_coords[ts]))

                    distance_p1_p3 = np.mean(distances_p1_p3)
                    distance_p2_p3 = np.mean(distances_p2_p3)
                    distance_l1_p3 = np.mean(distances_l1_p3)

                    if distance_p1_p3 < max_distance and distance_p2_p3 < max_distance and distance_l1_p3 < max_distance:
                        p3_candidates.append(p3)
                        distance_products.append(distance_p1_p3 * distance_p2_p3)

            # Add ALL valid combinations
            for idx, p3 in enumerate(p3_candidates):
                all_protein_combinations.append([p1+1, p2+1, p3+1])

    print(f"Found {len(all_protein_combinations)} protein triplets.")
    return all_protein_combinations

def compute_dist_angle_dih(complex_psf, complex_equil_dcd, restrained_atoms):
    """ Compute distance, angles, dihedrals.
    :param complex_psf:
        psf file of complex
    :param complex_equil_dcd:
        equilibration dcd file of solvated protein-ligand complex
    :param restrained_atoms:
        List of atom indices (1-based) to be used for restraints. Ordered p1, p2, p3, l1, l2, l3
    :returns distance_l1_p1, theta1, theta2, phi1, phi2, phi3
        Values for distance (l1 to p1), theta1 (l1, p1, p2), theta2 (l2, l1, p1), phi1 (l1, p1, p2, p3),
        phi2 (l2, l1, p1, p2), and phi3 (l3, l2, l1, p1)
    """

    # 1-based indexing ---> 0-based indexing
    p1 = restrained_atoms[0] - 1
    p2 = restrained_atoms[1] - 1
    p3 = restrained_atoms[2] - 1
    l1 = restrained_atoms[3] - 1
    l2 = restrained_atoms[4] - 1
    l3 = restrained_atoms[5] - 1

    u = mda.Universe(complex_psf, complex_equil_dcd)

    distances_l1_p1 = []
    theta1s = []
    theta2s = []
    phi1s = []
    phi2s = []
    phi3s = []

    for ts in u.trajectory:
        p1_coords = u.atoms[p1].position
        p2_coords = u.atoms[p2].position
        p3_coords = u.atoms[p3].position
        l1_coords = u.atoms[l1].position
        l2_coords = u.atoms[l2].position
        l3_coords = u.atoms[l3].position

        distances_l1_p1.append(distance_array(l1_coords, p1_coords))
        theta1s.append(calc_angles(l1_coords, p1_coords, p2_coords))
        theta2s.append(calc_angles(l2_coords, l1_coords, p1_coords))
        phi1s.append(calc_dihedrals(l1_coords, p1_coords, p2_coords, p3_coords))
        phi2s.append(calc_dihedrals(l2_coords, l1_coords, p1_coords, p2_coords))
        phi3s.append(calc_dihedrals(l3_coords, l2_coords, l1_coords, p1_coords))

    distance_l1_p1 = np.mean(distances_l1_p1)

    theta1 = np.mean(np.degrees(theta1s))
    theta2 = np.mean(np.degrees(theta2s))

    phi1s_cos_average = np.mean(np.cos(phi1s))
    phi1s_sin_average = np.mean(np.sin(phi1s))
    phi2s_cos_average = np.mean(np.cos(phi2s))
    phi2s_sin_average = np.mean(np.sin(phi2s))
    phi3s_cos_average = np.mean(np.cos(phi3s))
    phi3s_sin_average = np.mean(np.sin(phi3s))
    phi1 = np.degrees(np.atan2(phi1s_sin_average, phi1s_cos_average))
    phi2 = np.degrees(np.atan2(phi2s_sin_average, phi2s_cos_average))
    phi3 = np.degrees(np.atan2(phi3s_sin_average, phi3s_cos_average))

    return distance_l1_p1, theta1, theta2, phi1, phi2, phi3

def return_atom_info(complex_psf, boresch_list, ligand_segid, protein_segid):
    # Save atom info from .psf, 1-based indexing

    l1name = ""
    l2name = ""
    l3name = ""
    p1resid = ""
    p1resnum = ""
    p1name = ""
    p2resid = ""
    p2resnum = ""
    p2name = ""
    p3resid = ""
    p3resnum = ""
    p3name = ""

    with open(complex_psf) as file:
        for line in file:
            p1 = f"{boresch_list[0]} {protein_segid}"
            p2 = f"{boresch_list[1]} {protein_segid}"
            p3 = f"{boresch_list[2]} {protein_segid}"
            l1 = f"{boresch_list[3]} {ligand_segid}"
            l2 = f"{boresch_list[4]} {ligand_segid}"
            l3 = f"{boresch_list[5]} {ligand_segid}"

            fields = line.split()
            if len(fields) >= 2:
                candidate = f"{fields[0]} {fields[1]}"
                if p1 == candidate:
                    p1resnum = fields[2]
                    p1resid = fields[3]
                    p1name = fields[4]
                elif p2 == candidate:
                    p2resnum = fields[2]
                    p2resid = fields[3]
                    p2name = fields[4]
                elif p3 == candidate:
                    p3resnum = fields[2]
                    p3resid = fields[3]
                    p3name = fields[4]
                elif l1 == candidate:
                    l1name = fields[4]
                elif l2 == candidate:
                    l2name = fields[4]
                elif l3 == candidate:
                    l3name = fields[4]
            else:
                continue
            if "!NBOND" in line:
                break

    return (l1name, l2name, l3name, p1resid, p1resnum, p1name,
            p2resid, p2resnum, p2name, p3resid, p3resnum, p3name)

def write_boresch_variables(complex_psf, boresch_list, equilibrium_conditions, ligand_segid, protein_segid, n):
    # Write boresch_variables.str, use 1 or 2 as final param to signify reference

    d = equilibrium_conditions[0] # equilibrium distance of L1-P1, angstroms
    theta1 = equilibrium_conditions[1] # equilibrium angle of L1-P1-P2, degrees
    theta2 = equilibrium_conditions[2] # equilibrium angle of L2-L1-P2, degrees
    phi1 = equilibrium_conditions[3]  # equilibrium angle of L1-P1-P2-P3 dihedral, degrees
    phi2 = equilibrium_conditions[4] # equilibrium angle of L2-L1-P1-P2 dihedral, degrees
    phi3 = equilibrium_conditions[5] # equilibrium angle of L3-L2-L1-P1 dihedral, degrees

    dk = 20.0 # distance force constant, kcal/molA**2
    theta1k = 1.6 * d**2 # theta 1 force constant, kcal/mol*rad**2, scaling quadratically by distance, 5 A = 40 kcal/mol*rad**2
    theta2k = 20.0 # theta 2 force constant, kcal/mol*rad**2
    phi1k = (d * np.sin(np.deg2rad(theta1)))**2 # phi1 force constant, kcal/mol*rad**2
    phi23k = 20.0 # phi1 force constant, kcal/mol*rad**2

    atom_info = return_atom_info(complex_psf, boresch_list, ligand_segid, protein_segid)

    file = open(f"boresch_variables{n}.str", "w")
    file.write(f"SET LIGSEGID = {ligand_segid}\n\n")
    file.write(f"SET L1NAME = {atom_info[0]}\n")
    file.write(f"SET L2NAME = {atom_info[1]}\n")
    file.write(f"SET L3NAME = {atom_info[2]}\n\n")
    file.write(f"SET PROTSEGID = {protein_segid}\n\n")
    file.write(f"SET P1RESID = {atom_info[3]}\n")
    file.write(f"SET P1RESNUM = {atom_info[4]}\n")
    file.write(f"SET P1NAME = {atom_info[5]}\n\n")
    file.write(f"SET P2RESID = {atom_info[6]}\n")
    file.write(f"SET P2RESNUM = {atom_info[7]}\n")
    file.write(f"SET P2NAME = {atom_info[8]}\n\n")
    file.write(f"SET P3RESID = {atom_info[9]}\n")
    file.write(f"SET P3RESNUM = {atom_info[10]}\n")
    file.write(f"SET P3NAME = {atom_info[11]}\n\n")
    file.write(f"SET DISTANCEL1P1 = {d}\n")
    file.write(f"SET THETA1 = {theta1}\n")
    file.write(f"SET THETA2 = {theta2}\n")
    file.write(f"SET PHI1 = {phi1}\n")
    file.write(f"SET PHI2 = {phi2}\n")
    file.write(f"SET PHI3 = {phi3}\n\n")
    file.write(f"SET DISTANCEK = {dk}\n")
    file.write(f"SET THETA1K = {theta1k}\n")
    file.write(f"SET THETA2K = {theta2k}\n")
    file.write(f"SET PHI1K = {phi1k}\n")
    file.write(f"SET PHI23K = {phi23k}\n")
    file.close()

    return None
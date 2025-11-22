import numpy as np
from MDAnalysis.analysis import dssp
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals
import boresch_chk

MIN_DISTANCE = 5.0 # Angstroms
DEFAULT_MAX_DISTANCE = 20.0 # Angstroms
MAX_DIHEDRAL = 150.0 # Degrees

def select_protein_atoms(u, l1, box_size=84, skip_start=20, skip_end=10, min_len_H=8,
    min_len_E=5, trim_H=3, trim_E=2):
    """
        First filter of protein atoms for boresch selection. Criteria:
            - C-alpha/C-beta
            - Middle of a helix or B-sheet
            - RMSF < 0.1 nm
            - Atom distance away from ligand COM (l1) 1nm < distance < 3nm
        :param u:
            MDAnalysis universe, 0-based atom indices
        :param l1:
            Center of longest shortest path within ligand, returned by select_ligand_atoms (L1)
        :param skip_start, skip_end:
            Length of protein chain start and end trim
        :param min_len_H, min_len_E:
            Minimum length of helix and B-sheet, respectively
        :param trim_H, trim_E:
            Trim threshold for middle of secondary structure
        :return protein_list:
            list of 0-based protein atom indices
        """

    u.dimensions = [box_size, box_size, box_size, 90, 90, 90]
    d = dssp.DSSP(u).run()
    resids = d.results.resids.astype(int)
    sec = np.char.strip(np.asarray(d.results.dssp[0], dtype=str))
    res_count = len(resids)

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
            e2 = min(e, res_count - skip_end)
            if e2 - s2 < min_len:
                continue
            # trim ends of the run
            s3 = s2 + trim
            e3 = e2 - trim
            if e3 <= s3:
                continue
            core_pos.extend(range(s3, e3))

    if not core_pos:
        return []

    core_resids = resids[sorted(set(core_pos))]
    resid_str = " ".join(map(str, core_resids.tolist()))
    core_atoms = u.select_atoms(f"protein and (name CA CB) and (resid {resid_str})")
    if core_atoms.n_atoms == 0:
        return []

    # L1 is 1-based index
    lig = u.atoms[l1 - 1]

    candidates_atoms = core_atoms

    dA = np.array([calc_bonds(atom.position, lig.position, box=u.dimensions) for atom in candidates_atoms])

    protein_list = []

    for i, j in enumerate(dA):
        if 10.0 < j < 30.0:
            protein_list.append(candidates_atoms.indices[i])

    return [int(i) for i in protein_list]


def find_triplets(u, protein_atoms, l1, l2, l3):
    """ Select final protein atoms for Boresch-style restraints
    :param u:
        MDAnalysis universe, 0-based atom indices
    :param l1, l2, l3:
        Indices of selected ligand atoms
    :param protein_atoms:
        List of protein candidates overlapped by all ligands
    :return restrained_atoms:
        List of atom indices (1-based) to be used for restraints. Ordered l1, l2, l3, p1, p2, p3
    """

    coords = u.atoms.positions.copy() # shape: (n_atoms, 3)
    all_triplets = []

    for i, p1 in enumerate(protein_atoms):
        p1_coords = coords[p1, :]
        l1_coords = coords[l1 - 1, :]
        l2_coords = coords[l2 - 1, :]
        l3_coords = coords[l3 - 1, :]

        a1 = float(np.degrees(calc_angles(p1_coords, l1_coords, l2_coords, box=u.dimensions)))
        dih1 = float(np.degrees(calc_dihedrals(p1_coords, l1_coords, l2_coords, l3_coords, box=u.dimensions)))

        check_a1 = boresch_chk.check_angle(a1)
        collinear1 = boresch_chk.is_collinear(coords, [p1, l1 - 1, l2 - 1, l3 - 1])

        if not (check_a1 and not collinear1 and abs(dih1) < MAX_DIHEDRAL):
            continue

        # Collect valid p2s for this p1
        max_distance = (u.dimensions[0] / 2) if u.dimensions[0] > 0 else DEFAULT_MAX_DISTANCE
        valid_p2s = []

        for p2 in protein_atoms:
            if p2 == p1:
                continue

            p2_coords = coords[p2, :]

            a2 = float(np.degrees(calc_angles(p2_coords, p1_coords, l1_coords, box=u.dimensions)))
            dih2 = float(np.degrees(calc_dihedrals(p2_coords, p1_coords, l1_coords, l2_coords, box=u.dimensions)))
            dp1p2 = float(calc_bonds(p1_coords, p2_coords, box=u.dimensions))

            check_a2 = boresch_chk.check_angle(a2)
            collinear2 = boresch_chk.is_collinear(coords, [p2, p1, l1 - 1, l2 - 1])

            if (check_a2 and not collinear2 and abs(dih2) < MAX_DIHEDRAL
                and MIN_DISTANCE < dp1p2 < max_distance):
                    valid_p2s.append(p2)

        # Collect all valid p3 for each p2
        for p2 in valid_p2s:
            p2_coords = coords[p2, :]
            p3_candidates = []
            distance_products = []

            for p3 in protein_atoms:
                if p3 in (p1, p2):
                    continue

                p3_coords = coords[p3, :]
                dih3 = float(np.degrees(calc_dihedrals(p3_coords, p2_coords, p1_coords, l1_coords, box=u.dimensions)))
                collinear3 = boresch_chk.is_collinear(coords, [p3, p2, p1, l1 - 1])

                if collinear3 or not (abs(dih3) < MAX_DIHEDRAL):
                    continue

                dp1p3 = float(calc_bonds(p1_coords, p3_coords, box=u.dimensions))
                dp2p3 = float(calc_bonds(p2_coords, p3_coords, box=u.dimensions))
                dl1p3 = float(calc_bonds(l1_coords, p3_coords, box=u.dimensions))

                if (MIN_DISTANCE < dp1p3 < max_distance
                    and MIN_DISTANCE < dp2p3 < max_distance
                    and MIN_DISTANCE < dl1p3 < max_distance):
                    p3_candidates.append(p3)
                    distance_products.append(dp1p3 * dp2p3)

            # Add ALL valid combinations
            if p3_candidates:
                best_idx = int(np.argmax(distance_products))
                best_p3 = p3_candidates[best_idx]
                all_triplets.append([p1 + 1, p2 + 1, best_p3 + 1])
    return all_triplets


def conditions_met(u, lig_idx, prot_idx):
    """ Check if specific protein triplet from reference system meets conditions for other system
    :param u:
        Merged MDAnalysis universe of ref_prot + ligand
    :param lig_idx:
        Indices of l1, l2, l3 -- 1-based
    :param prot_idx:
        Indices of p1, p2, p3 -- 1-based
    :returns bool:
        True if conditions met, False otherwise
    """

    coords = u.atoms.positions.copy()

    # 1-based
    l1 = lig_idx[0]
    l2 = lig_idx[1]
    l3 = lig_idx[2]
    p1 = prot_idx[0]
    p2 = prot_idx[1]
    p3 = prot_idx[2]

    l1_coords = coords[l1 - 1, :]
    l2_coords = coords[l2 - 1, :]
    l3_coords = coords[l3 - 1, :]
    p1_coords = coords[p1 - 1, :]
    p2_coords = coords[p2 - 1, :]
    p3_coords = coords[p3 - 1, :]

    a1 = float(np.degrees(calc_angles(p1_coords, l1_coords, l2_coords, box=u.dimensions)))
    check_a1 = boresch_chk.check_angle(a1)
    collinear1 = boresch_chk.is_collinear(coords, [p1, l1 - 1, l2 - 1, l3 - 1])
    dih1 = float(np.degrees(calc_dihedrals(p1_coords, l1_coords, l2_coords, l3_coords, box=u.dimensions)))

    if not (check_a1 and not collinear1 and abs(dih1) < MAX_DIHEDRAL):
        return False

    max_distance = (u.dimensions[0] / 2) if u.dimensions[0] > 0 else DEFAULT_MAX_DISTANCE

    a2 = float(np.degrees(calc_angles(p2_coords, p1_coords, l1_coords, box=u.dimensions)))
    check_a2 = boresch_chk.check_angle(a2)
    collinear2 = boresch_chk.is_collinear(coords, [p2, p1, l1 - 1, l2 - 1])
    dih2 = float(np.degrees(calc_dihedrals(p2_coords, p1_coords, l1_coords, l2_coords, box=u.dimensions)))
    dp1p2 = float(calc_bonds(p1_coords, p2_coords, box=u.dimensions))

    if not (check_a2 and not collinear2 and abs(dih2) < MAX_DIHEDRAL
            and MIN_DISTANCE < dp1p2 < max_distance):
        return False

    dih3 = float(np.degrees(calc_dihedrals(p3_coords, p2_coords, p1_coords, l1_coords, box=u.dimensions)))
    collinear3 = boresch_chk.is_collinear(coords, [p3, p2, p1, l1 - 1])

    if collinear3 or not (abs(dih3) < MAX_DIHEDRAL):
        return False

    dp1p3 = float(calc_bonds(p1_coords, p3_coords, box=u.dimensions))
    dp2p3 = float(calc_bonds(p2_coords, p3_coords, box=u.dimensions))
    dl1p3 = float(calc_bonds(l1_coords, p3_coords, box=u.dimensions))

    if not (MIN_DISTANCE < dp1p3 < max_distance
        and MIN_DISTANCE < dp2p3 < max_distance
        and MIN_DISTANCE < dl1p3 < max_distance):
        return False

    return True
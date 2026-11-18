import numpy as np
import MDAnalysis as mda
from MDAnalysis.topology import guessers
from MDAnalysis.analysis import dssp
from MDAnalysis.lib.distances import distance_array, calc_angles, calc_dihedrals
import boresch_chk

def _dist(a, b, box):
    """
    Minimum-image distance between two atoms.

    Parameters:
        a (np.ndarray(shape=(3,), dtype=np.float32)): coordinates of atom a
        b (np.ndarray(shape=(3,), dtype=np.float32)): coordinates of atom b
        box (np.ndarray(shape=(6,), dtype=np.float32)): contents of u.dimensions
    
    Returns:
        (float): distancei in Angstroms
    """
    return float(distance_array(a.reshape(1, 3), b.reshape(1, 3), box=box)[0, 0])

def select_protein_atoms(u, l1_coords, prot_segid,
                         skip_start=15, skip_end=15,
                         min_len_H=8, min_len_E=5,
                         trim_H=3, trim_E=2):
    """
    First-pass selection of candidate protein atoms for Boresch restraints.

    Criteria applied in order:
        1. C-alpha or C-beta atoms only.
        2. Residue is in the CORE of a helix (H) or beta-sheet (E) -- not
           near chain termini and not near secondary-structure ends.
        3. Distance from L1 is 10 Å < d < 30 Å.

    Parameters:
        u (mda.Universe): complex universe
        l1_coords (np.ndarray(shape=(3,), dtype=np.float32)):
        prot_segid (str): protein SEGID
        skip_start (int): Skip residues within this many positions of the chain N-terminus
        skip_end (int): Skip residues within this many positions of the chain C-terminus.
        min_len_H (int): Minimum run length for a helix to be used.
        min_len_E (int): Minimum run length for a beta sheet to be used.
        trim_H (int): Trim this many residues from each end of a helix run.
        trim_E (int): Trim this many residues from each end of a sheet run.

    Returns:
        (list[int]): 3 element list containing three 0-based indices that refer to atoms in the complex
    """
    
    # "guess" element information from ligand universe if not filled
    if not hasattr(u.atoms, "elements"):
        names = [str(n) for n in u.atoms.names]
        elems = [guessers.guess_atom_element(n) for n in names]
        u.add_TopologyAttr("elements", elems)


    # prep protein selection for secondary structure prediction
    prot_u = u.select_atoms(f"segid {prot_segid}")
    # prot_u = prot_u.residues[1:-1].atoms

    n_N  = prot_u.select_atoms("name N").n_atoms
    n_CA = prot_u.select_atoms("name CA").n_atoms
    n_C  = prot_u.select_atoms("name C").n_atoms
    n_O  = prot_u.select_atoms("name O").n_atoms

    print(f"Protein atoms: {prot_u.n_atoms}")
    print(f"Backbone counts: N={n_N}, CA={n_CA}, C={n_C}, O={n_O}")
    
    sec_data = dssp.DSSP(prot_u).run()

    # results.resids is 1-D (n_residues,); results.dssp is (n_frames, n_residues)
    dssp_resids = sec_data.results.resids
    dssp_sec    = sec_data.results.dssp[0] # frame 0
    sec_structs_by_resid = {int(r): str(s).strip() for r, s in zip(dssp_resids, dssp_sec)}

    sorted_resids = sorted(sec_structs_by_resid)
    n_res = len(sorted_resids)

    # determine number of alpha helices and beta sheets
    num_H = sum(1 for s in sec_structs_by_resid.values() if s == "H")
    num_E = sum(1 for s in sec_structs_by_resid.values() if s == "E")

    structs = ["H"] if num_H >= num_E else ["H", "E"]

    core_resids = []
    for struct in structs:
        min_len = min_len_H if struct == "H" else min_len_E
        trim = trim_H if struct == "H" else trim_E

        # Build binary mask aligned with sorted_resids
        candidate_mask = [1 if sec_structs_by_resid.get(resid, "") == struct else 0 for resid in sorted_resids]

        padded = np.pad(candidate_mask, (1, 1))
        edges = np.diff(padded)
        starts = np.where(edges ==  1)[0]
        ends = np.where(edges == -1)[0]

        for s, e in zip(starts, ends):
            if e - s < min_len:
                continue
            # Apply chain-terminus skip
            s2 = max(s, skip_start)
            e2 = min(e, n_res - skip_end)
            if e2 - s2 < min_len:
                continue
            # Trim ends of the secondary structure run
            s3 = s2 + trim
            e3 = e2 - trim
            if e3 <= s3:
                continue
            core_resids.extend(sorted_resids[s3:e3])

    if not core_resids:
        return []

    core_resids = sorted(set(core_resids))
    resids_str = " ".join(str(r) for r in core_resids)
    candidate_atoms = u.select_atoms(f"protein and (name CA CB) and (resid {resids_str})")
    
    if candidate_atoms.n_atoms == 0:
        return []

    l1_coords = np.asarray(l1_coords).reshape(1, 3)
    dists = distance_array(candidate_atoms.positions, l1_coords, box=u.dimensions)[:, 0]
    protein_atoms = [int(candidate_atoms.indices[i]) for i, d in enumerate(dists) if 10.0 < d < 30.0]
    
    return protein_atoms

def find_triplets(u, protein_atoms, l1_name, l2_name, l3_name, lig_segid):
    """
    Find all valid (P1, P2, P3) protein triplets for Boresch restraints.

    Validity requires that every angle, dihedral, and inter-atom distance in
    the six-atom restrained set {L1, L2, L3, P1, P2, P3} satisfies:
        - Angles not within 10 kT of 0° or 180°.
        - No three consecutive atoms are collinear (dot-product threshold 0.9).
        - (UNIMPLEMENTED) Dihedrals not within ±150° of ±180°.
        - P2/P3 distances between 5 Å and half the box.

    Ligand atoms are located by name, so the function works correctly
    even when atom numbers differ across complexes.

    Parameters:
        u (mda.Universe): complex universe
        protein_atoms (list[int]): List of 0-based protein atom indices (candidates).
        l1_name, l2_name, l3_name (list[tuple[int, str]]): Ligand atom names.
        lig_segid (str): Ligand segment ID.
        box_size (float): Cubic box side length in Angstroms.
    
    Returns:
        all_triplets (tuple[str]): List of [p1, p2, p3] triplets with *0-based* indices.
    """
    coords = u.atoms.positions.copy()

    l1_coords = u.select_atoms(f"segid {lig_segid} and name {l1_name}")[0].position
    l2_coords = u.select_atoms(f"segid {lig_segid} and name {l2_name}")[0].position
    l3_coords = u.select_atoms(f"segid {lig_segid} and name {l3_name}")[0].position

    min_distance = 5.0
    max_distance = (u.dimensions[0] / 2.0) if u.dimensions[0] > 0 else 20.0
    all_triplets = []

    for p1 in protein_atoms:
        p1_coords = coords[p1]

        a1 = np.degrees(calc_angles(p1_coords, l1_coords, l2_coords))
        if not boresch_chk.check_angle(a1):
            continue

        if boresch_chk.is_collinear(np.array([p1_coords, l1_coords, l2_coords, l3_coords])):
            continue

        valid_p2s = []
        for p2 in protein_atoms:
            if p2 == p1:
                continue

            p2_coords = coords[p2]
            dp1p2 = _dist(p1_coords, p2_coords, u.dimensions)
            if not (min_distance < dp1p2 < max_distance):
                continue

            a2 = np.degrees(calc_angles(p2_coords, p1_coords, l1_coords))
            if not boresch_chk.check_angle(a2):
                continue

            if boresch_chk.is_collinear(np.array([p2_coords, p1_coords, l1_coords, l2_coords])):
                continue

            valid_p2s.append(p2)

        for p2 in valid_p2s:
            p2_coords = coords[p2]
            p3_candidates = []
            distance_products = []

            for p3 in protein_atoms:
                if p3 in (p1, p2):
                    continue

                p3_coords = coords[p3]

                if boresch_chk.is_collinear(np.array([p3_coords, p2_coords, p1_coords, l1_coords])):
                    continue

                dp1p3 = _dist(p1_coords, p3_coords, u.dimensions)
                dp2p3 = _dist(p2_coords, p3_coords, u.dimensions)
                dl1p3 = _dist(l1_coords, p3_coords, u.dimensions)

                if (min_distance < dp1p3 < max_distance and
                    min_distance < dp2p3 < max_distance and
                    min_distance < dl1p3 < max_distance):
                    p3_candidates.append(p3)
                    distance_products.append(dp1p3 * dp2p3)

            if p3_candidates:
                best_idx = int(np.argmax(distance_products))
                best_p3  = p3_candidates[best_idx]
                # Return 0-based indices
                all_triplets.append([p1, p2, best_p3])

    return all_triplets

def check_ligand_order(u, lig_names, prot_triplet, lig_segid):
    """
    Test whether one complete ligand ordering (L1, L2, L3) is satisfies 
    Boresch geometry conditions in a different complex for a specified
    protein triplet (P1, P2, P3).
    
    The ligand ordering must be used consistently for:
        angle B:     P1-L1-L2
        dihedral B:  P2-P1-L1-L2
        dihedral C:  P1-L1-L2-L3
    
    Parameters:
        u (MDA.universe): MDAnalysis Universe of the target complex (u.dimensions should already be set).
        lig_names (tuple[str]): Tuple/list (l1_name, l2_name, l3_name)
        prot_triplet (list[int]): [p1, p2, p3] with *0-based* protein atom indices.
        lig_segid (str): Ligand segment ID.
        box_size (int): Cubic box side length in Angstroms.
    
    Returns:
        (bool): True if all conditions are satisfied.
    """
    coords = u.atoms.positions.copy()

    l1_name, l2_name, l3_name = lig_names
    p1, p2, p3 = prot_triplet

    l1_sel = u.select_atoms(f"segid {lig_segid} and name {l1_name}")
    l2_sel = u.select_atoms(f"segid {lig_segid} and name {l2_name}")
    l3_sel = u.select_atoms(f"segid {lig_segid} and name {l3_name}")

    if l1_sel.n_atoms != 1 or l2_sel.n_atoms != 1 or l3_sel.n_atoms != 1:
        return False

    l1_coords = l1_sel[0].position
    l2_coords = l2_sel[0].position
    l3_coords = l3_sel[0].position

    p1_coords = coords[p1]
    p2_coords = coords[p2]
    p3_coords = coords[p3]

    min_distance = 5.0
    max_distance = (u.dimensions[0] / 2.0) if u.dimensions[0] > 0 else 20.0

    # distance restraint: P1-L1
    dp1l1 = _dist(p1_coords, l1_coords, u.dimensions)
    if not (10.0 < dp1l1 < 30.0):
        return False

    # angle A: P2-P1-L1
    theta_a = float(np.degrees(calc_angles(p2_coords, p1_coords, l1_coords)))
    if not boresch_chk.check_angle(theta_a):
        return False

    # angle B: P1-L1-L2
    theta_b = float(np.degrees(calc_angles(p1_coords, l1_coords, l2_coords)))
    if not boresch_chk.check_angle(theta_b):
        return False

    # # dihedral A: P3-P2-P1-L1
    # phi_a = float(np.degrees(
    #     calc_dihedrals(p3_coords, p2_coords, p1_coords, l1_coords)
    # ))

    # # dihedral B: P2-P1-L1-L2
    # phi_b = float(np.degrees(
    #     calc_dihedrals(p2_coords, p1_coords, l1_coords, l2_coords)
    # ))

    # # dihedral C: P1-L1-L2-L3
    # phi_c = float(np.degrees(
    #     calc_dihedrals(p1_coords, l1_coords, l2_coords, l3_coords)
    # ))

    # if not (
    #     boresch_chk.check_dihedral(phi_a) and
    #     boresch_chk.check_dihedral(phi_b) and
    #     boresch_chk.check_dihedral(phi_c)
    # ):
    #     return False

    # test all relevant sequential triplets for near-collinearity.
    if boresch_chk.is_collinear(
        np.array([p3_coords, p2_coords, p1_coords, l1_coords])
    ):
        return False

    if boresch_chk.is_collinear(
        np.array([p2_coords, p1_coords, l1_coords, l2_coords])
    ):
        return False

    if boresch_chk.is_collinear(
        np.array([p1_coords, l1_coords, l2_coords, l3_coords])
    ):
        return False

    # protein-anchor and P3 separation checks.
    dp1p3 = _dist(p1_coords, p3_coords, u.dimensions)
    dp2p3 = _dist(p2_coords, p3_coords, u.dimensions)
    dl1p3 = _dist(l1_coords, p3_coords, u.dimensions)

    if not (
        min_distance < dp1p3 < max_distance and
        min_distance < dp2p3 < max_distance and
        min_distance < dl1p3 < max_distance
    ):
        return False

    return True

def conditions_met(u, lig_names, prot_triplet, lig_segid):
    return check_ligand_order(u, lig_names, prot_triplet, lig_segid)
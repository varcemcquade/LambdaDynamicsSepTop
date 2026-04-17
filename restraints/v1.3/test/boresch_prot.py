import numpy as np
import MDAnalysis as mda
from MDAnalysis.topology import guessers
from MDAnalysis.analysis import dssp
from MDAnalysis.lib.distances import distance_array, calc_angles, calc_dihedrals
import boresch_chk


# ---------------------------------------------------------------------------
# Internal distance helper
# ---------------------------------------------------------------------------

def dist(a, b, box):
    """
    Minimum-image distance between two atoms.

    :param a: Coordinates of atom a, shape (3,).
    :param b: Coordinates of atom b, shape (3,).
    :param box: MDAnalysis box dimensions array.
    :return: Float distance in Angstroms.
    """
    return float(distance_array(a.reshape(1, 3), b.reshape(1, 3), box=box)[0, 0])


# ---------------------------------------------------------------------------
# Protein atom selection
# ---------------------------------------------------------------------------

def select_protein_atoms(u, l1_coords, box_size=84, box_angle=90,
                         skip_start=15, skip_end=15,
                         min_len_H=8, min_len_E=5,
                         trim_H=3, trim_E=2):
    """
    First-pass selection of candidate protein atoms for Boresch restraints.

    Criteria applied in order:
      1. C-alpha or C-beta atoms only.
      2. Residue is in the *core* of a helix (H) or beta-strand (E) — not
         near chain termini and not near secondary-structure ends.
      3. Distance from L1 is 10 Å < d < 30 Å.

    BUG NOTE (fixed here):  previous versions extended ``core_resids`` with
    raw array-index values from ``range(start, end)`` instead of the actual
    residue IDs at those positions.  With non-trivial starting resids (e.g.
    resid 2) this produced an off-by-N mapping.  The fix is to look up actual
    resids via ``sorted_backbone_resids[s3:e3]``.

    :param u:            MDAnalysis Universe (protein-ligand complex).  Must
                         have ``u.dimensions`` set before calling.
    :param l1_coords:    Coordinates of L1 atom, shape (1, 3) or (3,).
    :param box_size:     Cubic box side length in Angstroms.
    :param box_angle:    Box angle in degrees.
    :param skip_start:   Skip residues within this many positions of the chain
                         N-terminus (in the DSSP-ordered residue list).
    :param skip_end:     Skip residues within this many positions of the chain
                         C-terminus.
    :param min_len_H:    Minimum run length (residues) for a helix to be used.
    :param min_len_E:    Minimum run length for a beta strand.
    :param trim_H:       Trim this many residues from each end of a helix run.
    :param trim_E:       Trim this many residues from each end of a strand run.
    :return:             List of 0-based global atom indices passing all filters.
    """

    if not hasattr(u.atoms, "elements"):
        names = [str(n) for n in u.atoms.names]
        elems = [guessers.guess_atom_element(n) for n in names]
        u.add_TopologyAttr("elements", elems)

    u.dimensions = [box_size, box_size, box_size, box_angle, box_angle, box_angle]
    l1_coords = np.asarray(l1_coords).reshape(1, 3)

    # --- Secondary structure via DSSP ---
    sec_data = dssp.DSSP(u).run()

    # results.resids is 1-D (n_residues,); results.dssp is (n_frames, n_residues)
    dssp_resids = sec_data.results.resids
    dssp_sec    = sec_data.results.dssp[0]           # frame 0

    # Strip whitespace from the single-character codes for safe comparison
    sec_structs_by_resid = {
        int(r): str(s).strip()
        for r, s in zip(dssp_resids, dssp_sec)
    }

    backbone = u.select_atoms("backbone")
    # Use sorted unique backbone resids so contiguous runs are meaningful
    sorted_backbone_resids = sorted(set(int(r) for r in backbone.atoms.resids))
    n_res = len(sorted_backbone_resids)

    num_H = sum(1 for s in sec_structs_by_resid.values() if s == "H")
    num_E = sum(1 for s in sec_structs_by_resid.values() if s == "E")

    structs = ["H"] if num_H >= num_E else ["H", "E"]

    core_resids = []
    for struct in structs:
        min_len = min_len_H if struct == "H" else min_len_E
        trim    = trim_H    if struct == "H" else trim_E

        # Build binary mask aligned with sorted_backbone_resids
        candidate_mask = [
            1 if sec_structs_by_resid.get(r, "") == struct else 0
            for r in sorted_backbone_resids
        ]

        padded = np.pad(candidate_mask, (1, 1))
        edges  = np.diff(padded)
        starts = np.where(edges ==  1)[0]   # inclusive start of run (0-based idx)
        ends   = np.where(edges == -1)[0]   # exclusive end of run

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
            # FIX: map array indices → actual residue IDs
            core_resids.extend(sorted_backbone_resids[s3:e3])

    if not core_resids:
        return []

    core_resids = sorted(set(core_resids))
    resids_str = " ".join(str(r) for r in core_resids)
    candidate_atoms = u.select_atoms(
        f"protein and (name CA CB) and (resid {resids_str})"
    )
    if candidate_atoms.n_atoms == 0:
        return []

    dists = distance_array(
        candidate_atoms.positions, l1_coords, box=u.dimensions
    )[:, 0]

    protein_atoms = [
        int(candidate_atoms.indices[i])
        for i, d in enumerate(dists)
        if 10.0 < d < 30.0
    ]
    return protein_atoms


# ---------------------------------------------------------------------------
# Triplet discovery
# ---------------------------------------------------------------------------

def find_triplets(u, protein_atoms, l1_name, l2_name, l3_name, lig_segid,
                  box_size=84):
    """
    Find all valid (P1, P2, P3) protein triplets for Boresch restraints.

    Validity requires that every angle, dihedral, and inter-atom distance in
    the six-atom restrained set {L1, L2, L3, P1, P2, P3} satisfies:
      - Angles not within 10 kT of 0° or 180°.
      - No three consecutive atoms are collinear (dot-product threshold 0.9).
      - Dihedrals not within ±150° of ±180°.
      - Protein–protein and protein–L1 distances between 5 Å and half the box.

    Lig atoms are located by *name* (from ``lig_segid``), so the function
    works correctly even when atom numbers differ across complexes.

    :param u:             MDAnalysis Universe with ``u.dimensions`` set.
    :param protein_atoms: List of 0-based protein atom indices (candidates).
    :param l1_name, l2_name, l3_name: Ligand atom names.
    :param lig_segid:     Ligand segment ID.
    :param box_size:      Cubic box side length in Angstroms.
    :return:              List of [p1, p2, p3] triplets with *1-based* indices.
    """

    u.dimensions = [box_size, box_size, box_size, 90, 90, 90]
    coords = u.atoms.positions.copy()

    l1_coords = u.select_atoms(f"segid {lig_segid} and name {l1_name}")[0].position
    l2_coords = u.select_atoms(f"segid {lig_segid} and name {l2_name}")[0].position
    l3_coords = u.select_atoms(f"segid {lig_segid} and name {l3_name}")[0].position

    min_distance = 5.0
    max_distance = (u.dimensions[0] / 2.0) if u.dimensions[0] > 0 else 20.0
    all_triplets = []

    for p1 in protein_atoms:
        p1_coords = coords[p1]

        # --- P1 checks ---
        a1 = np.degrees(calc_angles(p1_coords, l1_coords, l2_coords))
        if not boresch_chk.check_angle(a1):
            continue

        dih1 = np.degrees(calc_dihedrals(p1_coords, l1_coords, l2_coords, l3_coords))
        if abs(dih1) >= 150.0:
            continue

        if boresch_chk.is_collinear(
            np.array([p1_coords, l1_coords, l2_coords, l3_coords])
        ):
            continue

        # --- P2 candidates ---
        valid_p2s = []
        for p2 in protein_atoms:
            if p2 == p1:
                continue

            p2_coords = coords[p2]
            dp1p2 = dist(p1_coords, p2_coords, u.dimensions)
            if not (min_distance < dp1p2 < max_distance):
                continue

            a2 = np.degrees(calc_angles(p2_coords, p1_coords, l1_coords))
            if not boresch_chk.check_angle(a2):
                continue

            dih2 = np.degrees(
                calc_dihedrals(p2_coords, p1_coords, l1_coords, l2_coords)
            )
            if abs(dih2) >= 150.0:
                continue

            if boresch_chk.is_collinear(
                np.array([p2_coords, p1_coords, l1_coords, l2_coords])
            ):
                continue

            valid_p2s.append(p2)

        # --- P3 candidates for each valid P2 ---
        for p2 in valid_p2s:
            p2_coords = coords[p2]
            p3_candidates      = []
            distance_products  = []

            for p3 in protein_atoms:
                if p3 in (p1, p2):
                    continue

                p3_coords = coords[p3]

                dih3 = np.degrees(
                    calc_dihedrals(p3_coords, p2_coords, p1_coords, l1_coords)
                )
                if abs(dih3) >= 150.0:
                    continue

                if boresch_chk.is_collinear(
                    np.array([p3_coords, p2_coords, p1_coords, l1_coords])
                ):
                    continue

                dp1p3 = dist(p1_coords, p3_coords, u.dimensions)
                dp2p3 = dist(p2_coords, p3_coords, u.dimensions)
                dl1p3 = dist(l1_coords, p3_coords, u.dimensions)

                if (min_distance < dp1p3 < max_distance
                        and min_distance < dp2p3 < max_distance
                        and min_distance < dl1p3 < max_distance):
                    p3_candidates.append(p3)
                    distance_products.append(dp1p3 * dp2p3)

            if p3_candidates:
                best_idx = int(np.argmax(distance_products))
                best_p3  = p3_candidates[best_idx]
                # Return 1-based indices for downstream PSF lookup
                all_triplets.append([p1 + 1, p2 + 1, best_p3 + 1])

    return all_triplets


# ---------------------------------------------------------------------------
# Geometry check for a candidate triplet in a different complex
# ---------------------------------------------------------------------------

def conditions_met(u, lig_names, prot_triplet, lig_segid, box_size=84):
    """
    Test whether a (P1, P2, P3) triplet found for the reference complex
    still satisfies Boresch geometry conditions in a different complex.

    :param u:            MDAnalysis Universe of the target complex
                         (``u.dimensions`` should already be set).
    :param lig_names:    Tuple/list (l1_name, l2_name, l3_name).
    :param prot_triplet: [p1, p2, p3] with *1-based* protein atom indices.
    :param lig_segid:    Ligand segment ID.
    :param box_size:     Cubic box side length in Angstroms.
    :return:             True if all conditions are satisfied.
    """

    u.dimensions = [box_size, box_size, box_size, 90, 90, 90]
    coords = u.atoms.positions.copy()

    l1_name, l2_name, l3_name = lig_names
    # prot_triplet uses 1-based indexing → convert to 0-based
    p1, p2, p3 = prot_triplet[0] - 1, prot_triplet[1] - 1, prot_triplet[2] - 1

    l1_coords = u.select_atoms(f"segid {lig_segid} and name {l1_name}")[0].position
    l2_coords = u.select_atoms(f"segid {lig_segid} and name {l2_name}")[0].position
    l3_coords = u.select_atoms(f"segid {lig_segid} and name {l3_name}")[0].position
    p1_coords = coords[p1]
    p2_coords = coords[p2]
    p3_coords = coords[p3]

    min_distance = 5.0
    max_distance = (u.dimensions[0] / 2.0) if u.dimensions[0] > 0 else 20.0

    # P1–L1–L2 angle
    a1 = np.degrees(calc_angles(p1_coords, l1_coords, l2_coords))
    if not boresch_chk.check_angle(a1):
        return False

    # P1–L1–L2–L3 dihedral
    dih1 = np.degrees(calc_dihedrals(p1_coords, l1_coords, l2_coords, l3_coords))
    if abs(dih1) >= 150.0:
        return False

    if boresch_chk.is_collinear(
        np.array([p1_coords, l1_coords, l2_coords, l3_coords])
    ):
        return False

    # P2–P1 distance
    dp1p2 = dist(p1_coords, p2_coords, u.dimensions)
    if not (min_distance < dp1p2 < max_distance):
        return False

    # P2–P1–L1 angle
    a2 = np.degrees(calc_angles(p2_coords, p1_coords, l1_coords))
    if not boresch_chk.check_angle(a2):
        return False

    # P2–P1–L1–L2 dihedral
    dih2 = np.degrees(calc_dihedrals(p2_coords, p1_coords, l1_coords, l2_coords))
    if abs(dih2) >= 150.0:
        return False

    if boresch_chk.is_collinear(
        np.array([p2_coords, p1_coords, l1_coords, l2_coords])
    ):
        return False

    # P3–P2–P1–L1 dihedral
    dih3 = np.degrees(calc_dihedrals(p3_coords, p2_coords, p1_coords, l1_coords))
    if abs(dih3) >= 150.0:
        return False

    if boresch_chk.is_collinear(
        np.array([p3_coords, p2_coords, p1_coords, l1_coords])
    ):
        return False

    # Pairwise distance checks
    dp1p3 = dist(p1_coords, p3_coords, u.dimensions)
    dp2p3 = dist(p2_coords, p3_coords, u.dimensions)
    dl1p3 = dist(l1_coords, p3_coords, u.dimensions)

    if not (min_distance < dp1p3 < max_distance
            and min_distance < dp2p3 < max_distance
            and min_distance < dl1p3 < max_distance):
        return False

    return True

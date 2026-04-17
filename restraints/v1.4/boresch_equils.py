import numpy as np
from MDAnalysis.lib.distances import distance_array, calc_angles, calc_dihedrals

def compute_equils(u, lig_names, prot_triplet, lig_segid):
    """
    Compute equilibrium values for the six Boresch restraint terms.

    The six Boresch degrees of freedom are:
      - d      : distance P1-L1 (Angstroms)
      - thetaA : angle    P2-P1-L1 (degrees)
      - thetaB : angle    P1-L1-L2 (degrees)
      - phiA   : dihedral P3-P2-P1-L1 (degrees)
      - phiB   : dihedral P2-P1-L1-L2 (degrees)
      - phiC   : dihedral P1-L1-L2-L3 (degrees)

    :param u: MDAnalysis Universe of the complex. u.dimensions should already be set before calling.
    :param lig_names: Tuple/list (l1_name, l2_name, l3_name) -- atom names within the ligand segment.
    :param prot_triplet: [p1, p2, p3] with *0-based* protein atom indices.
    :param lig_segid: Ligand segment ID (e.g. "HETA").
    :return: Tuple (d, thetaA, thetaB, phiA, phiB, phiC).
    """
    l1_name, l2_name, l3_name = lig_names
    p1_idx, p2_idx, p3_idx = prot_triplet[0], prot_triplet[1], prot_triplet[2]

    l1_coords = u.select_atoms(f"segid {lig_segid} and name {l1_name}")[0].position
    l2_coords = u.select_atoms(f"segid {lig_segid} and name {l2_name}")[0].position
    l3_coords = u.select_atoms(f"segid {lig_segid} and name {l3_name}")[0].position
    p1_coords = u.atoms[p1_idx].position
    p2_coords = u.atoms[p2_idx].position
    p3_coords = u.atoms[p3_idx].position

    d = _dist(l1_coords, p1_coords, u.dimensions)
    thetaA = float(np.degrees(calc_angles(l1_coords, p1_coords, p2_coords)))
    thetaB = float(np.degrees(calc_angles(l2_coords, l1_coords, p1_coords)))
    phiA = float(np.degrees(calc_dihedrals(l1_coords, p1_coords, p2_coords, p3_coords)))
    phiB = float(np.degrees(calc_dihedrals(l2_coords, l1_coords, p1_coords, p2_coords)))
    phiC = float(np.degrees(calc_dihedrals(l3_coords, l2_coords, l1_coords, p1_coords)))

    return d, thetaA, thetaB, phiA, phiB, phiC

def _dist(a, b, box):
    return float(distance_array(a.reshape(1, 3), b.reshape(1, 3), box=box)[0, 0])
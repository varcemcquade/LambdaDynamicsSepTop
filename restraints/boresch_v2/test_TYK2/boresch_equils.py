import numpy as np
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals

def compute_dist_angle_dih(u, boresch_atoms):
    """ Compute distance, angles, dihedrals.
    :param u:
        MDAnalysis universe
    :param boresch_atoms:
         Array of boresch restraint atoms ordered l1, l2, l3, p1, p2, p3
    :returns dl1p1, theta1, theta2, phi1, phi2, phi3
        Values for distance (l1 to p1), theta1 (l1, p1, p2), theta2 (l2, l1, p1), phi1 (l1, p1, p2, p3),
        phi2 (l2, l1, p1, p2), and phi3 (l3, l2, l1, p1)
    """

    # 1-based indexing ---> 0-based indexing
    l1 = boresch_atoms[0] - 1
    l2 = boresch_atoms[1] - 1
    l3 = boresch_atoms[2] - 1
    p1 = boresch_atoms[3] - 1
    p2 = boresch_atoms[4] - 1
    p3 = boresch_atoms[5] - 1

    l1_coords = u.atoms[l1].position
    l2_coords = u.atoms[l2].position
    l3_coords = u.atoms[l3].position
    p1_coords = u.atoms[p1].position
    p2_coords = u.atoms[p2].position
    p3_coords = u.atoms[p3].position

    dl1p1 = float(calc_bonds(l1_coords, p1_coords, box=u.dimensions))
    theta1 = float(np.degrees(calc_angles(l1_coords, p1_coords, p2_coords)))
    theta2 = float(np.degrees(calc_angles(l2_coords, l1_coords, p1_coords)))
    phi1 = float(np.degrees(calc_dihedrals(l1_coords, p1_coords, p2_coords, p3_coords)))
    phi2 = float(np.degrees(calc_dihedrals(l2_coords, l1_coords, p1_coords, p2_coords)))
    phi3 = float(np.degrees(calc_dihedrals(l3_coords, l2_coords, l1_coords, p1_coords)))

    return dl1p1, theta1, theta2, phi1, phi2, phi3



import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array, calc_angles, calc_dihedrals

def compute_dist_angle_dih(complex_psf, complex_coords, boresch_atoms, box_size=84):
    """ Compute distance, angles, dihedrals.
    :param complex_psf:
        psf file of complex
    :param complex_coords:
        Coordinates of complex atoms
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

    u = mda.Universe(complex_psf, complex_coords)
    u.dimensions = [box_size, box_size, box_size, 90, 90, 90]

    l1_coords = u.atoms[l1].position
    l2_coords = u.atoms[l2].position
    l3_coords = u.atoms[l3].position
    p1_coords = u.atoms[p1].position
    p2_coords = u.atoms[p2].position
    p3_coords = u.atoms[p3].position

    dl1p1 = dist(l1_coords, p1_coords, u.dimensions)
    theta1 = np.degrees(calc_angles(l1_coords, p1_coords, p2_coords))
    theta2 = np.degrees(calc_angles(l2_coords, l1_coords, p1_coords))
    phi1 = np.degrees(calc_dihedrals(l1_coords, p1_coords, p2_coords, p3_coords))
    phi2 = np.degrees(calc_dihedrals(l2_coords, l1_coords, p1_coords, p2_coords))
    phi3 = np.degrees(calc_dihedrals(l3_coords, l2_coords, l1_coords, p1_coords))

    return dl1p1, theta1, theta2, phi1, phi2, phi3

def dist(a, b, box):
    return float(distance_array(a.reshape(1,3), b.reshape(1,3), box=box)[0,0])


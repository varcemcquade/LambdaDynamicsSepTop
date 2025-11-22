import numpy as np

FORCE_CONST = 83.68
R = 8.31445985 * 0.001  # kJ/mol/K
T = 298.15
RT = R*T
ENERGY_THRESHOLD_KT = 10

def check_angle(angle):
    # check if angle is <10kT from 0 or 180
    check1 = 0.5 * FORCE_CONST * ((angle - 0.0) / 180.0 * np.pi)**2
    check2 = 0.5 * FORCE_CONST * ((angle - 180.0) / 180.0 * np.pi)**2
    ang_check_1 = check1 / RT
    ang_check_2 = check2 / RT
    if ang_check_1 < ENERGY_THRESHOLD_KT or ang_check_2 < ENERGY_THRESHOLD_KT:
        return False
    return True

def is_collinear(positions, atoms, threshold=0.9):
    """ Report whether any sequential vectors in a sequence of atoms are collinear
    :param positions:
        n_atoms x 3 simtk.unit.Quantity -- reference positions to use for imposing restraints (units of length)
    :param atoms:
        Iterable of int -- the indices of the atoms to test
    :param threshold:
        float, optional, default=0.9 -- atoms are not collinear if their sequential vector separation dot
        products are less than threshold
    :returns result:
        Bool, returns True if any sequential pair of vectors is collinear and False otherwise.
    """
    for i in range(len(atoms) - 2):
        v1 = positions[atoms[i + 1], :] - positions[atoms[i], :]
        v2 = positions[atoms[i + 2], :] - positions[atoms[i + 1], :]
        normalized_inner_product = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))

        if np.abs(normalized_inner_product) > threshold:
            return True

    return False
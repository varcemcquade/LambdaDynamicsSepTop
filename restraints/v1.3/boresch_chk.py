import numpy as np

force_const = 83.68     # kJ/mol/rad^2
R = 8.31445985 * 0.001  # kJ/mol/K
T = 298.15              # K
RT = R*T                # kJ/mol

def check_angle(angle):
    """
    Checks if a given angle is too close (<10 kT) to 0 or 180 degrees
    given harmonic restraint angle potential.
    :param angle:
        Float
    :return:
        Boolean
    """

    u_low = 0.5 * force_const * (np.pi * (angle - 0.0) / 180.0) ** 2
    u_high = 0.5 * force_const * (np.pi * (angle - 180.0) / 180.0) ** 2
    u_low_reduced = u_low / RT
    u_high_reduced = u_high / RT
    if u_low_reduced < 10.0 or u_high_reduced < 10.0:
        return False

    return True

def is_collinear(positions, threshold=0.9):
    """
    Checks if atoms are collinear.
    :param positions:
        Array of coordinates sized n_atoms x 3
    :param threshold:
        Threshold for what constitutes collinear
        For threshold=0.9, ~26 degrees < angle < ~154 degrees
    :return:
        Boolean
    """

    for i in range(len(positions) - 2):
        v1 = positions[i] - positions[i + 1]
        v2 = positions[i + 1] - positions[i + 2]
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)
        angle = np.dot(v1, v2) / (v1_mag * v2_mag)
        if np.abs(angle) > threshold:
            return True

    return False
import numpy as np

def return_atom_info(complex_psf, atoms_idx, lig_segid, prot_segid):
    """ Return atom info for writing .str
     :param complex_psf:
         psf file of complex
     :param atoms_idx:
         Array of boresch restraint atoms ordered l1, l2, l3, p1, p2, p3 - ZERO-BASED
     :param lig_segid, prot_segid:
         Ligand and protein segment IDs
     :returns distance_l1_p1, theta1, theta2, phi1, phi2, phi3
         Values for distance (l1 to p1), theta1 (l1, p1, p2), theta2 (l2, l1, p1), phi1 (l1, p1, p2, p3),
         phi2 (l2, l1, p1, p2), and phi3 (l3, l2, l1, p1)
     """
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

    # Convert to 1-based for PSF lookup
    l1 = f"{atoms_idx[0] + 1} {lig_segid}"
    l2 = f"{atoms_idx[1] + 1} {lig_segid}"
    l3 = f"{atoms_idx[2] + 1} {lig_segid}"
    p1 = f"{atoms_idx[3] + 1} {prot_segid}"
    p2 = f"{atoms_idx[4] + 1} {prot_segid}"
    p3 = f"{atoms_idx[5] + 1} {prot_segid}"


    with open(complex_psf) as file:
        for line in file:
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
    atom_info = [l1name, l2name, l3name, p1resid, p1resnum, p1name, p2resid, p2resnum, p2name, p3resid, p3resnum, p3name]
    return atom_info

def write_boresch_variables(complex_psf, atoms_idx, equils, lig_segid, prot_segid, n):
    """ Write boresch_variables{n}.str
     :param complex_psf:
         psf file of complex
     :param atoms_idx:
         Array of boresch restraint atoms ordered l1, l2, l3, p1, p2, p3
     :param equils:
         Equilibrium conditions of boresch restraint atoms
     :param lig_segid, prot_segid:
         Ligand and protein segment IDs
     :param n:
         Ligand number
     """

    d = equils[0] # L1-P1, angstroms
    theta1 = equils[1] # L1-P1-P2, degrees
    theta2 = equils[2] # L2-L1-P2, degrees
    phi1 = equils[3]  # L1-P1-P2-P3 dihedral, degrees
    phi2 = equils[4] # L2-L1-P1-P2 dihedral, degrees
    phi3 = equils[5] # L3-L2-L1-P1 dihedral, degrees

    dk = 20.0 # kcal/molA**2
    theta1k = 1.6 * d**2 # kcal/mol*rad**2, initializer 5 A = 40 kcal/mol*rad**2
    theta2k = 20.0 # kcal/mol*rad**2
    phi1k = (d * np.sin(np.deg2rad(theta1)))**2 # kcal/mol*rad**2
    phi23k = 20.0 # kcal/mol*rad**2

    atom_info = return_atom_info(complex_psf, atoms_idx, lig_segid, prot_segid)

    file = open(f"boresch_variables{n}.str", "w")
    file.write(f"SET LIGSEGID = {lig_segid}\n\n")
    file.write(f"SET L1NAME = {atom_info[0]}\n")
    file.write(f"SET L2NAME = {atom_info[1]}\n")
    file.write(f"SET L3NAME = {atom_info[2]}\n\n")
    file.write(f"SET PROTSEGID = {prot_segid}\n\n")
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
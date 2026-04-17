import numpy as np

def return_atom_info(complex_psf, atoms_idx, lig_segid, prot_segid):
    """
    Return atom info for writing stream file.
     :param complex_psf:
         psf file of complex
     :param atoms_idx:
         Array of boresch restraint atoms ordered l1, l2, l3, p1, p2, p3
     :param lig_segid, prot_segid:
         Ligand and protein segment IDs
     :returns distance, thetaA, thetaB, phiA, phiB, phiC
         Values for distance (l1 to p1), thetaA (l1, p1, p2), thetaB (l2, l1, p1), phiA (l1, p1, p2, p3),
         phiB (l2, l1, p1, p2), and phiC (l3, l2, l1, p1)
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

    l1 = f"{atoms_idx[0]} {lig_segid}"
    l2 = f"{atoms_idx[1]} {lig_segid}"
    l3 = f"{atoms_idx[2]} {lig_segid}"
    p1 = f"{atoms_idx[3]} {prot_segid}"
    p2 = f"{atoms_idx[4]} {prot_segid}"
    p3 = f"{atoms_idx[5]} {prot_segid}"

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

    atom_info = [l1name, l2name, l3name,
                 p1resid, p1resnum, p1name,
                 p2resid, p2resnum, p2name,
                 p3resid, p3resnum, p3name]

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
    thetaA = equils[1] # L1-P1-P2, degrees
    thetaB = equils[2] # L2-L1-P2, degrees
    phiA = equils[3]  # L1-P1-P2-P3 dihedral, degrees
    phiB = equils[4] # L2-L1-P1-P2 dihedral, degrees
    phiC = equils[5] # L3-L2-L1-P1 dihedral, degrees

    dk = 20.0 # kcal/molA**2
    thetaAk = 1.6 * d**2 # kcal/mol*rad**2, initializer 5 A = 40 kcal/mol*rad**2
    thetaBk = 20.0 # kcal/mol*rad**2
    phiAk = (d * np.sin(np.deg2rad(thetaA)))**2 # kcal/mol*rad**2
    phiBk = 20.0 # kcal/mol*rad**2
    phiCk = 20.0  # kcal/mol*rad**2

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
    file.write(f"SET DISTANCE = {d}\n")
    file.write(f"SET THETAA = {thetaA}\n")
    file.write(f"SET THETAB = {thetaB}\n")
    file.write(f"SET PHIA = {phiA}\n")
    file.write(f"SET PHIB = {phiB}\n")
    file.write(f"SET PHIC = {phiC}\n\n")
    file.write(f"SET DISTANCEK = {dk}\n")
    file.write(f"SET THETAAK = {thetaAk}\n")
    file.write(f"SET THETABK = {thetaBk}\n")
    file.write(f"SET PHIAK = {phiAk}\n")
    file.write(f"SET PHIBK = {phiBk}\n")
    file.write(f"SET PHICK = {phiCk}\n")
    file.close()

    return None
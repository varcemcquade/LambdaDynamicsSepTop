import numpy as np

def return_atom_info(psf, lig_names, prot_atoms_idx, prot_segid):
    """
    Return atom info for writing stream file.
    Ligand atom names are passed directly; only protein atoms are looked up in the PSF.
    Works with either a protein-only PSF or a complex PSF.

     :param psf: PSF file (protein-only or complex).
     :param lig_names: Tuple (l1_name, l2_name, l3_name).
     :param prot_atoms_idx: 1-based indices [p1, p2, p3] for PSF lookup.
     :param prot_segid: Protein segment ID.
     """
    p1resid = ""
    p1resnum = ""
    p1name = ""
    p2resid = ""
    p2resnum = ""
    p2name = ""
    p3resid = ""
    p3resnum = ""
    p3name = ""

    p1 = f"{prot_atoms_idx[0]} {prot_segid}"
    p2 = f"{prot_atoms_idx[1]} {prot_segid}"
    p3 = f"{prot_atoms_idx[2]} {prot_segid}"

    with open(psf) as file:
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
            if "!NBOND" in line:
                break

    atom_info = [lig_names[0], lig_names[1], lig_names[2],
                 p1resid, p1resnum, p1name,
                 p2resid, p2resnum, p2name,
                 p3resid, p3resnum, p3name]

    return atom_info

def write_boresch_variables(psf, lig_names, prot_atoms_idx, equils, lig_segid, prot_segid, n, strong=True):
    """ Write boresch_variables{n}.str
     :param psf: Protein (or complex) PSF file.
     :param lig_names: Tuple (l1_name, l2_name, l3_name).
     :param prot_atoms_idx: 1-based indices [p1, p2, p3] for PSF lookup.
     :param equils: Equilibrium values (d, thetaA, thetaB, phiA, phiB, phiC).
     :param lig_segid: Ligand segment ID.
     :param prot_segid: Protein segment ID.
     :param n: Ligand number (used in output filename).
     :param strong: If True, use 5x standard force constants; otherwise use 1x.
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

    if strong:
        dk *= 5.0
        thetaAk *= 5.0
        thetaBk *= 5.0
        phiAk *= 5.0
        phiBk *= 5.0
        phiCk *= 5.0

    atom_info = return_atom_info(psf, lig_names, prot_atoms_idx, prot_segid)

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
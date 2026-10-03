import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_dihedrals

lig1 = "mol1"
lig2 = "mol2"
lig3 = "mol3"

lig1_pdb = lig1 + ".pdb"
lig2_pdb = lig2 + ".pdb"
lig3_pdb = lig3 + ".pdb"

u1 = mda.Universe(lig1_pdb)
u2 = mda.Universe(lig2_pdb)
u3 = mda.Universe(lig3_pdb)

all_atom_lists = [
    [["C3", "C4", "C7", "N1"], ["C4", "C7", "N1", "C8"], ["C7", "N1", "C8", "C12"]],
    [["C3", "C4", "C7", "N1"], ["C4", "C7", "N1", "C8"], ["C7", "N1", "C8", "C12"]],
    [["C3", "C4", "C7", "N1"], ["C4", "C7", "N1", "C8"], ["C7", "N1", "C8", "C12"]]
]


def get_dihedral(u, atom_lists):
    dihs = []

    for names in atom_lists:
        coords = []
        for name in names:
            coords.append(u.select_atoms(f"name {name}")[0].position)
        dih = np.degrees(calc_dihedrals(coords[0], coords[1], coords[2], coords[3]))
        dihs.append(dih)
    return dihs

def write_internal_restraints(dih1_names, dih2_names, dih3_names, dihs, lig_resid):
    dihk = 5.0  # kcal/mol*rad**2

    file = open(f"internal_vars{lig_resid}.str", "w")
    file.write(f"SET DIH1L1NAME = {dih1_names[0]}\n")
    file.write(f"SET DIH1L2NAME = {dih1_names[1]}\n")
    file.write(f"SET DIH1L3NAME = {dih1_names[2]}\n")
    file.write(f"SET DIH1L4NAME = {dih1_names[3]}\n")
    file.write(f"SET INTERNALPHI1 = {dihs[0]}\n\n")

    file.write(f"SET DIH2L1NAME = {dih2_names[0]}\n")
    file.write(f"SET DIH2L2NAME = {dih2_names[1]}\n")
    file.write(f"SET DIH2L3NAME = {dih2_names[2]}\n")
    file.write(f"SET DIH2L4NAME = {dih2_names[3]}\n")
    file.write(f"SET INTERNALPHI2 = {dihs[1]}\n\n")

    file.write(f"SET DIH3L1NAME = {dih3_names[0]}\n")
    file.write(f"SET DIH3L2NAME = {dih3_names[1]}\n")
    file.write(f"SET DIH3L3NAME = {dih3_names[2]}\n")
    file.write(f"SET DIH3L4NAME = {dih3_names[3]}\n")
    file.write(f"SET INTERNALPHI3 = {dihs[2]}\n\n")

    file.write(f"SET INTERNALPHIK = {dihk}\n")
    file.close()
    return None

lig1dih1_names = all_atom_lists[0][0]
lig1dih2_names = all_atom_lists[0][1]
lig1dih3_names = all_atom_lists[0][2]

lig2dih1_names = all_atom_lists[1][0]
lig2dih2_names = all_atom_lists[1][1]
lig2dih3_names = all_atom_lists[1][2]

lig3dih1_names = all_atom_lists[2][0]
lig3dih2_names = all_atom_lists[2][1]
lig3dih3_names = all_atom_lists[2][2]

lig1dihs = get_dihedral(u1, all_atom_lists[0])
lig2dihs = get_dihedral(u2, all_atom_lists[0])
lig3dihs = get_dihedral(u3, all_atom_lists[0])

write_internal_restraints(lig1dih1_names, lig1dih2_names, lig1dih3_names, lig1dihs, 1)
write_internal_restraints(lig2dih1_names, lig2dih2_names, lig2dih3_names, lig2dihs, 2)
write_internal_restraints(lig3dih1_names, lig3dih2_names, lig3dih3_names, lig3dihs, 3)
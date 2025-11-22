import sys
from boresch_restraints import select_ligand_atoms, select_protein_atoms, select_boresch_atoms, compute_dist_angle_dih, write_boresch_variables

if len(sys.argv) != 7:
    print("Error: Invalid number of arguments.")
    sys.exit(1)

complexA_psf = sys.argv[1]
complexA_dcd = sys.argv[2]
complexB_psf = sys.argv[3]
complexB_dcd = sys.argv[4]
ligand_segid = sys.argv[5]
protein_segid = sys.argv[6]

l1A, l2A, l3A = select_ligand_atoms(complexA_psf, complexA_dcd, ligand_segid) # 1-based indices
l1B, l2B, l3B = select_ligand_atoms(complexB_psf, complexB_dcd, ligand_segid) # 1-based indices

all_protA = set(select_protein_atoms(complexA_psf, complexA_dcd, l1A))
all_protB = set(select_protein_atoms(complexB_psf, complexB_dcd, l1B))

overlap = list(all_protA & all_protB)

all_protA_combos = select_boresch_atoms(complexA_psf, complexA_dcd, overlap, l1A, l2A, l3A)
all_protB_combos = select_boresch_atoms(complexB_psf, complexB_dcd, overlap, l1B, l2B, l3B)

p1 = 0
p2 = 0
p3 = 0

for i in all_protA_combos:
    if i in all_protB_combos:
        p1 = i[0]
        p2 = i[1]
        p3 = i[2]
        break

boresch_finalA = [p1, p2, p3, l1A, l2A, l3A]
boresch_finalB = [p1, p2, p3, l1B, l2B, l3B]
equilsA = compute_dist_angle_dih(complexA_psf, complexA_dcd, boresch_finalA)
equilsB = compute_dist_angle_dih(complexB_psf, complexB_dcd, boresch_finalB)

write_boresch_variables(complexA_psf, boresch_finalA, equilsA, ligand_segid, protein_segid, 1)
write_boresch_variables(complexB_psf, boresch_finalB, equilsB, ligand_segid, protein_segid, 2)
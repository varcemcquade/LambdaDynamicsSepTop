import boresch_lig
import boresch_prot
import boresch_equils
import boresch_wrt
import MDAnalysis as mda

# Define System and File Variables
file = "complexes.txt"
complexes = []
lig_idx = []                        # List of lists of l1, l2, l3
prots_final = []                    # p1, p2, p3 for all complexes

lig_segid = "HETA"
prot_segid = "PROA"
box_size = 84

# Read "complexes.txt"
fp = open(file, 'r')
for line in fp:
    complexes.append(line.rstrip())
fp.close()

ref_psf = complexes[0]+".psf"
ref_pdb = complexes[0]+".pdb"
lig_idx.append(boresch_lig.select_ligand_atoms(ref_psf, ref_pdb, lig_segid))
ref_prot_candidates = boresch_prot.select_protein_atoms(ref_psf, ref_pdb, lig_idx[0][0], box_size)
prot_triplets_ref = boresch_prot.find_triplets(ref_psf, ref_pdb, ref_prot_candidates, lig_idx[0][0], lig_idx[0][1], lig_idx[0][2], box_size)
candidate_triplets = prot_triplets_ref.copy()
final_triplets = []

for i, complex in enumerate(complexes[1:], start = 1):
    psf = complex+".psf"
    pdb = complex+".pdb"
    lig_atoms = boresch_lig.select_ligand_atoms(psf, pdb, lig_segid)
    lig_idx.append(lig_atoms)

    local_candidates = []
    u = mda.Universe(psf, pdb)
    u.dimensions = [box_size, box_size, box_size, 90, 90, 90]
    for t in candidate_triplets:
        if boresch_prot.conditions_met(u, lig_atoms, t):
            local_candidates.append(t)
    candidate_triplets = local_candidates
    print(f"All triplets evaluated against {complex}. Remaining: {len(candidate_triplets)}")

if candidate_triplets:
    prots_final = candidate_triplets[0]
else:
    print("No triplets found across all complexes.")

for i, complex in enumerate(complexes):
    psf = complex + ".psf"
    pdb = complex + ".pdb"
    boresch_atoms = lig_idx[i]+prots_final
    equils = boresch_equils.compute_dist_angle_dih(psf, pdb, boresch_atoms, box_size)
    boresch_wrt.write_boresch_variables(psf, boresch_atoms, equils, lig_segid, prot_segid, i+1)

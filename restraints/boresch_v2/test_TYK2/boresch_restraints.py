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
with open(file, 'r') as fp:
    for line in fp:
        complexes.append(line.rstrip())

ref_psf = f"{complexes[0]}.psf"
ref_pdb = f"{complexes[0]}.pdb"

u_ref = mda.Universe(ref_psf, ref_pdb)
u_ref.dimensions = [box_size, box_size, box_size, 90, 90, 90]
ref_prot_select = u_ref.select_atoms("protein")

lig_idx.append(boresch_lig.select_ligand_atoms(u_ref, lig_segid))
ref_prot_candidates = boresch_prot.select_protein_atoms(u_ref, lig_idx[0][0], box_size)
prot_triplets_ref = boresch_prot.find_triplets(u_ref, ref_prot_candidates, lig_idx[0][0], lig_idx[0][1], lig_idx[0][2])
candidate_triplets = prot_triplets_ref.copy()

universes = [u_ref]

for i, complex in enumerate(complexes[1:], start=1):
    psf = f"{complex}.psf"
    pdb = f"{complex}.pdb"
    u = mda.Universe(psf, pdb)
    u.dimensions = [box_size, box_size, box_size, 90, 90, 90]
    universes.append(u)
    lig_atoms = boresch_lig.select_ligand_atoms(u, lig_segid)
    lig_idx.append(lig_atoms)

    ligand_select = u.select_atoms(f"segid {lig_segid}")
    new_u = mda.Merge(ref_prot_select, ligand_select)
    new_u.dimensions = [box_size, box_size, box_size, 90, 90, 90]

    local_candidates = []

    for t in candidate_triplets:
        if boresch_prot.conditions_met(new_u, lig_atoms, t):
            local_candidates.append(t)
    candidate_triplets = local_candidates
    print(f"All triplets evaluated against {complex}. Remaining: {len(candidate_triplets)}")

if candidate_triplets:
    prots_final = candidate_triplets[0]
else:
    print("No triplets found across all complexes.")

for i, complex in enumerate(complexes):
    psf = f"{complex}.psf"
    boresch_atoms = lig_idx[i]+prots_final
    u = universes[i]
    equils = boresch_equils.compute_dist_angle_dih(u, boresch_atoms)
    boresch_wrt.write_boresch_variables(psf, boresch_atoms, equils, lig_segid, prot_segid, i + 1)
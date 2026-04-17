"""
boresch_restraints.py
=====================
Main entry point for the SepTop Boresch restraint selection algorithm.

Workflow
--------
1.  Read a list of complex names from ``complexes.txt`` (one per line,
    no extension).  Each complex needs a matching ``.psf`` and ``.pdb``
    file in the same directory.
2.  Use the *first* complex as the reference to select L1/L2/L3 (by atom
    name) and find all valid (P1, P2, P3) protein triplets.
3.  Filter the triplet pool so only triplets that pass the Boresch geometry
    checks for *every* complex survive.
4.  Write a ``boresch_variables{n}.str`` CHARMM stream file for each complex.

Indexing convention
-------------------
* Ligand atoms are carried as **atom names** (strings) throughout, so they
  remain valid across complexes that share the same ligand topology but may
  have different atom numbering.
* Protein atoms use **1-based** integer indices (matching PSF line numbers)
  for output, and **0-based** indices only during internal geometry screening.
"""

import os
import sys
import warnings
import MDAnalysis as mda

import boresch_lig
import boresch_prot
import boresch_equils
import boresch_wrt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------

COMPLEXES_FILE = "complexes.txt"
LIG_SEGID      = "HETA"
PROT_SEGID     = "PROA"
BOX_SIZE       = 84          # Angstroms (cubic box assumed)

# ---------------------------------------------------------------------------
# Read complex list
# ---------------------------------------------------------------------------

if not os.path.isfile(COMPLEXES_FILE):
    sys.exit(f"ERROR: {COMPLEXES_FILE} not found in {os.getcwd()}")

with open(COMPLEXES_FILE) as fh:
    complexes = [line.strip() for line in fh if line.strip()]

if len(complexes) == 0:
    sys.exit("ERROR: complexes.txt is empty.")

print(f"Found {len(complexes)} complex(es): {complexes}")

# ---------------------------------------------------------------------------
# Reference complex: select ligand atoms and build protein triplet pool
# ---------------------------------------------------------------------------

ref_name = complexes[0]
ref_psf  = f"{ref_name}.psf"
ref_pdb  = f"{ref_name}.pdb"

print(f"\n--- Reference complex: {ref_name} ---")

ref_u = mda.Universe(ref_psf, ref_pdb)
ref_u.dimensions = [BOX_SIZE, BOX_SIZE, BOX_SIZE, 90, 90, 90]

# Select L1/L2/L3 for the reference complex (returns atom names)
ref_lig_names = boresch_lig.select_ligand_atoms(ref_u, LIG_SEGID)
l1_name, l2_name, l3_name = ref_lig_names
print(f"Ligand atoms selected: L1={l1_name}, L2={l2_name}, L3={l3_name}")

# Coordinates of L1 for protein candidate distance filter
l1_coords = ref_u.select_atoms(
    f"segid {LIG_SEGID} and name {l1_name}"
)[0].position.reshape(1, 3)

# First-pass protein atom filter
prot_candidates = boresch_prot.select_protein_atoms(
    ref_u, l1_coords, box_size=BOX_SIZE
)
print(f"Protein atom candidates (first pass): {len(prot_candidates)}")

if not prot_candidates:
    sys.exit(
        "ERROR: No protein candidate atoms found for the reference complex.\n"
        "Consider relaxing skip_start/skip_end or distance thresholds in "
        "boresch_prot.select_protein_atoms."
    )

# Find all valid protein triplets for the reference complex
print("Searching for valid protein triplets (this may take a moment)...")
candidate_triplets = boresch_prot.find_triplets(
    ref_u, prot_candidates, l1_name, l2_name, l3_name, LIG_SEGID,
    box_size=BOX_SIZE
)
print(f"Triplets found for reference complex: {len(candidate_triplets)}")

if not candidate_triplets:
    sys.exit(
        "ERROR: No valid protein triplets found for the reference complex.\n"
        "Try adjusting distance/angle thresholds or choosing a different "
        "reference complex."
    )

# ---------------------------------------------------------------------------
# Store per-complex ligand atom names (needed for output later)
# ---------------------------------------------------------------------------

# Index 0 is the reference complex
all_lig_names = [ref_lig_names]

# ---------------------------------------------------------------------------
# Filter triplets across remaining complexes
# ---------------------------------------------------------------------------

for cx_name in complexes[1:]:
    psf = f"{cx_name}.psf"
    pdb = f"{cx_name}.pdb"

    cx_u = mda.Universe(psf, pdb)
    cx_u.dimensions = [BOX_SIZE, BOX_SIZE, BOX_SIZE, 90, 90, 90]

    # Select THIS complex's ligand atoms (may differ from reference names)
    cx_lig_names = boresch_lig.select_ligand_atoms(cx_u, LIG_SEGID)
    all_lig_names.append(cx_lig_names)

    surviving = [
        t for t in candidate_triplets
        if boresch_prot.conditions_met(cx_u, cx_lig_names, t, LIG_SEGID,
                                       box_size=BOX_SIZE)
    ]
    candidate_triplets = surviving
    print(
        f"After {cx_name}: {len(candidate_triplets)} triplet(s) remaining."
    )

    if not candidate_triplets:
        break

if not candidate_triplets:
    sys.exit(
        "ERROR: No protein triplets survived geometry checks across all "
        "complexes.  Consider relaxing the dihedral threshold or the distance "
        "bounds in boresch_prot."
    )

# Use the first surviving triplet (all are geometrically valid)
prots_final = candidate_triplets[0]
p1, p2, p3 = prots_final
print(f"\nFinal protein triplet (1-based): P1={p1}, P2={p2}, P3={p3}")

# ---------------------------------------------------------------------------
# Write output for each complex
# ---------------------------------------------------------------------------

print("\n--- Writing output files ---")
for i, cx_name in enumerate(complexes):
    psf = f"{cx_name}.psf"
    pdb = f"{cx_name}.pdb"

    cx_u = mda.Universe(psf, pdb)
    cx_u.dimensions = [BOX_SIZE, BOX_SIZE, BOX_SIZE, 90, 90, 90]

    cx_lig_names = all_lig_names[i]

    # Compute equilibrium values for this complex
    equils = boresch_equils.compute_equils(
        cx_u, cx_lig_names, prots_final, LIG_SEGID, box_size=BOX_SIZE
    )

    # Convert ligand atom names → 1-based indices (for PSF lookup in boresch_wrt)
    l1_1b = int(cx_u.select_atoms(
        f"segid {LIG_SEGID} and name {cx_lig_names[0]}"
    )[0].index) + 1
    l2_1b = int(cx_u.select_atoms(
        f"segid {LIG_SEGID} and name {cx_lig_names[1]}"
    )[0].index) + 1
    l3_1b = int(cx_u.select_atoms(
        f"segid {LIG_SEGID} and name {cx_lig_names[2]}"
    )[0].index) + 1

    atoms_idx = [l1_1b, l2_1b, l3_1b] + prots_final

    boresch_wrt.write_boresch_variables(
        psf, atoms_idx, equils, LIG_SEGID, PROT_SEGID, i + 1
    )
    print(f"  Written: boresch_variables{i+1}.str  ({cx_name})")

print("\nDone.")

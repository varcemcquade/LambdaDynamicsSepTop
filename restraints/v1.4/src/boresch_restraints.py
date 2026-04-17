"""
Lambda-Dynamic Boresch Restraints Atom Selection Algorithm
-----------------------------------------------------------------------------
This set of scripts implements the atom selection algorithm for automatically identifying
suitable ligand and protein atoms for Boresch-style restraints to be used in lambda-dynamics.

The algorithm is designed to find a single set of ligand and protein atoms that satisfy the
geometric criteria for Boresch restraints across multiple ligands bound to a common protein.

Please ensure all ligands are aligned to the same protein coordinates.

Single protein PSF/PDB + per-ligand MOL2/PDB.  No per-ligand complex files.

Author: Valentino Arce-McQuade
"""

import os
import sys
import numpy as np
import MDAnalysis as mda

import boresch_lig
import boresch_prot
import boresch_equils
import boresch_wrt

PROTEIN_PSF = "protein.psf"
PROTEIN_PDB = "protein.pdb"
PROT_SEGID  = "PROA"
LIG_NAMES   = "lig_names.txt" # one ligand base name per line (no extension)
LIG_SEGID   = "HETA"
BOX_SIZE    = 84 # Å, cubic box assumed

if not os.path.isfile(LIG_NAMES):
    sys.exit(f"ERROR: {LIG_NAMES} not found in {os.getcwd()}")

with open(LIG_NAMES) as fh:
    ligand_names = [line.strip() for line in fh if line.strip()]

if not ligand_names:
    sys.exit(f"ERROR: {LIG_NAMES} is empty.")

print(f"Found {len(ligand_names)} ligand(s): {ligand_names}")

for path in (PROTEIN_PSF, PROTEIN_PDB):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: required file not found: {path}")

prot_u = mda.Universe(PROTEIN_PSF, PROTEIN_PDB)
print(f"Loaded protein: {prot_u.n_atoms} atoms")

def _load_ligand(name):
    """
    Load a ligand-only universe from {name}.mol2 and/or {name}.pdb.
    MOL2 is preferred as topology source (preserves bond information).
    LIG_SEGID is stamped onto all atoms so downstream segid selections work.
    """
    mol2 = f"{name}.mol2"
    pdb  = f"{name}.pdb"
    if os.path.isfile(mol2) and os.path.isfile(pdb):
        u = mda.Universe(mol2, pdb)
    elif os.path.isfile(mol2):
        u = mda.Universe(mol2)
    elif os.path.isfile(pdb):
        u = mda.Universe(pdb)
    else:
        sys.exit(f"ERROR: no .mol2 or .pdb found for ligand '{name}'.")
    u.atoms.segids = np.full(u.n_atoms, LIG_SEGID)
    return u

def _build_complex(lig_u):
    """Merge protein and ligand universes (protein first) and set periodic box."""
    cx = mda.Merge(prot_u.atoms, lig_u.atoms)
    cx.dimensions = [BOX_SIZE, BOX_SIZE, BOX_SIZE, 90, 90, 90]
    return cx


def _triplet_to_ids(complex_u, triplet):
    """
    Convert [p1, p2, p3] 0-based atom indices to stable (resid, name) tuples.
    These survive across universe rebuilds. Never store raw indices between universes.
    """
    return [(int(complex_u.atoms[i].resid), str(complex_u.atoms[i].name)) for i in triplet]


def _ids_to_triplet(complex_u, triplet_ids):
    """
    Convert [(resid, name), ...] stable identifiers back to [p1, p2, p3]
    0-based atom indices within complex_u.
    """
    result = []
    for resid, name in triplet_ids:
        sel = complex_u.select_atoms(f"protein and resid {resid} and name {name}")
        if sel.n_atoms == 0:
            raise RuntimeError(f"Protein atom resid={resid} name={name} not found in universe.")
        result.append(int(sel[0].index))
    return result

# ---------------------------------------------------------------------------
# Identify candidate protein triplets for the reference ligand
# ---------------------------------------------------------------------------

ref_name  = ligand_names[0]
ref_lig_u = _load_ligand(ref_name)
ref_complex_u  = _build_complex(ref_lig_u)

print(f"\n--- Reference ligand: {ref_name} ---")

ref_lig_names = boresch_lig.select_ligand_atoms(ref_lig_u)
l1_name, l2_name, l3_name = ref_lig_names
print(f"Ligand atoms selected: L1 = {l1_name}, L2 = {l2_name}, L3 = {l3_name}")

l1_coords = ref_complex_u.select_atoms(f"segid {LIG_SEGID} and name {l1_name}")[0].position.reshape(1, 3)

prot_candidates = boresch_prot.select_protein_atoms(ref_complex_u, l1_coords, box_size=BOX_SIZE)
print(f"Protein atom candidates (first pass): {len(prot_candidates)}")

if not prot_candidates:
    sys.exit(
        "ERROR: No protein candidate atoms found for the reference ligand.\n"
        "Consider relaxing skip_start/skip_end or distance thresholds in "
        "boresch_prot.select_protein_atoms()."
    )

print("Searching for valid protein triplets (this may take a moment)...")

candidate_triplets = boresch_prot.find_triplets(ref_complex_u, prot_candidates, l1_name, l2_name, l3_name, LIG_SEGID, box_size=BOX_SIZE)

print(f"Triplets found for reference ligand: {len(candidate_triplets)}")

if not candidate_triplets:
    sys.exit(
        "ERROR: No valid protein triplets found for the reference ligand.\n"
        "Try adjusting distance/angle thresholds or choosing a different "
        "reference ligand."
    )

# Convert to stable (resid, name) identifiers -- never carry raw indices across universes.
candidate_ids = [_triplet_to_ids(ref_complex_u, t) for t in candidate_triplets]
all_lig_names = [ref_lig_names]

# ---------------------------------------------------------------------------
# Filter triplets across remaining ligands
# ---------------------------------------------------------------------------

for lig_name in ligand_names[1:]:
    lig_u = _load_ligand(lig_name)
    complex_u  = _build_complex(lig_u)
    complex_lig_names = boresch_lig.select_ligand_atoms(lig_u)
    all_lig_names.append(complex_lig_names)

    surviving = []
    for ids in candidate_ids:
        triplet = _ids_to_triplet(complex_u, ids)
        if boresch_prot.conditions_met(complex_u, complex_lig_names, triplet, LIG_SEGID, box_size=BOX_SIZE):
            surviving.append(ids)
    candidate_ids = surviving
    print(f"After {lig_name}: {len(candidate_ids)} triplet(s) remaining.")

    if not candidate_ids:
        break

if not candidate_ids:
    sys.exit(
        "ERROR: No protein triplets survived geometry checks across all "
        "ligands.  Consider relaxing the dihedral threshold or the distance "
        "bounds in boresch_prot."
    )

prots_ids = candidate_ids[0]
(resid1, name1), (resid2, name2), (resid3, name3) = prots_ids
print(f"\nFinal protein triplet: P1 = {name1} (resid {resid1}), P2 = {name2} (resid {resid2}), P3 = {name3} (resid {resid3})")

# ---------------------------------------------------------------------------
# Write output for each ligand
# ---------------------------------------------------------------------------

print("\n--- Writing output files ---")

for i, lig_name in enumerate(ligand_names):
    lig_u = _load_ligand(lig_name)
    complex_u  = _build_complex(lig_u)
    complex_lig_names = all_lig_names[i]

    prots_final = _ids_to_triplet(complex_u, prots_ids) # 0-based

    equils = boresch_equils.compute_equils(complex_u, complex_lig_names, prots_final, LIG_SEGID, box_size=BOX_SIZE)

    prot_atoms_idx = [p + 1 for p in prots_final] # 1-based for PSF lookup

    boresch_wrt.write_boresch_variables(PROTEIN_PSF, complex_lig_names, prot_atoms_idx, equils, LIG_SEGID, PROT_SEGID, i + 1)
    print(f"Written: restraint_variables{i+1}.str ({lig_name})")

print("\nDone.")
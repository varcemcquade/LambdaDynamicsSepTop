"""
Lambda-Dynamic Boresch Restraints Atom Selection Algorithm
-----------------------------------------------------------------------------
This set of scripts implements the atom selection algorithm for automatically identifying
suitable ligand and protein atoms for Boresch-style restraints to be used in lambda-dynamics.

The algorithm is designed to find a single set of ligand and protein atoms that satisfy the
geometric criteria for Boresch restraints across multiple ligands bound to a common protein.

Please ensure all ligands are aligned to the same protein coordinates.

Single protein PSF/PDB + per-ligand MOL2/PDB.  No per-ligand complex files.

Original Code by Valentino Arce-McQuade
Modified by Dylan Nam Nguyen
"""

# import libraries and scripts
import os
import sys
import numpy as np
import MDAnalysis as mda

import boresch_lig
import boresch_prot
import boresch_equils
import boresch_wrt

# pre-defined variables
PROTEIN_PSF = "protein.psf"
PROTEIN_PDB = "protein.pdb"
PROT_SEGID  = "PROA"
LIG_NAMES   = "lig_names.txt" # one ligand base name per line (no extension)
LIG_SEGID   = "HETA"
BOX_SIZE    = 84 # Å, cubic box assumed

# check lig_names.txt exists in this directory and parse for all ligands
if not os.path.isfile(LIG_NAMES):
    sys.exit(f"ERROR: {LIG_NAMES} not found in {os.getcwd()}")

with open(LIG_NAMES) as fh:
    ligand_names = [line.strip() for line in fh if line.strip()]

if not ligand_names:
    sys.exit(f"ERROR: {LIG_NAMES} is empty.")

print(f"Found {len(ligand_names)} ligand(s): {ligand_names}")

# check for protein input files and create a protein universe object
for path in (PROTEIN_PSF, PROTEIN_PDB):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: required file not found: {path}")

prot_u = mda.Universe(PROTEIN_PSF, PROTEIN_PDB)
print(f"Loaded protein: {prot_u.atoms.n_atoms} atoms")

# pre-defined functions
def _load_ligand(name):
    """
    Load a ligand-only universe from {name}.mol2 and/or {name}.pdb.
    MOL2 is preferred as topology source (preserves bond information).
    LIG_SEGID is stamped onto all atoms so downstream segid selections work.
    
    Parameters:
        name (str): name of ligand
        
    Returns:
        u (mda.Universe): universe object of given ligand with renamed SEGID
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
        
    # reassign SEGIDs of this ligand universe to {LIG_SEGID}
    u.segments.segids = np.full(u.segments.n_segments, LIG_SEGID)
    return u

def _build_complex(lig_u):
    """
    Merge protein and ligand universes (protein first) and set periodic box.
    
    Parameters:
        lig_u (mda.Universe): ligand universe
    
    Returns:
        complex (mda.Universe): a universe containing lig_u and external variable, prot_u, with a box size of {BOX_SIZE}
    """
    complex = mda.Merge(prot_u.select_atoms(f"not segid {LIG_SEGID} and not (resname SOL WAT TIP3 TIP3P HOH) and not (resname SOD CLA POT)"), lig_u.atoms)
    complex.dimensions = [BOX_SIZE, BOX_SIZE, BOX_SIZE, 90, 90, 90]
    return complex

def _triplet_to_ids(complex_u, triplet):
    """
    Convert [p1, p2, p3] 0-based atom indices to stable (resid, name) tuples.
    These survive across universe rebuilds. Never store raw indices between universes.
    
    Parameters:
        complex_u (mda.Universe): universe complex containing protein and ligand
        triplet (list[int]): 3 element list containing three 0-based indices that refer to atoms in the complex
    
    Returns:
        (list[tuple[int, str]]): 3 element list of tuples that track RESID and ATOM NAME
    
    """
    return [(int(complex_u.atoms[i].resid), str(complex_u.atoms[i].name)) for i in triplet]

def _ids_to_triplet(complex_u, triplet_ids):
    """
    Convert [(resid, name), ...] stable identifiers back to [p1, p2, p3]
    0-based atom indices within complex_u.
    
    Parameters:
        Parameters:
            complex_u (mda.Universe): universe complex containing protein and ligand
            triplet_ids (list[tuple[int, str]]): 3 element list of tuples that track RESID and ATOM NAME
        
        Returns:
            (list[int]): 3 element list containing three indices that refer to atoms in the complex
    """
    result = []
    for resid, name in triplet_ids:
        sel = complex_u.select_atoms(f"protein and resid {resid} and name {name}")
        if sel.n_atoms == 0:
            raise RuntimeError(f"Protein atom resid={resid} name={name} not found in universe.")
        result.append(int(sel[0].index))
    return result

# internal functions for testing the other orderings; is used in the protein triplet determination and ligand triplet determination based on the set protein triplets
def _ligand_orderings(lig_names):

    l1, l2, l3 = lig_names

    orderings = [
        (l1, l2, l3),
        (l1, l3, l2),
    ]

    # avoid duplicate orderings
    unique_orderings = []
    for ordering in orderings:
        if ordering not in unique_orderings:
            unique_orderings.append(ordering)

    return unique_orderings

def _first_valid_ordering(complex_u, base_lig_names, protein_triplet):
    # takes the first valid ordering of ligand triplet

    for ligand_order in _ligand_orderings(base_lig_names):
        if boresch_prot.conditions_met(complex_u, ligand_order, protein_triplet, LIG_SEGID):
            return ligand_order
    return None



# perform candidate triplet checks on reference ligand
ref_name  = ligand_names[0]
ref_lig_u = _load_ligand(ref_name)
ref_complex_u  = _build_complex(ref_lig_u)
print(f"\nReference ligand: {ref_name}")

# receive L1, L2, and L3 from this reference ligand
ref_lig_names_base = boresch_lig.select_ligand_atoms(ref_lig_u)

ref_l1_name = ref_lig_names_base[0]
ref_l1_coords = ref_complex_u.select_atoms(f"segid {LIG_SEGID} and name {ref_l1_name}")[0].position.reshape(1, 3)

# select protein candidate triplets based on L1
prot_candidates = boresch_prot.select_protein_atoms(ref_complex_u, ref_l1_coords, PROT_SEGID)
if not prot_candidates:
    sys.exit(
        "ERROR: No protein candidate atoms found for the reference ligand.\n"
        "Consider relaxing skip_start/skip_end or distance thresholds in "
        "boresch_prot.select_protein_atoms()."
    )

# begin searching for P1, P2, P3 candidates based on the reference ligand
candidate_records = [] # dictionary of triplet protein ids and ligand orders

for lig_order in _ligand_orderings(ref_lig_names_base):
    l1_name, l2_name, l3_name = lig_order
    print(f"Ligand atoms selected: L1 = {l1_name}, L2 = {l2_name}, L3 = {l3_name}")

    # get all triplets based on ligand triplets
    triplets = boresch_prot.find_triplets(ref_complex_u, prot_candidates, l1_name, l2_name, l3_name, LIG_SEGID)
     
    for triplet in triplets:
        candidate_records.append(
                {"protein_ids": _triplet_to_ids(ref_complex_u, triplet),
                # Store the accepted L1/L2/L3 ordering separately for each ligand.
                "lig_names": {ref_name: lig_order}}
            )

# traverse candidate records and keep only the unique protein triplets
unique_records = []
seen_records = set()

for record in candidate_records:
    record_key = (tuple(record["protein_ids"]), tuple(record["lig_names"][ref_name]))

    if record_key not in seen_records:
        seen_records.add(record_key)
        unique_records.append(record)

candidate_records = unique_records

print(
    f"Total reference candidate records after L2/L3-order testing: "
    f"{len(candidate_records)}"
)

if len(candidate_records) == 0:
    sys.exit(
        "ERROR: No valid protein triplets found for the reference ligand.\n"
        "Try adjusting distance/angle thresholds or choosing a different "
        "reference ligand."
    )


# get ligand triplets for remaining ligands based on the confirmed protein triplets
for lig_name in ligand_names[1:]:
    lig_u = _load_ligand(lig_name)
    complex_u  = _build_complex(lig_u)
    complex_lig_names = boresch_lig.select_ligand_atoms(lig_u)

    surviving = []
    for record in candidate_records:
        triplet = _ids_to_triplet(complex_u, record["protein_ids"])
        
        accepted_lig_order = _first_valid_ordering(complex_u, complex_lig_names, triplet)
        
        if accepted_lig_order is not None:
            updated_record = {"protein_ids": record["protein_ids"], "lig_names": dict(record["lig_names"])}

            updated_record["lig_names"][lig_name] = accepted_lig_order
            surviving.append(updated_record)
        
    candidate_records = surviving
    
    print(
        f"After {lig_name}: "
        f"{len(candidate_records)} candidate record(s) remain."
    )
    
if len(candidate_records) == 0:
    sys.exit(
        "ERROR: No protein triplet survived the geometry checks across all "
        "ligands, even after trying both L2/L3 orderings.\n"
        "Consider relaxing geometry thresholds, selecting different ligand "
        "atoms, or choosing a different reference ligand."
    )
    
    
final_record = candidate_records[0]
prots_ids = final_record["protein_ids"]

(resid1, name1), (resid2, name2), (resid3, name3) = prots_ids

print(f"\nFinal protein triplet: P1 = {name1} (resid {resid1}), P2 = {name2} (resid {resid2}), P3 = {name3} (resid {resid3})")

print("\nFinal ligand orderings: ")

for lig_name in ligand_names:
    l1_name, l2_name, l3_name = final_record["lig_names"][lig_name]

    print(f"{lig_name}: L1 = {l1_name}, L2 = {l2_name}, L3 = {l3_name}")


# write output for each ligand
print("\n--- Writing output files ---")

for index, lig_name in enumerate(ligand_names, start=1):
    lig_u = _load_ligand(lig_name)
    complex_u = _build_complex(lig_u)

    prots_final = _ids_to_triplet(complex_u, prots_ids)


    lig_order = final_record["lig_names"][lig_name]

    equils = boresch_equils.compute_equils(complex_u, lig_order, prots_final, LIG_SEGID,
    )

    # boresch_wrt.return_atom_info() expects 1-based PSF atom indices.
    prot_atoms_idx = [protein_index + 1 for protein_index in prots_final]

    boresch_wrt.write_boresch_variables(PROTEIN_PSF, lig_order, prot_atoms_idx,equils, LIG_SEGID, PROT_SEGID, index)

    print(
        f"Written: restraint_variables{index}.str "
        f"({lig_name}; L1/L2/L3 = {lig_order})"
    )

print("\nDone.")
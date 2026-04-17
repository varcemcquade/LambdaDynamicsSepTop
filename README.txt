Instructions:

Collect all scripts in a single directory alongside protein.psf and protein.pdb.
For each ligand, provide a {name}.mol2 (preferred, for bond topology) and/or {name}.pdb.
List ligand base names (no extension) one per line in lig_names.txt.

The first ligand in lig_names.txt is the reference — L1/L2/L3 are selected from it and
protein triplet candidates are filtered across all remaining ligands.

See TYK2.inp for CHARMM implementation of the restraint atoms and equilibrium parameters.

Run boresch_restraints.py
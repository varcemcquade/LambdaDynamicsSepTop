import MDAnalysis as mda
from MDAnalysis.topology import guessers
import networkx as nx
import numpy as np
import boresch_chk

def select_ligand_atoms(complex_psf, complex_dcd, lig_segid):
    """
    Select three ligand atoms for Boresch restraints.
    Uses graph-based heuristic: finds the longest shortest path, picks its middle
    atom as L1, then selects L2/L3 from neighbors ensuring:
      - All three atoms are distinct
      - The triplet is not collinear

    :param complex_psf:
        psf file of solvated protein-ligand complex
    :param complex_dcd:
        equilibration dcd/pdb file of solvated protein-ligand complex
    :param lig_segid:
        segment id of ligand

    :return [l1, l2, l3]:
        0-based atom indices
    """

    u = mda.Universe(complex_psf, complex_dcd)

    # Make sure atoms have element attribute for RDKit conversion
    if not hasattr(u.atoms, "elements"):
        names = [str(n) for n in u.atoms.names]
        elems = [guessers.guess_atom_element(n) for n in names]
        u.add_TopologyAttr("elements", elems)

    ligand = u.select_atoms("segid %s" % lig_segid)
    heavy_ligand = ligand.select_atoms("not name H* LP*")

    local = {atom.index: i for i, atom in enumerate(heavy_ligand)}
    inv_local = {i: j for j, i in local.items()}
    idx = set(local.keys())
    ligand_graph = nx.Graph()
    ligand_graph.add_nodes_from(range(len(heavy_ligand.atoms)))

    for b in heavy_ligand.bonds:
        i = b.atoms[0].index
        j = b.atoms[1].index
        if i in idx and j in idx:
            ligand_graph.add_edge(local[i], local[j])

    # Find longest shortest path in ligand, get middle atom
    short_paths = dict(nx.shortest_path(ligand_graph))
    longest_paths = []
    longest_path_length = 0
    center = 0

    for i in short_paths.values():
        for key, value in i.items():
            if len(value) > longest_path_length:
                longest_path_length = len(value)
                longest_paths.clear()
                longest_paths.append(value)
            elif len(value) == longest_path_length:
                longest_paths.append(value)
        center = longest_paths[0][int(len(longest_paths[0]) / 2)]

    center_global = inv_local[center]
    coords = u.atoms.positions

    # Gather all neighbor global indices, preferring aromatic ones first
    aromatic_atoms = ligand.select_atoms("smarts a")
    aromatic_set = set(int(j) for j in aromatic_atoms.indices)

    neighbor_locals = list(ligand_graph[center].keys())
    neighbor_globals = [int(inv_local[n]) for n in neighbor_locals]

    aromatic_neighbors = [n for n in neighbor_globals if n in aromatic_set]
    non_aromatic_neighbors = [n for n in neighbor_globals if n not in aromatic_set]
    ordered_neighbors = aromatic_neighbors + non_aromatic_neighbors

    # Also gather 2nd-shell neighbors (neighbors of neighbors) as fallback
    second_shell = []
    for n_local in neighbor_locals:
        for nn_local in ligand_graph[n_local].keys():
            nn_global = int(inv_local[nn_local])
            if nn_global != int(center_global) and nn_global not in ordered_neighbors:
                second_shell.append(nn_global)
    second_shell_aromatic = [n for n in second_shell if n in aromatic_set]
    second_shell_other = [n for n in second_shell if n not in aromatic_set]

    all_candidates = ordered_neighbors + second_shell_aromatic + second_shell_other

    # Try all pairs of (l2, l3) from candidates, pick first non-collinear triplet
    l1 = int(center_global)
    for i, l2 in enumerate(all_candidates):
        for l3 in all_candidates[i+1:]:
            if not boresch_chk.is_collinear(coords, [l1, l2, l3]):
                return [l1, l2, l3]

    raise ValueError(
        f"Could not find 3 non-collinear ligand atoms for Boresch restraints. "
        f"Center atom index={center_global} ({u.atoms[int(center_global)].name}), "
        f"candidates tried: {all_candidates}"
    )

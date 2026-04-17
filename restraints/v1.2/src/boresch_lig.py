import MDAnalysis as mda
from MDAnalysis.topology import guessers
import networkx as nx
import numpy as np
import boresch_chk
from itertools import combinations

def select_ligand_atoms(complex_psf, complex_dcd, lig_segid):
    """
    This function selects three ligand atoms for boresch restraints. Shortest distance between
    each pair of atoms is calculated, and the longest path of this set is collected. Middle of longest shortest path
    is chosen to be center of mass atom (L1). L2 and L3 are chosen as direct neighbors of L1,
    preferring aromatic neighbors, ensuring all three are distinct and non-collinear.

    If the middle atom cannot form a valid non-collinear triplet from its neighbors,
    adjacent atoms along the longest path are tried, expanding outward from the center.

    :param complex_psf:
        psf file of solvated protein-ligand complex
    :param complex_dcd:
        equilibration dcd/pdb file of solvated protein-ligand complex
    :param lig_segid:
        editable segment id of ligand

    :return [l1, l2, l3]:
        0-based atom indices of atom selection heuristic
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

    # Find longest shortest path in ligand
    short_paths = dict(nx.shortest_path(ligand_graph))
    longest_paths = []
    longest_path_length = 0

    for i in short_paths.values():
        for key, value in i.items():
            if len(value) > longest_path_length:
                longest_path_length = len(value)
                longest_paths.clear()
                longest_paths.append(value)
            elif len(value) == longest_path_length:
                longest_paths.append(value)

    path = longest_paths[0]
    mid = int(len(path) / 2)
    coords = u.atoms.positions

    aromatic_atoms = ligand.select_atoms("smarts a")
    aromatic_set = set(int(j) for j in aromatic_atoms.indices)

    # Try center candidates: start at middle of path, expand outward
    offsets = [0]
    for d in range(1, len(path)):
        offsets.append(d)
        offsets.append(-d)

    for offset in offsets:
        pos = mid + offset
        if pos < 0 or pos >= len(path):
            continue

        center = path[pos]
        l1 = int(inv_local[center])

        # Get neighbors, ordered: aromatic first, then non-aromatic
        neighbor_locals = list(ligand_graph[center].keys())
        neighbor_globals = [int(inv_local[n]) for n in neighbor_locals]
        aromatic_nbrs = [n for n in neighbor_globals if n in aromatic_set]
        other_nbrs = [n for n in neighbor_globals if n not in aromatic_set]
        ordered_neighbors = aromatic_nbrs + other_nbrs

        if len(ordered_neighbors) < 2:
            continue

        # Try all pairs, preferring aromatic-heavy pairs (earlier in ordered list)
        for l2, l3 in combinations(ordered_neighbors, 2):
            if not boresch_chk.is_collinear(coords, [l1, l2, l3]):
                return [l1, l2, l3]

    raise ValueError(
        f"Could not find 3 non-collinear ligand atoms for Boresch restraints. "
        f"Longest path length={len(path)}, all center candidates exhausted."
    )
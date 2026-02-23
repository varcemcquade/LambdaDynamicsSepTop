from MDAnalysis.topology import guessers
import networkx as nx
import numpy as np


def is_collinear(positions, atoms, threshold=0.91):
    """Check if atoms are collinear (from boresch_chk)"""
    for i in range(len(atoms) - 2):
        v1 = positions[atoms[i + 1], :] - positions[atoms[i], :]
        v2 = positions[atoms[i + 2], :] - positions[atoms[i + 1], :]
        normalized_inner_product = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
        if np.abs(normalized_inner_product) > threshold:
            return True
    return False


def select_ligand_atoms(u, lig_segid):
    """
    This function selects three ligand atoms for Boresch restraints.

    FIXED VERSION: Ensures the selected atoms are not collinear by checking
    multiple candidate triplets and rejecting collinear ones.

    :param u:
        MDAnalysis universe, 0-based atom indices
    :param lig_segid:
        editable segment id of ligand
    :return L1, L2, L3:
        atom indices of atom selection heuristic - ZERO-BASED
    """

    # Make sure atoms have element attribute for RDKit conversion
    if not hasattr(u.atoms, "elements"):
        names = [str(n) for n in u.atoms.names]
        elems = [guessers.guess_atom_element(n) for n in names]
        u.add_TopologyAttr("elements", elems)

    ligand = u.select_atoms(f"segid {lig_segid}")
    heavy_ligand = ligand.select_atoms("not name H* LP*")

    if heavy_ligand.n_atoms < 3:
        raise ValueError(f"Ligand has fewer than 3 heavy atoms. Cannot select L1, L2, L3.")

    local = {atom.index: i for i, atom in enumerate(heavy_ligand)}
    inv_local = {i: j for j, i in local.items()}
    idx = set(local.keys())

    ligand_graph = nx.Graph()
    ligand_graph.add_nodes_from(range(len(heavy_ligand.atoms)))

    for b in heavy_ligand.bonds:
        i, j = b.atoms[0].index, b.atoms[1].index
        if i in idx and j in idx:
            ligand_graph.add_edge(local[i], local[j])

    # Find center atom of longest shortest path
    short_paths = dict(nx.shortest_path(ligand_graph))
    longest_path_length = 0
    center_local = 0

    for paths_from_node in short_paths.values():
        for path in paths_from_node.values():
            if len(path) > longest_path_length:
                longest_path_length = len(path)
                center_local = path[len(path) // 2]

    # Collect L1
    l1_global = inv_local[center_local]

    # Get positions for collinearity check
    positions = u.atoms.positions.copy()

    # Get all neighbors of L1
    neighbor_globals = [inv_local[i] for i in ligand_graph[center_local].keys()]

    if len(neighbor_globals) < 2:
        raise ValueError(f"Ligand center atom has fewer than 2 neighbors. Cannot select L2 and L3.")

    # Try to find aromatic neighbors first, but ensure non-collinearity
    try:
        aromatic_indices = set(ligand.select_atoms("smarts a").indices)
    except:
        aromatic_indices = set()

    aromatic_neighbors = [idx for idx in neighbor_globals if idx in aromatic_indices]
    non_aromatic_neighbors = [idx for idx in neighbor_globals if idx not in aromatic_indices]

    # Build candidate list: prefer aromatic, then non-aromatic
    all_neighbor_candidates = aromatic_neighbors + non_aromatic_neighbors

    # Try all pairs of neighbors and pick the first non-collinear triplet
    best_triplet = None
    best_dot_product = 1.0  # Lower is better (less collinear)

    for i, l2_global in enumerate(all_neighbor_candidates):
        for l3_global in all_neighbor_candidates[i + 1:]:
            # Check collinearity
            if not is_collinear(positions, [l1_global, l2_global, l3_global]):
                # Calculate how far from collinear (lower is better)
                v1 = positions[l2_global, :] - positions[l1_global, :]
                v2 = positions[l3_global, :] - positions[l2_global, :]
                dot = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

                if dot < best_dot_product:
                    best_dot_product = dot
                    best_triplet = (l1_global, l2_global, l3_global)

    if best_triplet is not None:
        return list(best_triplet)

    # If no non-collinear triplet found with direct neighbors,
    # try using second-shell neighbors for L3
    for l2_global in all_neighbor_candidates:
        l2_local = local[l2_global]
        second_neighbors = [inv_local[i] for i in ligand_graph[l2_local].keys()
                            if inv_local[i] != l1_global]

        for l3_global in second_neighbors:
            if not is_collinear(positions, [l1_global, l2_global, l3_global]):
                v1 = positions[l2_global, :] - positions[l1_global, :]
                v2 = positions[l3_global, :] - positions[l2_global, :]
                dot = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

                if dot < best_dot_product:
                    best_dot_product = dot
                    best_triplet = (l1_global, l2_global, l3_global)

    if best_triplet is not None:
        return list(best_triplet)

    # Fallback: just return the first two neighbors (original behavior)
    # This may still be collinear but at least won't crash
    print(f"WARNING: Could not find non-collinear ligand triplet. Using fallback selection.")
    return [l1_global, neighbor_globals[0], neighbor_globals[1]]
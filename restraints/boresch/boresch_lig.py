import numpy as np
from itertools import combinations
from MDAnalysis.topology import guessers
import networkx as nx

def _positions(lig_u, indices):
    """Return (n, 3) coordinate array for a list of local ligand atom indices."""
    return np.array([lig_u.atoms[i].position for i in indices])

def _is_collinear(positions, threshold=0.9):
    """True if any consecutive vector triplet in *positions* is collinear."""
    for i in range(len(positions) - 2):
        v1 = positions[i] - positions[i + 1]
        v2 = positions[i + 1] - positions[i + 2]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return True
        if abs(np.dot(v1, v2) / (n1 * n2)) > threshold:
            return True
    return False

def select_ligand_atoms(lig_u):
    """
    Select three ligand atoms for Boresch restraints.
    Operates on a ligand-only universe. Returns atom names, which are stable
    identifiers regardless of how the universe is later merged or re-indexed.

    Steps:
        1. Build a heavy-atom graph using local MDAnalysis indices as nodes.
        2. Find the longest shortest path. The path midpoint is the "center" 
           atom.
        3. Selection of candidate center atoms has priority system whereby
           atoms part of a aromatic system and ring system are preferred. L2/L3
           are also selected with this candidate tiering in mind. If nothing is
           available, fall back to graph neighbours.
        4. After building a pool of candidates, check for collinearity. If the
           primary selection is collinear, exhaust other center and neighbour
           combinations before falling back to any non-collinear triplet from
           the heavy-atom graph.

    Requires bond information on lig_u — load from mol2 or equivalent.
    Atom names within the ligand must be unique.

    Parameters:
        lig_u (mda.Universe): ligand universe
    
    Returns:
        (tuple[str]): 3 element tuple of atom name strings (l1_name, l2_name, l3_name)
    """
    
    # "guess" element information from ligand universe if not filled
    if not hasattr(lig_u.atoms, "elements"):
        names = [str(n) for n in lig_u.atoms.names]
        elems = [guessers.guess_atom_element(n) for n in names]
        lig_u.add_TopologyAttr("elements", elems)

    # define a network graph of heavy atoms only
    heavy = lig_u.select_atoms("not name H* LP*")

    if heavy.n_atoms < 3:
        raise RuntimeError("Ligand universe has fewer than 3 heavy atoms.")

    lig_graph = nx.Graph()
    lig_graph.add_nodes_from(heavy.atoms.indices)

    for bond in heavy.bonds:
        i, j = bond.atoms[0].index, bond.atoms[1].index
        if lig_graph.has_node(i) and lig_graph.has_node(j):
            lig_graph.add_edge(i, j)

    # find all longest short paths and the center(s) of that path
    all_short_paths = dict(nx.shortest_path(lig_graph))
    long_paths = []
    longest_length = 0

    for source, targets in all_short_paths.items():
        for target, path in targets.items():
            curr_len = len(path)
            if curr_len > longest_length:
                long_paths.clear()
                long_paths.append(path)
                longest_length = curr_len
            elif curr_len == longest_length:
                long_paths.append(path)

    centers = set(path[longest_length // 2] for path in long_paths)

    # determine which atoms are part of an aromatic ring
    aromatic_atom_indices = set(lig_u.select_atoms("smarts a").indices)

    # determine which atoms are part of a ring system
    ring_atom_indices = set()
    graph_rings = nx.cycle_basis(lig_graph)
    for rings in graph_rings:
        ring_atom_indices.update(rings)

    # choose candidate atoms by first prioritizing aromatic atoms then ring system atoms
    candidate_atoms = []

    for center in sorted(centers):
        atoms = [center]

        if center in aromatic_atom_indices:
            for neighbor in lig_graph[center].keys():
                if neighbor in aromatic_atom_indices and neighbor not in atoms:
                    atoms.append(neighbor)
                elif neighbor in ring_atom_indices and neighbor not in atoms:
                    atoms.append(neighbor)
        elif center in ring_atom_indices:
            for neighbor in lig_graph[center].keys():
                if neighbor in ring_atom_indices and neighbor not in atoms:
                    atoms.append(neighbor)
        else:
            for neighbor in lig_graph[center].keys():
                if neighbor in aromatic_atom_indices:
                    atoms = [neighbor, center]
                elif neighbor in ring_atom_indices:
                    atoms = [neighbor, center]
                    for neighbor2 in lig_graph[neighbor].keys():
                        if neighbor2 in aromatic_atom_indices and neighbor2 not in atoms:
                            atoms.append(neighbor2)
                        if neighbor2 in ring_atom_indices and neighbor2 not in atoms:
                            atoms.append(neighbor2)
                    break

        if len(atoms) < 3:
            anchor = atoms[0]
            for neighbor in lig_graph[anchor].keys():
                if neighbor not in atoms:
                    atoms.append(neighbor)

        candidate_atoms.append(atoms)

    # Pick the first non-collinear triplet: try prioritized candidate atom groups
    # (aromatic/center selections) first, then fall back to all heavy atoms.
    search_groups = candidate_atoms + [list(heavy.atoms.indices)]
    result_indices = None  # default if nothing found

    for atoms in search_groups:        
        for c in combinations(atoms, 3):
            positions = _positions(lig_u, c)
            if not _is_collinear(positions):
                result_indices = c
                break
        if result_indices is not None:
            break


    if result_indices is None:
        raise RuntimeError(
            "Could not find 3 non-collinear heavy atoms in the ligand universe. "
            "This ligand may be too small or too linear."
        )

    l1_idx, l2_idx, l3_idx = result_indices

    return (lig_u.atoms[l1_idx].name,
            lig_u.atoms[l2_idx].name,
            lig_u.atoms[l3_idx].name)
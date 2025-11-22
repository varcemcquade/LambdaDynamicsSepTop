from MDAnalysis.topology import guessers
import networkx as nx

def select_ligand_atoms(u, lig_segid):
    """
    This function selects three ligand atoms for boresch restraints. Shortest distance between
    each pair of atoms is calculated, and the longest path of this set is collected. Middle of longest shortest path
    is chosen to be center of mass atom. Closest ring atom to COM atom is the first true atom selection, L1. Subsequently,
    L2 and L3 are chosen as the closest ring atoms to L1.
    :param u:
        MDAnalysis universe, 0-based atom indices
    :param lig_segid:
        editable segment id of ligand
    :return L1, L2, L3:
        atom indices of atom selection heuristic
    """

    # Make sure atoms have element attribute for RDKit conversion
    if not hasattr(u.atoms, "elements"):
        names = [str(n) for n in u.atoms.names]  # plain Python strings
        elems = [guessers.guess_atom_element(n) for n in names]  # per-atom
        u.add_TopologyAttr("elements", elems)

    ligand = u.select_atoms(f"segid {lig_segid}")
    heavy_ligand = ligand.select_atoms("not name H* LP*")

    local = {atom.index: i for i, atom in enumerate(heavy_ligand)}  # Dictionary for global index -> local index
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

    # Get neighbors
    aromatic_indices = set(ligand.select_atoms("smarts a").indices)
    neighbor_globals = [inv_local[i] for i in ligand_graph[center_local].keys()]
    aromatic_neighbors = [idx for idx in neighbor_globals if idx in aromatic_indices]

    if len(aromatic_neighbors) >= 2:
        l2_global, l3_global = aromatic_neighbors[:2]
    elif len(aromatic_neighbors) == 1:
        l2_global = aromatic_neighbors[0]
        l3_global = next((idx for idx in neighbor_globals if idx != l2_global), neighbor_globals[0])
    else:
        if len(neighbor_globals) >= 2:
            l2_global, l3_global = neighbor_globals[:2]
        else:
            raise ValueError(f"Ligand center atom has fewer than 2 neighbors. Cannot select L2 and L3.")

    return [l1_global + 1, l2_global + 1, l3_global + 1]
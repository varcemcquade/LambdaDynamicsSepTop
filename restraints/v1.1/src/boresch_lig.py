import MDAnalysis as mda
from MDAnalysis.topology import guessers
import networkx as nx

def select_ligand_atoms(complex_psf, complex_dcd, lig_segid):
    """
    This function selects three ligand atoms for boresch restraints. Shortest distance between
    each pair of atoms is calculated, and the longest path of this set is collected. Middle of longest shortest path
    is chosen to be center of mass atom. Closest ring atom to COM atom is the first true atom selection, L1. Subsequently,
    L2 and L3 are chosen as the closest ring atoms to L1.

    :param complex_psf:
        psf file of solvated protein-ligand complex
    :param complex_dcd:
        equilibration dcd file of solvated protein-ligand complex
    :param lig_segid:
        editable segment id of ligand

    :return L1, L2, L3:
        atom indices of atom selection heuristic
    """

    ligand_list = []

    # 0-based indexing starts
    u = mda.Universe(complex_psf, complex_dcd)

    # Make sure atoms have element attribute for RDKit conversion
    if not hasattr(u.atoms, "elements"):
        names = [str(n) for n in u.atoms.names]  # plain Python strings
        elems = [guessers.guess_atom_element(n) for n in names]  # per-atom
        u.add_TopologyAttr("elements", elems)

    ligand = u.select_atoms("segid %s" % lig_segid)
    heavy_ligand = ligand.select_atoms("not name H* LP*")

    local = {atom.index: i for i, atom in enumerate(heavy_ligand)}  # Dictionary for global index -> local index
    inv_local = {i: j for j, i in local.items()}
    idx = set(local.keys())
    ligand_graph = nx.Graph()
    ligand_graph.add_nodes_from(range(len(heavy_ligand.atoms)))

    for b in heavy_ligand.bonds:
        i = b.atoms[0].index  # global index of first atom
        j = b.atoms[1].index  # global index of second atom
        if i in idx and j in idx:
            ligand_graph.add_edge(local[i], local[j])

    # Find longest shortest path in ligand, get middle atom of that
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
        # there might be multiple longest path, choose first one
        center = longest_paths[0][int(len(longest_paths[0]) / 2)]

    # Collect L1
    ligand_list.append(inv_local[center])

    aromatic_atoms = ligand.select_atoms("smarts a")

    # Collect L2 and L3
    for i in ligand_graph[center].keys():  # Loop through local neighbor indices
        index = inv_local[i]  # Get global neighbor indices
        for j in aromatic_atoms.indices:
            if index == j:
                ligand_list.append(index)

    # If not enough aromatic neighbors, move to ordinary neighbors
    if len(ligand_list) < 3:
        for i in ligand_graph[center].keys():
            index = inv_local[i]
            ligand_list.append(index)

    if len(ligand_list) > 3:
        ligand_list = ligand_list[:3]

    # 0-based indexing stops
    for i in range(len(ligand_list)):
        ligand_list[i] += 1

    l1 = ligand_list[0]
    l2 = ligand_list[1]
    l3 = ligand_list[2]

    return [l1, l2, l3]
import numpy as np
from itertools import combinations
import MDAnalysis as mda
from MDAnalysis.topology import guessers
import networkx as nx


def _positions(u, indices):
    """Return (n, 3) coordinate array for a list of global atom indices."""
    return np.array([u.atoms[i].position for i in indices])


def _is_collinear(positions, threshold=0.9):
    """True if any consecutive vector triplet in *positions* is collinear."""
    for i in range(len(positions) - 2):
        v1 = positions[i]     - positions[i + 1]
        v2 = positions[i + 1] - positions[i + 2]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return True                          # degenerate (zero-length vector)
        if abs(np.dot(v1, v2) / (n1 * n2)) > threshold:
            return True
    return False


def select_ligand_atoms(u, lig_segid, lig_u=None):
    """
    Select three ligand atoms for Boresch restraints.

    Strategy
    --------
    1. Build a heavy-atom graph using global MDAnalysis indices as nodes.
    2. Find the longest shortest path; the midpoint is the "center" atom.
    3. If the center is aromatic, L1 = center and L2/L3 are aromatic
       neighbours.  If the centre is not aromatic but has an aromatic
       neighbour, that neighbour becomes L1 with its own aromatic neighbours
       as L2/L3.  If nothing aromatic is available, fall back to graph
       neighbours.
    4. After building a pool of candidates, check for collinearity.  If the
       primary selection is collinear, exhaust other aromatic/neighbour
       combinations before falling back to **any** non-collinear triplet from
       the heavy-atom graph.
    5. Returns atom *names* (strings) — portable across complexes that share
       the same ligand topology but may differ in atom numbering.

    ** Atom names within the ligand segment must be unique. **

    :param u:         MDAnalysis Universe of the full complex (psf + pdb).
    :param lig_segid: Segment ID string of the ligand (e.g. "HETA").
    :param lig_u:     Optional separate ligand Universe; merged if supplied.
    :return:          Tuple (l1_name, l2_name, l3_name).
    """

    if lig_u:
        u = mda.Merge(u, lig_u)

    if not hasattr(u.atoms, "elements"):
        names = [str(n) for n in u.atoms.names]
        elems = [guessers.guess_atom_element(n) for n in names]
        u.add_TopologyAttr("elements", elems)

    lig_sel  = u.select_atoms(f"segid {lig_segid}")
    heavy_lig = lig_sel.select_atoms("not name H* LP*")

    if heavy_lig.n_atoms < 3:
        raise RuntimeError(
            f"Ligand (segid {lig_segid}) has fewer than 3 heavy atoms."
        )

    # Build graph with *global* MDAnalysis indices as nodes
    lig_idx_set = set(heavy_lig.atoms.indices)
    lig_graph   = nx.Graph()
    lig_graph.add_nodes_from(heavy_lig.atoms.indices)
    for bond in heavy_lig.bonds:
        i, j = bond.atoms[0].index, bond.atoms[1].index
        if i in lig_idx_set and j in lig_idx_set:
            lig_graph.add_edge(i, j)

    # Longest shortest path → midpoint centres
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

    aromatic_atoms = lig_sel.select_atoms("smarts a")
    aromatic_set   = set(aromatic_atoms.indices)

    # -----------------------------------------------------------------------
    # Build a prioritised pool of candidate triplets
    # -----------------------------------------------------------------------
    candidate_pools = []   # each element is a list of ≥3 global indices

    for center in sorted(centers):   # sorted for reproducibility
        pool = [center]

        if center in aromatic_set:
            for nb in lig_graph[center].keys():
                if nb in aromatic_set and nb not in pool:
                    pool.append(nb)
        else:
            for nb in lig_graph[center].keys():
                if nb in aromatic_set:
                    pool = [nb, center]
                    for nb2 in lig_graph[nb].keys():
                        if nb2 in aromatic_set and nb2 not in pool:
                            pool.append(nb2)
                    break

        # Generic fallback: fill up to enough candidates from graph neighbours
        if len(pool) < 3:
            anchor = pool[0]
            for nb in lig_graph[anchor].keys():
                if nb not in pool:
                    pool.append(nb)

        candidate_pools.append(pool)

    # -----------------------------------------------------------------------
    # Pick the first non-collinear ordered 3-tuple from the candidate pools
    # -----------------------------------------------------------------------
    def _try_pool(pool):
        """Return (l1_idx, l2_idx, l3_idx) or None if all triples are collinear."""
        # Try the natural ordering first (preserves intent of the selection)
        if len(pool) >= 3:
            trio = pool[:3]
            if not _is_collinear(_positions(u, trio)):
                return tuple(trio)
        # Exhaustively try ordered combinations (limited to first 8 candidates)
        for combo in combinations(pool[:8], 3):
            if not _is_collinear(_positions(u, combo)):
                return combo
        return None

    result_indices = None
    for pool in candidate_pools:
        result_indices = _try_pool(pool)
        if result_indices is not None:
            break

    # -----------------------------------------------------------------------
    # Last-resort fallback: try all heavy-atom combinations
    # -----------------------------------------------------------------------
    if result_indices is None:
        all_indices = list(heavy_lig.atoms.indices)
        for combo in combinations(all_indices, 3):
            if not _is_collinear(_positions(u, combo)):
                result_indices = combo
                break

    if result_indices is None:
        raise RuntimeError(
            f"Could not find 3 non-collinear heavy atoms in ligand segment "
            f"{lig_segid}.  This ligand may be too small or too linear."
        )

    l1_idx, l2_idx, l3_idx = result_indices
    l1_name = u.atoms[l1_idx].name
    l2_name = u.atoms[l2_idx].name
    l3_name = u.atoms[l3_idx].name

    return l1_name, l2_name, l3_name

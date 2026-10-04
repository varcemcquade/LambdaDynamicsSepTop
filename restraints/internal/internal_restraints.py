import argparse
import shutil
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import GetDihedralDeg


def load(path):
    if path.suffix.lower() == ".mol2":
        mol = Chem.MolFromMol2File(path, sanitize=False, removeHs=False, cleanupSubstructures=False)
        names = [a.GetProp("_TriposAtomName") for a in mol.GetAtoms()]
    else:
        mol = ChemMolFromPDBFile(str(path), sanitize=False, removeHs=False)
        names = [a.GetPDBResidueInfo().GetName().strip() for a in mol.GetAtoms()]
    
    Chem.FastFindRings(mol)  # needed for IsInRing() when sanitize=False
    
    return mol, names


def get_dihedrals(mol, rotatable_only=True):
    out = []
    for b in mol.GetBonds():
        j, k = b.GetBeginAtom(), b.GetEndAtom()
        if rotatable_only:
            if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
                continue
            starts = [n.GetIdx() for n in j.GetNeighbors() if n.GetIdx() != k.GetIdx() and n.GetAtomicNum() > 1]
            ends = [n.GetIdx() for n in k.GetNeighbors() if n.GetIdx() != j.GetIdx() and n.GetAtomicNum() > 1]
            if starts and ends:
                out.append((starts[0], j.GetIdx(), k.GetIdx(), ends[0]))
        else:
            for i in (n.GetIdx() for n in j.GetNeighbors() if n.GetIdx() != k.GetIdx() and n.GetAtomicNum() > 1):
                for l in (n.GetIdx() for n in k.GetNeighbors() if n.GetIdx() not in (j.GetIdx(), i) and n.GetAtomicNum() > 1):
                    out.append((i, j.GetIdx(), k.GetIdx(), l))

    return out


# CHARMM dihedral restraint; {n} is the dihedral number, {force}/{block} pick the lambda-scaled or unscaled copy
GEO_DIHEDRAL = """\
GEO sphere RCM dihedral -
   harmonic symmetric force {force} tref @internalphi{i} dtoff 0.0{block} -
   select atom @ligsegid @lig @dih{i}l1name end -
   select atom @ligsegid @lig @dih{i}l2name end -
   select atom @ligsegid @lig @dih{i}l3name end -
   select atom @ligsegid @lig @dih{i}l4name end
"""


def write_internal_restraints(dihedrals, lig_resid, outdir=".", dihk=5.0):
    """
    dihedrals: list of ((name1, name2, name3, name4), phi)
    dihk in kcal/(mol*rad**2)
    """
    path = Path(outdir) / f"internal_variables{lig_resid}.str"

    with open(path, "w") as f:
        f.write(f"!! Internal dihedral restraints for ligand {lig_resid}\n\n")

        # write variables
        for i, (names, phi) in enumerate(dihedrals, 1):
            for j, name in enumerate(names, 1):
                f.write(f"SET DIH{i}L{j}NAME = {name}\n")

            f.write(f"SET INTERNALPHI{i} = {phi:.3f}\n\n")
        f.write(f"SET NDIH = {len(dihedrals)}\n")
        f.write(f"SET INTERNALPHIK = {dihk}\n")

        # write mmfp restraints
        f.write("MMFP\n\n")
        for i in range(1, len(dihedrals) + 1):
            f.write(GEO_DIHEDRAL.format(i=i, force="-@internalphik", block=" block @ligblock") + "\n")
            f.write(GEO_DIHEDRAL.format(i=i, force="@internalphik", block="") + "\n")
        f.write("END\n\n")
    
    return None


def add_str_to_inp(inp, stream_dir="@builddir"):
    inp = Path(inp)
    stream_line = f"stream {stream_dir}/internal_variables@{{lig}}.str"
    lines = inp.read_text().splitlines()
    
    if stream_line in lines:
        return None

    insert_at = None
    seen_boresch = False

    for i, line in enumerate(lines):
        if "boresch restraints" in line.lower():
            seen_boresch = True
        elif seen_boresch and line.strip().lower() == "incr lig":
            insert_at = i
            break

    if insert_at is None:
        raise ValueError(f"no 'incr lig' line found after the Boresch block in {inp}.")

    # backup .inp
    shutil.copy(inp, inp.with_name(inp.name + ".bak"))
    lines[insert_at:insert_at] = [stream_line, ""]
    inp.write_text("\n".join(lines) + "\n")
    
    return None


def process_ligand(path, lig_resid, outdir, rotatable_only, dihk):
    path = Path(path)
    mol, names = load(path)
    conf = mol.GetConformer()
    all_dihedral_indices = get_dihedrals(mol, rotatable_only)

    dihedrals = []
    for dihedral_indicies in all_dihedral_indices:
        dihedrals.append((tuple(names[idx] for idx in dihedral_indicies), GetDihedralDeg(conf, *dihedral_indicies)))

    write_internal_restraints(dihedrals, lig_resid, outdir, dihk)
    
    return lig_resid, path.name, len(dihedrals)


LIG_NAMES = "lig_names.txt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lig_names_file", default="lig_names.txt", help="Specify text file listing all ligand names.")
    parser.add_argument("--ligdir", default=".", help="directory containing {name}.mol2 (or {name}.pdb)")
    parser.add_argument("--outdir", default=".", help="where to write internal_restraints{i}.str")
    parser.add_argument("--inp", help="CHARMM .inp to add the stream line to")
    parser.add_argument("--stream-dir", default="@builddir", help="--outdir as CHARMM should see it (default: @builddir)")
    parser.add_argument("--rotatable", action="store_true", help="one torsion per rotatable bond instead of all")
    parser.add_argument("--k", type=float, default=5.0, help="force constant, kcal/mol*rad**2")
    args = parser.parse_args()

    with open(args.lig_names_file) as f:
        lig_names = [line.strip() for line in f if line.strip()]

    print(f"Found {len(lig_names)} ligand(s): {lig_names}")

    for lig_resid, name in enumerate(lig_names, 1):
        path = Path(args.ligdir) / f"{name}.mol2"
        if not path.exists():
            path = path.with_suffix(".pdb")

        _, file_name, ndih = process_ligand(path, lig_resid, args.outdir, args.rotatable, args.k)
        print(f"lig {lig_resid}: {file_name} -> {ndih} dihedrals")

    if args.inp:
        add_str_to_inp(args.inp, args.stream_dir)


if __name__ == "__main__":
    main()

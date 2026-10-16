"""Command line interface.

    confdesc split   crest_conformers.xyz -o confs/ --window 3
    confdesc inputs  confs/*.xyz --route "B3LYP/def2SVP EmpiricalDispersion=GD3BJ Opt Freq" -o inputs/
    confdesc parse   logs/*.log -o conformers.csv
    confdesc aggregate conformers.csv --id "^(.*)_conf\\d+$" -o descriptors.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from .ensemble import aggregate, split_ensemble
from .gaussian import read_output, write_input
from .xyz import read_multixyz


def main(argv=None):
    ap = argparse.ArgumentParser(prog="confdesc")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split", help="split a CREST ensemble into XYZ files within an energy window")
    p.add_argument("ensemble"); p.add_argument("-o", "--outdir", required=True)
    p.add_argument("--window", type=float, default=3.0, help="kcal/mol"); p.add_argument("--max-n", type=int)
    p.add_argument("--prefix", default="conf")

    p = sub.add_parser("inputs", help="write Gaussian inputs from XYZ files")
    p.add_argument("xyz", nargs="+"); p.add_argument("-o", "--outdir", required=True)
    p.add_argument("--route", required=True); p.add_argument("--link1-route")
    p.add_argument("--charge", type=int, default=0); p.add_argument("--mult", type=int, default=1)
    p.add_argument("--nproc", type=int); p.add_argument("--mem")

    p = sub.add_parser("parse", help="parse Gaussian outputs into a per-conformer table")
    p.add_argument("logs", nargs="+"); p.add_argument("-o", "--out", required=True)
    p.add_argument("--charges", choices=["nbo", "mulliken"], help="also export atomic charges (q_<atom index>)")

    p = sub.add_parser("aggregate", help="Boltzmann/min/max/low_E aggregation per molecule")
    p.add_argument("table"); p.add_argument("-o", "--out", required=True)
    p.add_argument("--id", required=True, help="column name, or a regex with one group applied to 'name'")
    p.add_argument("--energy", default="G"); p.add_argument("--temperature", type=float, default=298.15)

    a = ap.parse_args(argv)
    if a.cmd == "split":
        files = split_ensemble(a.ensemble, a.outdir, a.prefix, window=a.window, max_n=a.max_n)
        print(f"{len(files)} conformers written to {a.outdir}")
    elif a.cmd == "inputs":
        out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
        for f in a.xyz:
            mol = read_multixyz(f)[0]
            write_input(mol, out / (Path(f).stem + ".com"), a.route, a.charge, a.mult,
                        nproc=a.nproc, mem=a.mem, link1_route=a.link1_route)
        print(f"{len(a.xyz)} inputs written to {out}")
    elif a.cmd == "parse":
        rows = []
        for f in a.logs:
            r = read_output(f)
            row = r.summary()
            q = {"nbo": r.nbo_charges, "mulliken": r.mulliken_charges}.get(a.charges, [])
            row |= {f"q_{i + 1}": v for i, v in enumerate(q)}
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(a.out, index=False)
        bad = (~df.normal_termination).sum() + (df.n_imag.fillna(0) > 0).sum()
        print(f"{len(df)} outputs parsed -> {a.out}" + (f"  ({bad} with errors or imaginary frequencies)" if bad else ""))
    elif a.cmd == "aggregate":
        df = pd.read_csv(a.table)
        if a.id not in df.columns:
            df["molecule"] = df["name"].str.extract(a.id, expand=False)
            a.id = "molecule"
        df = df[df.get("normal_termination", True) & (df.get("n_imag", 0).fillna(0) == 0)]
        num = [c for c in df.select_dtypes("number").columns
               if c not in {a.energy, "E_scf", "E_zpe", "H", "G", "n_imag", "lowest_freq"}]
        res = aggregate(df, a.id, a.energy, num, a.temperature)
        res.to_csv(a.out, index=False)
        print(f"{len(res)} molecules -> {a.out}")


if __name__ == "__main__":
    main()

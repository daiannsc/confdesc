"""Conformer-ensemble handling: energy-window filtering and Boltzmann aggregation of descriptors."""
from pathlib import Path

import numpy as np
import pandas as pd

from .xyz import HARTREE_TO_KCAL, Molecule, read_multixyz, write_xyz

R_KCAL = 1.987204e-3  # kcal/(mol K)


def select_conformers(mols: list[Molecule], window: float = 3.0, max_n: int | None = None) -> list[Molecule]:
    """Keep conformers within `window` kcal/mol of the lowest one (energies in Hartree), lowest first."""
    if any(m.energy is None for m in mols):
        raise ValueError("all conformers need an energy in the comment line")
    e = np.array([m.energy for m in mols])
    rel = (e - e.min()) * HARTREE_TO_KCAL
    order = [i for i in np.argsort(rel) if rel[i] <= window]
    return [mols[i] for i in order[:max_n]]


def split_ensemble(path: str | Path, outdir: str | Path, prefix: str = "conf", **kw) -> list[Path]:
    """Split a CREST `crest_conformers.xyz` into individual XYZ files (conf001.xyz, ...)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return [write_xyz(m, outdir / f"{prefix}{k + 1:03d}.xyz") for k, m in enumerate(select_conformers(read_multixyz(path), **kw))]


def boltzmann_weights(energies, temperature: float = 298.15) -> np.ndarray:
    """Energies in Hartree -> normalized Boltzmann populations."""
    rel = (np.asarray(energies, float) - np.min(energies)) * HARTREE_TO_KCAL
    w = np.exp(-rel / (R_KCAL * temperature))
    return w / w.sum()


def aggregate(df: pd.DataFrame, id_col: str, energy_col: str = "G", descriptors: list[str] | None = None,
              temperature: float = 298.15) -> pd.DataFrame:
    """One row per conformer -> one row per molecule with <desc>_Boltz, _min, _max and _low_E
    (value in the lowest-energy conformer). Conformers without energy are dropped."""
    df = df.dropna(subset=[energy_col])
    if descriptors is None:
        descriptors = [c for c in df.select_dtypes("number").columns if c != energy_col]
    rows = []
    for mol_id, g in df.groupby(id_col, sort=False):
        w = boltzmann_weights(g[energy_col].to_numpy(), temperature)
        low = g[energy_col].to_numpy().argmin()
        row = {id_col: mol_id, "n_conformers": len(g)}
        for d in descriptors:
            v = g[d].to_numpy(float)
            row |= {f"{d}_Boltz": float(w @ v), f"{d}_min": v.min(), f"{d}_max": v.max(), f"{d}_low_E": v[low]}
        rows.append(row)
    return pd.DataFrame(rows)

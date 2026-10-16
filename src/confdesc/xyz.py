"""Minimal molecule container and XYZ / multi-XYZ (CREST ensemble) I/O."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SYMBOLS = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
           "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
           "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
           "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
           "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi"]

HARTREE_TO_KCAL = 627.509474


@dataclass
class Molecule:
    symbols: list[str]
    coords: np.ndarray          # (n, 3) Angstrom
    comment: str = ""
    energy: float | None = None  # Hartree, if known (e.g. from a CREST ensemble)

    def to_block(self) -> str:
        return "\n".join(f"{s:<3s}{x:16.8f}{y:16.8f}{z:16.8f}" for s, (x, y, z) in zip(self.symbols, self.coords))

    def to_xyz(self) -> str:
        return f"{len(self.symbols)}\n{self.comment}\n{self.to_block()}\n"

    def distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def angle(self, i: int, j: int, k: int) -> float:
        """Angle i-j-k in degrees (j is the vertex)."""
        a, b = self.coords[i] - self.coords[j], self.coords[k] - self.coords[j]
        return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))

    def dihedral(self, i: int, j: int, k: int, l: int) -> float:
        p = self.coords[[i, j, k, l]]
        b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
        b1 /= np.linalg.norm(b1)
        v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
        return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def read_multixyz(path: str | Path) -> list[Molecule]:
    """Read a (multi-)XYZ file. If the comment line starts with a number (CREST/xtb style), it is the energy."""
    lines = Path(path).read_text().splitlines()
    mols, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        comment = lines[i + 1].strip()
        block = [l.split() for l in lines[i + 2:i + 2 + n]]
        try:
            energy = float(comment.split()[0])
        except (ValueError, IndexError):
            energy = None
        mols.append(Molecule([b[0] for b in block], np.array([[float(x) for x in b[1:4]] for b in block]),
                             comment, energy))
        i += 2 + n
    return mols


def write_xyz(mol: Molecule, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(mol.to_xyz())
    return path

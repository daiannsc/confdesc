"""Read Gaussian 16 output files and write Gaussian input files."""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .xyz import SYMBOLS, Molecule

_FLOAT = re.compile(r"-?\d+\.\d+")


@dataclass
class GaussianResult:
    name: str
    normal_termination: bool = False
    scf_energy: float | None = None          # Hartree, last "SCF Done"
    zpe_energy: float | None = None          # E + ZPE
    enthalpy: float | None = None            # H
    free_energy: float | None = None         # G
    homo: float | None = None                # Hartree
    lumo: float | None = None
    frequencies: list[float] = field(default_factory=list)
    nbo_charges: list[float] = field(default_factory=list)
    mulliken_charges: list[float] = field(default_factory=list)
    geometry: Molecule | None = None         # last standard (or input) orientation

    @property
    def n_imaginary(self) -> int:
        return sum(f < 0 for f in self.frequencies)

    def summary(self) -> dict:
        """Flat dict of scalar properties, ready for a pandas row."""
        gap = self.lumo - self.homo if self.homo is not None and self.lumo is not None else None
        return {
            "name": self.name, "normal_termination": self.normal_termination,
            "E_scf": self.scf_energy, "E_zpe": self.zpe_energy, "H": self.enthalpy, "G": self.free_energy,
            "HOMO": self.homo, "LUMO": self.lumo, "gap": gap,
            "n_imag": self.n_imaginary if self.frequencies else None,
            "lowest_freq": min(self.frequencies) if self.frequencies else None,
        }


def _last_float(line: str) -> float:
    return float(_FLOAT.findall(line)[-1])


def _read_orientation(lines, start):
    """Parse a 'Standard orientation:' / 'Input orientation:' table starting at header line `start`."""
    symbols, coords = [], []
    i = start + 5  # title, dashes, 2 header lines, dashes
    while not lines[i].lstrip().startswith("---"):
        p = lines[i].split()
        symbols.append(SYMBOLS[int(p[1])])
        coords.append([float(x) for x in p[-3:]])
        i += 1
    return Molecule(symbols, np.array(coords))


def _read_nbo(lines, start):
    charges = []
    i = start
    while not lines[i].lstrip().startswith("---"):
        i += 1
    i += 1
    while not lines[i].lstrip().startswith("==="):
        charges.append(float(lines[i].split()[2]))
        i += 1
    return charges


def _read_mulliken(lines, start):
    charges = []
    i = start + 2  # title + column index line
    while "Sum of Mulliken" not in lines[i]:
        charges.append(float(lines[i].split()[2]))
        i += 1
    return charges


def read_output(path: str | Path) -> GaussianResult:
    """Parse one Gaussian .log/.out. Last occurrence wins (final geometry / final SCF)."""
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    res = GaussianResult(name=path.stem)
    occ, virt, block_open = [], [], False
    std_geom = inp_geom = None
    for i, line in enumerate(lines):
        if block_open and "eigenvalues" not in line:
            block_open = False
        if "SCF Done:" in line:
            res.scf_energy = float(line.split("=")[1].split()[0])
        elif "Sum of electronic and zero-point Energies=" in line:
            res.zpe_energy = _last_float(line)
        elif "Sum of electronic and thermal Enthalpies=" in line:
            res.enthalpy = _last_float(line)
        elif "Sum of electronic and thermal Free Energies=" in line:
            res.free_energy = _last_float(line)
        elif line.startswith(" Frequencies --"):
            res.frequencies += [float(x) for x in _FLOAT.findall(line)]
        elif "Alpha  occ. eigenvalues" in line:
            if not block_open:  # new orbital printout: keep only the last one
                occ, virt, block_open = [], [], True
            occ += [float(x) for x in _FLOAT.findall(line.split("--")[1])]
        elif "Alpha virt. eigenvalues" in line:
            virt += [float(x) for x in _FLOAT.findall(line.split("--")[1])]
        if "Standard orientation:" in line:
            std_geom = _read_orientation(lines, i)
        elif "Input orientation:" in line:
            inp_geom = _read_orientation(lines, i)
        elif "Summary of Natural Population Analysis:" in line:
            res.nbo_charges = _read_nbo(lines, i)
        elif line.strip() in ("Mulliken charges:", "Mulliken charges and spin densities:"):
            res.mulliken_charges = _read_mulliken(lines, i)
        elif "Normal termination of Gaussian" in line:
            res.normal_termination = True
    if occ:
        res.homo = occ[-1]
    if virt:
        res.lumo = virt[0]
    res.geometry = std_geom or inp_geom  # NoSymm jobs only print the input orientation
    return res


def write_input(mol: Molecule, path: str | Path, route: str, charge: int = 0, mult: int = 1,
                title: str | None = None, nproc: int | None = None, mem: str | None = None,
                link1_route: str | None = None, tail: str = "") -> Path:
    """Write a Gaussian input. If `link1_route` is given, a second job reads the geometry from the checkpoint
    (typical opt+freq followed by single point / NBO)."""
    path = Path(path)
    name = path.stem
    head = []
    if nproc:
        head.append(f"%nprocshared={nproc}")
    if mem:
        head.append(f"%mem={mem}")
    head.append(f"%chk={name}.chk")
    body = [*head, f"#P {route}", "", title or name, "", f"{charge} {mult}", mol.to_block(), ""]
    if tail:
        body += [tail.strip("\n"), ""]
    if link1_route:
        body += ["--Link1--", *head, f"#P {link1_route} Geom=Check Guess=Read", "", f"{title or name} (link1)", "",
                 f"{charge} {mult}", ""]
        if tail:
            body += [tail.strip("\n"), ""]
    path.write_text("\n".join(body) + "\n")
    return path

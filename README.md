# confdesc

From conformer ensembles and Gaussian outputs to ML-ready descriptor tables.

A small, dependency-light toolkit for the routine part of data-driven computational chemistry:

```
CREST ensemble ──split──► conformer XYZ ──inputs──► Gaussian .com
                                                        │ (run)
descriptor table ◄──aggregate── per-conformer table ◄──parse── .log
 (X_Boltz, X_min, X_max, X_low_E)
```

- **split** a CREST `crest_conformers.xyz` within an energy window
- **inputs**: write Gaussian inputs (optionally a `--Link1--` single point / NBO job on the optimized geometry)
- **parse** Gaussian 16 outputs: termination, E / ZPE / H / G, HOMO / LUMO, frequencies, NBO and Mulliken charges, final geometry
- **aggregate** conformer descriptors per molecule: Boltzmann-weighted, min, max and lowest-energy-conformer values

Only `numpy` and `pandas` are required.

## Install

```bash
pip install git+https://github.com/daiannsc/confdesc
```

## Command line

```bash
confdesc split   crest_conformers.xyz -o confs/ --window 3
confdesc inputs  confs/*.xyz -o inputs/ --nproc 16 --mem 32GB \
                 --route "B3LYP/def2SVP EmpiricalDispersion=GD3BJ Opt Freq" \
                 --link1-route "PBE0/def2TZVP EmpiricalDispersion=GD3BJ Pop=NBO"
# ... run Gaussian ...
confdesc parse   logs/*.log -o conformers.csv --charges nbo
confdesc aggregate conformers.csv --id "^(.*)_conf\d+$" -o descriptors.csv
```

`parse` flags outputs that did not terminate normally or have imaginary frequencies; `aggregate` skips them.

## Python

```python
from confdesc import read_output, aggregate

r = read_output("mol1_conf001.log")
r.free_energy, r.homo, r.lumo, r.n_imaginary
r.geometry.distance(0, 1), r.geometry.angle(1, 0, 2)
r.nbo_charges
```

Geometric descriptors (distances, angles, dihedrals) can be added per conformer with `Molecule.distance / angle / dihedral`
before calling `aggregate`.

## Tests

```bash
pip install -e ".[test]" && pytest
```

## License

MIT

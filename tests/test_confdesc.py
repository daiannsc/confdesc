from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from confdesc import aggregate, boltzmann_weights, read_multixyz, read_output, select_conformers, write_input
from confdesc.cli import main

DATA = Path(__file__).parent / "data"

ENSEMBLE = """3
  -5.07000000
O 0.0 0.0 0.1
H 0.0 0.76 -0.48
H 0.0 -0.76 -0.48
3
  -5.06900000
O 0.0 0.0 0.2
H 0.0 0.70 -0.40
H 0.0 -0.70 -0.40
3
  -5.06000000
O 0.0 0.0 0.3
H 0.0 0.60 -0.30
H 0.0 -0.60 -0.30
"""


def test_read_output():
    r = read_output(DATA / "water_conf001.log")
    assert r.normal_termination
    assert r.scf_energy == pytest.approx(-76.36)          # last SCF wins
    assert r.free_energy == pytest.approx(-76.356)
    assert r.homo == pytest.approx(-0.29) and r.lumo == pytest.approx(0.02)  # last orbital block
    assert r.n_imaginary == 0 and len(r.frequencies) == 3
    assert r.nbo_charges == pytest.approx([-0.91, 0.455, 0.455])
    assert r.mulliken_charges == pytest.approx([-0.7, 0.35, 0.35])
    assert r.geometry.symbols == ["O", "H", "H"]
    assert r.geometry.coords[0, 2] == pytest.approx(0.1173)  # standard orientation
    assert r.geometry.angle(1, 0, 2) == pytest.approx(104.48, abs=0.01)


def test_select_conformers(tmp_path):
    f = tmp_path / "crest_conformers.xyz"
    f.write_text(ENSEMBLE)
    mols = read_multixyz(f)
    assert len(mols) == 3 and mols[0].energy == pytest.approx(-5.07)
    kept = select_conformers(mols, window=3.0)  # third one is ~6.3 kcal/mol above
    assert len(kept) == 2


def test_boltzmann_and_aggregate():
    w = boltzmann_weights([0.0, 1.0 / 627.509474])  # 1 kcal/mol apart
    assert w[0] / w[1] == pytest.approx(np.exp(1 / (1.987204e-3 * 298.15)))
    df = pd.DataFrame({"mol": ["a", "a", "b"], "G": [-1.0, -0.99, -2.0], "x": [1.0, 3.0, 5.0]})
    out = aggregate(df, "mol", "G", ["x"]).set_index("mol")
    assert out.loc["a", "x_low_E"] == 1.0 and out.loc["a", "x_max"] == 3.0
    assert out.loc["a", "x_Boltz"] == pytest.approx(1.0, abs=1e-3)  # second conformer ~6 kcal/mol higher
    assert out.loc["b", "n_conformers"] == 1


def test_write_input(tmp_path):
    mol = read_output(DATA / "water_conf001.log").geometry
    f = write_input(mol, tmp_path / "w.com", "B3LYP/def2SVP Opt Freq", nproc=4, link1_route="PBE0/def2TZVP Pop=NBO")
    text = f.read_text()
    assert "%chk=w.chk" in text and "--Link1--" in text and "Geom=Check Guess=Read" in text
    assert text.count("0 1") == 2


def test_cli_end_to_end(tmp_path):
    (tmp_path / "ens.xyz").write_text(ENSEMBLE)
    main(["split", str(tmp_path / "ens.xyz"), "-o", str(tmp_path / "confs")])
    assert len(list((tmp_path / "confs").glob("*.xyz"))) == 2
    main(["parse", str(DATA / "water_conf001.log"), "-o", str(tmp_path / "c.csv"), "--charges", "nbo"])
    main(["aggregate", str(tmp_path / "c.csv"), "--id", r"^(.*)_conf\d+$", "-o", str(tmp_path / "d.csv")])
    d = pd.read_csv(tmp_path / "d.csv")
    assert d.molecule.tolist() == ["water"] and d["q_1_Boltz"].iloc[0] == pytest.approx(-0.91)

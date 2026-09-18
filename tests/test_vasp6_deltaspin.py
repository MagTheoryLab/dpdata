from __future__ import annotations

import os
import tempfile
import unittest
import warnings

import numpy as np
from context import dpdata

OUTCAR = os.path.join("poscars", "OUTCAR.fe2.deltaspin")

MOMENT_HEADER = (
    "   ion elem cx cy cz                Mx                My                Mz               |M|\n"
)
SECOND_MOMENT_ROW = (
    "     2   Fe  1  1  1     -0.3500000098      0.0000000000      2.6770000828      2.6997832229\n"
)
SECOND_FORCE_ROW = (
    "     2   Fe  1  1  1     -0.0072805932      0.0000003982      0.0518628325\n"
)
SECOND_FORCE_ROW_UNCONSTRAINED = (
    "     2   Fe  0  0  0      0.0000000000      0.0000000000      0.0000000000\n"
)


def patched_outcar(directory, replacements):
    """Write a copy of the fixture with a few lines replaced."""
    with open(OUTCAR) as fp:
        text = fp.read()
    for old, new in replacements:
        assert old in text, f"fixture does not contain {old!r}"
        text = text.replace(old, new, 1)
    path = os.path.join(directory, "OUTCAR")
    with open(path, "w") as fp:
        fp.write(text)
    return path


class TestVasp6DeltaSpinOutcar(unittest.TestCase):
    def setUp(self):
        self.system = dpdata.LabeledSystem(OUTCAR, fmt="vasp6_deltaspin/outcar")

    def test_frame(self):
        self.assertEqual(self.system.get_nframes(), 1)
        self.assertEqual(self.system.get_natoms(), 2)
        self.assertEqual(self.system.data["atom_names"], ["Fe"])
        self.assertEqual(list(self.system.data["atom_types"]), [0, 0])
        np.testing.assert_allclose(self.system["energies"], [-14.38633426])
        np.testing.assert_allclose(
            self.system["forces"][0],
            [[0.018706, -0.016027, -0.005289], [-0.018706, 0.016027, 0.005289]],
            atol=1e-9,
        )

    def test_spins(self):
        np.testing.assert_allclose(
            self.system["spins"][0],
            [
                [0.3500000138, -0.0000000001, 2.6770001023],
                [-0.3500000098, 0.0000000000, 2.6770000828],
            ],
            atol=1e-12,
        )

    def test_force_mags(self):
        np.testing.assert_allclose(
            self.system["force_mags"][0],
            [
                [0.0072761923, 0.0000003343, 0.0518581897],
                [-0.0072805932, 0.0000003982, 0.0518628325],
            ],
            atol=1e-12,
        )

    def test_matches_plain_outcar_reader(self):
        plain = dpdata.LabeledSystem(OUTCAR, fmt="vasp/outcar")
        np.testing.assert_allclose(plain["energies"], self.system["energies"])
        np.testing.assert_allclose(plain["forces"], self.system["forces"])
        np.testing.assert_allclose(
            plain["virials"], self.system.data["virials"], atol=1e-6
        )

    def test_cell_and_coords(self):
        np.testing.assert_allclose(
            self.system["cells"][0], np.eye(3) * 2.87, atol=1e-9
        )
        np.testing.assert_allclose(
            self.system["coords"][0],
            [[0.0, 0.0, 0.0], [1.43701, 1.43328, 1.43586]],
            atol=1e-5,
        )


class TestVasp6DeltaSpinPartialConstraint(unittest.TestCase):
    def test_unconstrained_atom_has_no_moment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = patched_outcar(
                directory,
                [
                    (SECOND_MOMENT_ROW, ""),
                    (SECOND_FORCE_ROW, SECOND_FORCE_ROW_UNCONSTRAINED),
                ],
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                system = dpdata.LabeledSystem(path, fmt="vasp6_deltaspin/outcar")

        self.assertTrue(
            any("not moment constrained" in str(warning.message) for warning in caught)
        )
        np.testing.assert_allclose(
            system["spins"][0],
            [[0.3500000138, -0.0000000001, 2.6770001023], [0.0, 0.0, 0.0]],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            system["force_mags"][0],
            [[0.0072761923, 0.0000003343, 0.0518581897], [0.0, 0.0, 0.0]],
            atol=1e-12,
        )


class TestVasp6DeltaSpinConvergence(unittest.TestCase):
    UNCONVERGED = [
        (" Moment constraint      : reached", " Moment constraint      : not reached")
    ]

    def test_unconverged_constraint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = patched_outcar(directory, self.UNCONVERGED)
            with self.assertRaises(ValueError):
                dpdata.LabeledSystem(path, fmt="vasp6_deltaspin/outcar")

    def test_unconverged_constraint_can_be_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            path = patched_outcar(directory, self.UNCONVERGED)
            system = dpdata.LabeledSystem(
                path, fmt="vasp6_deltaspin/outcar", convergence_check=False
            )
        self.assertEqual(system.get_nframes(), 1)
        np.testing.assert_allclose(
            system["spins"][0][0], [0.3500000138, -0.0000000001, 2.6770001023]
        )


class TestVasp6DeltaSpinTableFormat(unittest.TestCase):
    def test_older_table_header_is_rejected(self):
        """Only the current ``elem cx cy cz`` table layout is supported."""
        old_header = (
            "   ion         element                   x                   y                   z\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = patched_outcar(directory, [(MOMENT_HEADER, old_header)])
            with self.assertRaises(ValueError) as context:
                dpdata.LabeledSystem(path, fmt="vasp6_deltaspin/outcar")
        self.assertIn("table header", str(context.exception))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
from comp_sys import CompSys, IsPBC
from context import dpdata


class TestReplicate123(unittest.TestCase, CompSys, IsPBC):
    def setUp(self):
        system_1_origin = dpdata.System("poscars/POSCAR.SiC", fmt="vasp/poscar")
        self.system_1 = system_1_origin.replicate(
            (
                1,
                2,
                3,
            )
        )
        self.system_2 = dpdata.System(
            "poscars/POSCAR.SiC.replicate123", fmt="vasp/poscar"
        )
        self.places = 6


class TestReplicate123_not_change_origin(unittest.TestCase, CompSys, IsPBC):
    def setUp(self):
        self.system_1 = dpdata.System("poscars/POSCAR.SiC", fmt="vasp/poscar")
        self.system_1.replicate(
            (
                1,
                2,
                3,
            )
        )
        self.system_2 = dpdata.System("poscars/POSCAR.SiC", fmt="vasp/poscar")
        self.places = 6


class TestReplicateTriclinicBox(unittest.TestCase, CompSys, IsPBC):
    def setUp(self):
        self.system_1 = dpdata.System()
        self.system_1.data["atom_names"] = ["foo", "bar"]
        self.system_1.data["atom_types"] = np.array([1, 0], dtype=int)
        self.system_1.data["atom_numbs"] = [1, 1]
        self.system_1.data["cells"] = np.array(
            [10, 0, 0, 0, 10, 0, 0, 0, 10], dtype=float
        ).reshape(1, 3, 3)
        self.system_1.data["coords"] = np.array(
            [0, 0, 0, 0, 0, 1], dtype=float
        ).reshape(1, 2, 3)
        self.system_1 = self.system_1.replicate([2, 1, 1])

        self.system_2 = dpdata.System()
        self.system_2.data["atom_names"] = ["foo", "bar"]
        self.system_2.data["atom_types"] = np.array([1, 1, 0, 0], dtype=int)
        self.system_2.data["atom_numbs"] = [2, 2]
        self.system_2.data["cells"] = np.array(
            [20, 0, 0, 0, 10, 0, 0, 0, 10], dtype=float
        ).reshape(1, 3, 3)
        self.system_2.data["coords"] = np.array(
            [0, 0, 0, 10, 0, 0, 0, 0, 1, 10, 0, 1], dtype=float
        ).reshape(1, 4, 3)
        self.places = 6


class TestReplicateSpins(unittest.TestCase):
    def test_spins_follow_atoms_in_each_frame(self):
        system = dpdata.System("poscars/POSCAR.SiC", fmt="vasp/poscar")
        system.append(system.copy())
        natoms = system.get_natoms()
        spins = np.arange(2 * natoms * 3, dtype=float).reshape(2, natoms, 3)
        system.data["spins"] = spins.copy()

        for ncopy in [(1, 1, 1), (2, 3, 2)]:
            with self.subTest(ncopy=ncopy):
                replicated = system.replicate(ncopy)
                self.assertEqual(
                    replicated["spins"].shape, (2, natoms * np.prod(ncopy), 3)
                )
                for frame in range(2):
                    for atom in range(natoms):
                        for copy, shift in enumerate(np.ndindex(ncopy)):
                            index = atom * np.prod(ncopy) + copy
                            np.testing.assert_array_equal(
                                replicated["spins"][frame, index], spins[frame, atom]
                            )
                            np.testing.assert_allclose(
                                replicated["coords"][frame, index],
                                system["coords"][frame, atom]
                                + np.array(shift) @ system["cells"][frame],
                            )
                replicated["spins"][0, 0, 0] = -100
                np.testing.assert_array_equal(system["spins"], spins)


if __name__ == "__main__":
    unittest.main()

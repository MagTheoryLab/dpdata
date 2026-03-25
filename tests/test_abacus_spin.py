from __future__ import annotations

import os
import shutil
import unittest

import numpy as np
from context import dpdata


class TestABACUSSpin(unittest.TestCase):
    def setUp(self):
        self.dump_path = "abacus.spin/dump"
        os.makedirs(self.dump_path, exist_ok=True)

    def tearDown(self):
        if os.path.isdir(self.dump_path):
            shutil.rmtree(self.dump_path)
        if os.path.isfile("abacus.spin/INPUT"):
            os.remove("abacus.spin/INPUT")

    def test_scf(self):
        os.system("cp abacus.spin/INPUT.scf abacus.spin/INPUT")
        mysys = dpdata.LabeledSystem("abacus.spin", fmt="abacus/scf")
        data = mysys.data
        self.assertAlmostEqual(data["energies"][0], -6818.719409466637)
        np.testing.assert_almost_equal(
            data["spins"][0],
            [
                [-0.0000002724, -0.0000001728, 2.4000001004],
                [-0.0000003180, -0.0000002299, 2.3999994597],
            ],
            decimal=8,
        )
        np.testing.assert_almost_equal(
            data["force_mags"][0],
            [
                [-0.0000175013, -0.0000418680, -0.3669618965],
                [-0.0000161517, -0.0000195198, -0.3669821632],
            ],
            decimal=8,
        )

        # dump to deepmd-npy
        mysys.to(file_name=self.dump_path, fmt="deepmd/npy")
        self.assertTrue(os.path.isfile(f"{self.dump_path}/set.000/spin.npy"))
        self.assertTrue(os.path.isfile(f"{self.dump_path}/set.000/force_mag.npy"))

        sys2 = dpdata.LabeledSystem(self.dump_path, fmt="deepmd/npy")
        np.testing.assert_almost_equal(data["spins"], sys2.data["spins"], decimal=8)
        np.testing.assert_almost_equal(
            data["force_mags"], sys2.data["force_mags"], decimal=8
        )

    def test_scf_nspin2(self):
        os.system("cp abacus.spin/INPUT.scf.nspin2 abacus.spin/INPUT")
        mysys = dpdata.LabeledSystem("abacus.spin", fmt="abacus/scf")
        data = mysys.data
        self.assertAlmostEqual(data["energies"][0], -6818.719409466637)
        np.testing.assert_almost_equal(
            data["spins"][0],
            [
                [0, 0, 2.4000001004],
                [0, 0, 2.3999994597],
            ],
            decimal=8,
        )
        np.testing.assert_almost_equal(
            data["force_mags"][0],
            [
                [0, 0, -0.3669618965],
                [0, 0, -0.3669821632],
            ],
            decimal=8,
        )

        # dump to deepmd-npy
        mysys.to(file_name=self.dump_path, fmt="deepmd/npy")
        self.assertTrue(os.path.isfile(f"{self.dump_path}/set.000/spin.npy"))
        self.assertTrue(os.path.isfile(f"{self.dump_path}/set.000/force_mag.npy"))

        sys2 = dpdata.LabeledSystem(self.dump_path, fmt="deepmd/npy")
        np.testing.assert_almost_equal(data["spins"], sys2.data["spins"], decimal=8)
        np.testing.assert_almost_equal(
            data["force_mags"], sys2.data["force_mags"], decimal=8
        )

    def test_relax(self):
        os.system("cp abacus.spin/INPUT.relax abacus.spin/INPUT")
        mysys = dpdata.LabeledSystem("abacus.spin", fmt="abacus/relax")
        data = mysys.data
        spins_ref = np.array(
            [
                [
                    [1.16909819, 1.16895965, 1.16895485],
                    [1.16827825, 1.16832716, 1.16836899],
                ],
                [
                    [1.25007143, 1.25006167, 1.25004587],
                    [1.25015764, 1.2501678, 1.25018344],
                ],
                [
                    [1.24984994, 1.24977108, 1.24978313],
                    [1.24996533, 1.2500441, 1.25003208],
                ],
            ]
        )
        magforces_ref = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [
                    [-0.16734626, -0.16735378, -0.16735617],
                    [-0.16836467, -0.16835897, -0.16835625],
                ],
                [
                    [-0.16573406, -0.16574627, -0.1657445],
                    [-0.16619489, -0.16617948, -0.16618272],
                ],
            ]
        )
        self.assertEqual(len(data["spins"]), 3)
        self.assertEqual(len(data["force_mags"]), 3)
        np.testing.assert_almost_equal(data["spins"], spins_ref, decimal=8)
        np.testing.assert_almost_equal(data["force_mags"], magforces_ref, decimal=8)

        # dump to deepmd-npy
        mysys.to(file_name=self.dump_path, fmt="deepmd/npy")
        self.assertTrue(os.path.isfile(f"{self.dump_path}/set.000/spin.npy"))
        self.assertTrue(os.path.isfile(f"{self.dump_path}/set.000/force_mag.npy"))

        sys2 = dpdata.LabeledSystem(self.dump_path, fmt="deepmd/npy")
        np.testing.assert_almost_equal(data["spins"], sys2.data["spins"], decimal=8)
        np.testing.assert_almost_equal(
            data["force_mags"], sys2.data["force_mags"], decimal=8
        )

    def test_md(self):
        os.system("cp abacus.spin/INPUT.md abacus.spin/INPUT")
        mysys = dpdata.LabeledSystem("abacus.spin", fmt="abacus/md")
        data = mysys.data
        spins_ref = np.array(
            [
                [
                    [1.16909819, 1.16895965, 1.16895485],
                    [1.16827825, 1.16832716, 1.16836899],
                ],
                [
                    [1.2500362, 1.25007501, 1.2500655],
                    [1.25019078, 1.25015253, 1.25016188],
                ],
                [
                    [1.24985138, 1.24976901, 1.2497695],
                    [1.24996388, 1.25004618, 1.25004561],
                ],
                [
                    [1.24982513, 1.24985445, 1.24985336],
                    [1.25005073, 1.25001814, 1.25002065],
                ],
            ]
        )
        magforces_ref = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [
                    [-0.16747275, -0.16747145, -0.16746776],
                    [-0.16853881, -0.16853935, -0.16854119],
                ],
                [
                    [-0.16521817, -0.16523256, -0.16523212],
                    [-0.16549418, -0.16547867, -0.16547913],
                ],
                [
                    [-0.16141172, -0.16140644, -0.1614127],
                    [-0.15901519, -0.15905932, -0.15904824],
                ],
            ]
        )
        self.assertEqual(len(data["spins"]), 4)
        self.assertEqual(len(data["force_mags"]), 4)
        np.testing.assert_almost_equal(data["spins"], spins_ref, decimal=8)
        np.testing.assert_almost_equal(data["force_mags"], magforces_ref, decimal=8)

        # dump to deepmd-npy
        mysys.to(file_name=self.dump_path, fmt="deepmd/npy")
        self.assertTrue(os.path.isfile(f"{self.dump_path}/set.000/spin.npy"))
        self.assertTrue(os.path.isfile(f"{self.dump_path}/set.000/force_mag.npy"))

        sys2 = dpdata.LabeledSystem(self.dump_path, fmt="deepmd/npy")
        np.testing.assert_almost_equal(data["spins"], sys2.data["spins"], decimal=8)
        np.testing.assert_almost_equal(
            data["force_mags"], sys2.data["force_mags"], decimal=8
        )

    def test_read_stru_spin(self):
        mysys = dpdata.System("abacus.spin/STRU.spin", fmt="abacus/stru")
        self.assertTrue("spins" in mysys.data)
        print(mysys.data["spins"])

        """
    0.0000000000     0.000000000     0.000000000 mag 0 0 2
    0.1000000000     0.1000000000     0.1000000000 mag 3
    0.2000000000     0.2000000000     0.2000000000 mag 3 angle1 90
    0.3000000000     0.3000000000     0.3000000000 mag 3 4 0 angle1 90 angle2 90
        """
        np.testing.assert_almost_equal(mysys.data["spins"][0][0], [0, 0, 2], decimal=8)
        np.testing.assert_almost_equal(mysys.data["spins"][0][1], [0, 0, 3], decimal=8)
        np.testing.assert_almost_equal(mysys.data["spins"][0][2], [3, 0, 0], decimal=8)
        np.testing.assert_almost_equal(mysys.data["spins"][0][3], [0, 5, 0], decimal=8)


class TestABACUSRelaxSpinUnconvergedFrame(unittest.TestCase):
    def setUp(self):
        self.input_path = "abacus.spin/INPUT"
        self.log_path = "abacus.spin/OUT.ABACUS/running_relax.log"
        self.original_log = None
        shutil.copy("abacus.spin/INPUT.relax", self.input_path)

        with open(self.log_path) as fp:
            self.original_log = fp.read()

        modified_log = self.original_log.replace(
            " final etot is -6825.6858753 eV",
            " convergence has not been achieved @_@",
            1,
        )
        with open(self.log_path, "w") as fp:
            fp.write(modified_log)

    def tearDown(self):
        if self.original_log is not None:
            with open(self.log_path, "w") as fp:
                fp.write(self.original_log)
        if os.path.isfile(self.input_path):
            os.remove(self.input_path)

    def test_relax(self):
        system = dpdata.LabeledSystem("abacus.spin", fmt="abacus/relax")
        data = system.data

        self.assertEqual(data["coords"].shape, (2, 2, 3))
        self.assertEqual(data["spins"].shape, (2, 2, 3))
        self.assertEqual(data["force_mags"].shape, (2, 2, 3))
        np.testing.assert_almost_equal(
            data["energies"], [-6825.60446372, -6825.60469264], decimal=8
        )
        np.testing.assert_almost_equal(
            data["spins"],
            [
                [
                    [1.25007143, 1.25006167, 1.25004587],
                    [1.25015764, 1.2501678, 1.25018344],
                ],
                [
                    [1.24984994, 1.24977108, 1.24978313],
                    [1.24996533, 1.2500441, 1.25003208],
                ],
            ],
            decimal=8,
        )
        np.testing.assert_almost_equal(
            data["force_mags"],
            [
                [
                    [-0.16734626, -0.16735378, -0.16735617],
                    [-0.16836467, -0.16835897, -0.16835625],
                ],
                [
                    [-0.16573406, -0.16574627, -0.1657445],
                    [-0.16619489, -0.16617948, -0.16618272],
                ],
            ],
            decimal=8,
        )


class TestABACUSMDSpinUnconvergedFrame(unittest.TestCase):
    def setUp(self):
        self.input_path = "abacus.spin/INPUT"
        self.log_path = "abacus.spin/OUT.ABACUS/running_md.log"
        self.original_log = None
        shutil.copy("abacus.spin/INPUT.md", self.input_path)

        with open(self.log_path) as fp:
            self.original_log = fp.read()

        modified_log = self.original_log.replace(
            " final etot is -6825.6858753 eV",
            " !! convergence has not been achieved @_@",
            1,
        )
        with open(self.log_path, "w") as fp:
            fp.write(modified_log)

    def tearDown(self):
        if self.original_log is not None:
            with open(self.log_path, "w") as fp:
                fp.write(self.original_log)
        if os.path.isfile(self.input_path):
            os.remove(self.input_path)

    def test_md(self):
        system = dpdata.LabeledSystem("abacus.spin", fmt="abacus/md")
        data = system.data

        self.assertEqual(data["coords"].shape, (3, 2, 3))
        self.assertEqual(data["spins"].shape, (3, 2, 3))
        self.assertEqual(data["force_mags"].shape, (3, 2, 3))
        np.testing.assert_almost_equal(
            data["energies"], [-6825.6043625, -6825.6049361, -6825.6046584], decimal=8
        )
        np.testing.assert_almost_equal(
            data["spins"],
            [
                [
                    [1.2500362, 1.25007501, 1.2500655],
                    [1.25019078, 1.25015253, 1.25016188],
                ],
                [
                    [1.24985138, 1.24976901, 1.2497695],
                    [1.24996388, 1.25004618, 1.25004561],
                ],
                [
                    [1.24982513, 1.24985445, 1.24985336],
                    [1.25005073, 1.25001814, 1.25002065],
                ],
            ],
            decimal=8,
        )
        np.testing.assert_almost_equal(
            data["force_mags"],
            [
                [
                    [-0.16747275, -0.16747145, -0.16746776],
                    [-0.16853881, -0.16853935, -0.16854119],
                ],
                [
                    [-0.16521817, -0.16523256, -0.16523212],
                    [-0.16549418, -0.16547867, -0.16547913],
                ],
                [
                    [-0.16141172, -0.16140644, -0.1614127],
                    [-0.15901519, -0.15905932, -0.15904824],
                ],
            ],
            decimal=8,
        )

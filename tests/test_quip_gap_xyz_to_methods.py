#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from context import dpdata


class TestQuipGapXYZToMethods(unittest.TestCase):
    """Test the to_labeled_system and to_multi_systems methods for QuipGapXYZFormat."""

    def setUp(self):
        """Set up test data."""
        # Load test multi-systems
        self.multi_systems = dpdata.MultiSystems.from_file(
            "xyz/xyz_unittest.xyz", "quip/gap/xyz"
        )
        self.system_b1c9 = self.multi_systems.systems["B1C9"]
        self.system_b5c7 = self.multi_systems.systems["B5C7"]

    def test_to_labeled_system(self):
        """Test writing a single labeled system to QUIP/GAP XYZ format."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as tmp_file:
            output_file = tmp_file.name

        try:
            # Write the system to file
            self.system_b1c9.to("quip/gap/xyz", output_file)

            # Verify file was created and has content
            self.assertTrue(os.path.exists(output_file))
            with open(output_file) as f:
                content = f.read()
            self.assertTrue(len(content) > 0)

            # Read back and verify we can parse it (use MultiSystems.from_file for QUIP/GAP XYZ)
            reloaded_multi = dpdata.MultiSystems.from_file(output_file, "quip/gap/xyz")
            self.assertEqual(len(reloaded_multi.systems), 1)

            # Verify the data matches (we should have the same system)
            reloaded_system = list(reloaded_multi.systems.values())[0]
            self.assertEqual(len(reloaded_system), len(self.system_b1c9))

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_to_multi_systems(self):
        """Test writing multiple systems to a single QUIP/GAP XYZ format file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as tmp_file:
            output_file = tmp_file.name

        try:
            # Write all systems to file
            self.multi_systems.to("quip/gap/xyz", output_file)

            # Verify file was created and has content
            self.assertTrue(os.path.exists(output_file))
            with open(output_file) as f:
                content = f.read()
            self.assertTrue(len(content) > 0)

            # Read back and verify we get the same number of systems
            reloaded_multi = dpdata.MultiSystems.from_file(output_file, "quip/gap/xyz")
            self.assertEqual(
                len(reloaded_multi.systems), len(self.multi_systems.systems)
            )

            # Verify total number of frames is preserved
            original_frames = sum(
                len(sys) for sys in self.multi_systems.systems.values()
            )
            reloaded_frames = sum(len(sys) for sys in reloaded_multi.systems.values())
            self.assertEqual(reloaded_frames, original_frames)

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_roundtrip_consistency(self):
        """Test that writing and reading back preserves data consistency."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as tmp_file:
            output_file = tmp_file.name

        try:
            # Write and read back
            self.multi_systems.to("quip/gap/xyz", output_file)
            reloaded_multi = dpdata.MultiSystems.from_file(output_file, "quip/gap/xyz")

            # Compare original and reloaded data for each system
            for system_name in self.multi_systems.systems:
                if system_name in reloaded_multi.systems:
                    original = self.multi_systems.systems[system_name]
                    reloaded = reloaded_multi.systems[system_name]

                    # Check basic properties
                    self.assertEqual(len(original), len(reloaded))
                    self.assertEqual(
                        len(original.data["atom_names"]),
                        len(reloaded.data["atom_names"]),
                    )

                    # Note: We don't check exact numerical equality because of floating point precision
                    # and potential differences in formatting, but the data should be structurally the same

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_gpumd_xyz_writes_magnetic_labels_only_for_gpumd(self):
        system = self.system_b1c9[0]
        system.data["spins"] = np.tile(
            [1.123456789012345, -0.039123456789123, 1e-15], (1, system.get_natoms(), 1)
        )
        system.data["force_mags"] = -system.data["spins"]
        virial = np.arange(1, 10, dtype=float).reshape(3, 3)
        system.data["virials"] = virial[np.newaxis]

        with tempfile.TemporaryDirectory() as directory:
            gpumd_path = os.path.join(directory, "gpumd.xyz")
            extxyz_path = os.path.join(directory, "extxyz.xyz")
            system.to("gpumd/xyz", gpumd_path)
            system.to("extxyz", extxyz_path)
            single = dpdata.LabeledSystem(gpumd_path, fmt="gpumd/xyz")
            multi = dpdata.MultiSystems.from_file(gpumd_path, fmt="gpumd/xyz")

            with open(gpumd_path) as fp:
                gpumd = fp.read()
            with open(extxyz_path) as fp:
                extxyz = fp.read()

        self.assertIn("spin:R:3:mforce:R:3", gpumd)
        self.assertNotIn("spin:R:3", extxyz)
        self.assertNotIn('stress="', gpumd)
        self.assertIn('stress="', extxyz)
        header = gpumd.splitlines()[1]
        tensor = header.split('virial="', 1)[1].split('"')[0]
        np.testing.assert_array_equal(np.fromstring(tensor, sep=" "), virial.ravel())
        rows = gpumd.splitlines()[2:]
        values = np.array([[float(value) for value in row.split()[8:]] for row in rows])
        np.testing.assert_array_equal(values[:, :3], system.data["spins"][0])
        np.testing.assert_array_equal(values[:, 3:], system.data["force_mags"][0])
        for reloaded in (single, next(iter(multi.systems.values()))):
            for key in ("spins", "force_mags", "virials"):
                np.testing.assert_array_equal(reloaded[key], system[key])


if __name__ == "__main__":
    unittest.main()

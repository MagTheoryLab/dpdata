from __future__ import annotations

import numpy as np

import dpdata.formats.vasp.poscar
import dpdata.formats.vasp6_deltaspin.outcar
import dpdata.formats.vasp6_deltaspin.poscar
from dpdata.format import Format
from dpdata.plugins.mag_data import register_mag_data
from dpdata.utils import open_file, uniq_atom_names


def _parse_flags(value, count, name):
    """Return ``count`` 0/1 flags from an INCAR keyword value."""
    if value is None:
        return ["1"] * count
    if isinstance(value, str):
        flags = value.split()
    else:
        flags = [str(item) for item in np.asarray(value).reshape(-1)]
    if len(flags) != count:
        raise ValueError(f"{name} must list {count} values, got {len(flags)}")
    if any(flag not in ("0", "1") for flag in flags):
        raise ValueError(f"{name} must contain only 0 and 1, got {value!r}")
    return flags


@Format.register("vasp6_deltaspin/poscar")
@Format.register("vasp6_deltaspin/contcar")
class VASP6DeltaSpinPoscarFormat(Format):
    """VASP 6 DeltaSpin POSCAR or CONTCAR together with its INCAR moments.

    A DeltaSpin structure file is only half of the input: the constraint lives
    in the INCAR, where ``MAGMOM`` holds the initial moments, ``M_DELTASPIN``
    the constrained targets, and ``DELTASPIN_ATOMS``/``DELTASPIN_COMPONENTS``
    select the constrained atoms and components. Reading returns the moments of
    the INCAR as ``spins``, and writing updates the INCAR next to the structure.
    """

    @Format.post("rot_lower_triangular")
    def from_system(self, file_name, **kwargs):
        """Load a DeltaSpin POSCAR or CONTCAR and the moments of its INCAR.

        Parameters
        ----------
        file_name : str
            POSCAR/CONTCAR input; the INCAR of the same directory provides the
            magnetic moments.
        **kwargs : dict
            Additional format arguments accepted for API compatibility.

        Returns
        -------
        dict
            System data, including ``spins`` when the INCAR lists moments.
        """
        with open_file(file_name) as fp:
            lines = [line.rstrip("\n") for line in fp]
        data = dpdata.formats.vasp.poscar.to_system_data(lines)
        values = dpdata.formats.vasp6_deltaspin.poscar.read_incar(file_name)
        moments = dpdata.formats.vasp6_deltaspin.poscar.moments_from_incar(
            values, sum(data["atom_numbs"])
        )
        if moments is not None:
            data["spins"] = moments[np.newaxis, ...]
        data = uniq_atom_names(data)
        register_mag_data(data)
        return data

    def to_system(
        self,
        data,
        file_name,
        frame_idx=0,
        deltaspin_atoms=None,
        deltaspin_components=None,
        **kwargs,
    ):
        """Dump the structure and update the DeltaSpin keys of the INCAR.

        Parameters
        ----------
        data : dict
            The system data; ``spins`` provides both the initial moments and
            the constrained targets.
        file_name : str
            POSCAR/CONTCAR output; the INCAR of the same directory is updated.
        frame_idx : int, default=0
            The index of the frame to dump.
        deltaspin_atoms : str or array-like, optional
            One 0/1 flag per atom for ``DELTASPIN_ATOMS``. All atoms are
            constrained by default.
        deltaspin_components : str or array-like, optional
            One 0/1 flag per Cartesian component for
            ``DELTASPIN_COMPONENTS``. All components are constrained by
            default.
        **kwargs : dict
            Additional format arguments accepted for API compatibility.
        """
        if "spins" not in data:
            raise ValueError(
                "vasp6_deltaspin/poscar needs 'spins' for MAGMOM and "
                "M_DELTASPIN"
            )
        spins = np.asarray(data["spins"][frame_idx], dtype=float)
        natoms = spins.shape[0]

        with open_file(file_name, "w") as fp:
            fp.write(dpdata.formats.vasp.poscar.from_system_data(data, frame_idx))

        poscar = dpdata.formats.vasp6_deltaspin.poscar
        moments = poscar.format_values(spins.reshape(-1))
        poscar.update_incar(
            file_name,
            {
                poscar.LDELTASPIN: ".TRUE.",
                poscar.DELTASPIN_ATOMS: " ".join(
                    _parse_flags(deltaspin_atoms, natoms, "deltaspin_atoms")
                ),
                poscar.DELTASPIN_COMPONENTS: " ".join(
                    _parse_flags(deltaspin_components, 3, "deltaspin_components")
                ),
                poscar.MAGMOM: moments,
                poscar.M_DELTASPIN: moments,
            },
        )


@Format.register("vasp6_deltaspin/outcar")
class VASP6DeltaSpinOutcarFormat(Format):
    """VASP 6 DeltaSpin ``OUTCAR`` labeled trajectory.

    The DeltaSpin build of VASP 6 constrains the local moments of selected
    atoms and prints the constrained moments, the magnetic forces, and the
    constraint convergence into the ``OUTCAR`` once per ionic step. This
    reader returns those observables as ``spins`` and ``force_mags`` next to
    the usual cells, coordinates, energies, forces, and virials.

    Atoms without a moment constraint have no moment label, so their ``spins``
    entry is zero. ``force_mags`` is zero for every unconstrained component by
    construction.
    """

    @Format.post("rot_lower_triangular")
    def from_labeled_system(
        self, file_name, begin=0, step=1, convergence_check=True, **kwargs
    ):
        """Load labeled ionic steps from a VASP 6 DeltaSpin OUTCAR.

        Parameters
        ----------
        file_name : str
            VASP ``OUTCAR`` file of a ``LDELTASPIN = .TRUE.`` calculation.
        begin : int, default=0
            Index of the first ionic step to load.
        step : int, default=1
            Load every ``step``-th ionic step.
        convergence_check : bool, default=True
            Drop the ionic steps whose moment constraint did not converge.
        **kwargs : dict
            Additional format arguments accepted for API compatibility.

        Returns
        -------
        dict
            Labeled trajectory data, including the constrained moments
            (``spins``) and the magnetic forces (``force_mags``).
        """
        data = {}
        (
            data["atom_names"],
            data["atom_numbs"],
            data["atom_types"],
            data["cells"],
            data["coords"],
            data["energies"],
            data["forces"],
            tmp_virial,
            data["spins"],
            data["force_mags"],
        ) = dpdata.formats.vasp6_deltaspin.outcar.get_frames(
            file_name,
            begin=begin,
            step=step,
            convergence_check=convergence_check,
        )
        if tmp_virial is not None:
            data["virials"] = tmp_virial
        # scale virial to the unit of eV
        if "virials" in data:
            v_pref = 1 * 1e3 / 1.602176621e6
            for ii in range(data["cells"].shape[0]):
                vol = np.linalg.det(np.reshape(data["cells"][ii], [3, 3]))
                data["virials"][ii] *= v_pref * vol
        data = uniq_atom_names(data)
        register_mag_data(data)

        return data

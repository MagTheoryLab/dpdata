from __future__ import annotations

import numpy as np

import dpdata.formats.vasp6_deltaspin.outcar
from dpdata.format import Format
from dpdata.plugins.abacus import register_mag_data
from dpdata.utils import uniq_atom_names


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

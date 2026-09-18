"""Registration of the data types that carry magnetic information.

``spins``, ``force_mags`` and ``hubbard_u`` are produced by several formats
(ABACUS, LAMMPS, the DeltaSpin VASP readers, and the DeePMD spin data), so
their definitions live here instead of being repeated by every plugin.

``hubbard_u`` keeps ``deepmd_name="aparam"`` because the on-site U values are
consumed as the ``aparam`` input of a DeePMD model. A user-defined data type
may claim the same ``aparam`` file with another shape; the readers therefore
only assign a stored array to the data type whose shape matches it.
"""

from __future__ import annotations

import numpy as np

import dpdata
from dpdata.data_type import Axis, DataType

MAG_DATA_TYPES = (
    DataType(
        "spins",
        np.ndarray,
        (Axis.NFRAMES, Axis.NATOMS, 3),
        required=False,
        deepmd_name="spin",
    ),
    DataType(
        "force_mags",
        np.ndarray,
        (Axis.NFRAMES, Axis.NATOMS, 3),
        required=False,
        deepmd_name="force_mag",
    ),
    DataType(
        "hubbard_u",
        np.ndarray,
        (Axis.NFRAMES, Axis.NATOMS, 1),
        required=False,
        deepmd_name="aparam",
    ),
)


def register_mag_data(data=None):
    """Register the magnetic data types on ``System`` and ``LabeledSystem``.

    Parameters
    ----------
    data : dict, optional
        When given, only the data types whose key is present in ``data`` are
        registered, which is what a reader that just loaded a file needs.
        Without it every magnetic data type is registered, which is what a
        reader of an existing data set needs to recognize those keys.
    """
    for dtype in MAG_DATA_TYPES:
        if data is not None and dtype.name not in data:
            continue
        dpdata.System.register_data_type(dtype)
        dpdata.LabeledSystem.register_data_type(dtype)

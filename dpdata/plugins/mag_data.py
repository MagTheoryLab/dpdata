"""Registration of the data types that carry magnetic information.

``spins``, ``force_mags`` and ``hubbard_u`` are produced by several formats
(ABACUS, LAMMPS, the DeltaSpin VASP readers, and the DeePMD spin data), so
their definitions live here instead of being repeated by every plugin.

``hubbard_u`` keeps ``deepmd_name="aparam"`` because the on-site U values are
consumed as the ``aparam`` input of a DeePMD model. That name is a generic
field of DeePMD-kit and of the dpdata protocol, so the alias is registered
for the data that actually carries on-site U values instead of globally: a
global alias would make ``aparam`` ambiguous for every other user of the
field, and the readers only assign a stored array to the data type whose
shape matches it.
"""

from __future__ import annotations

import os

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


def register_hubbard_u():
    """Register the ``hubbard_u`` alias of the generic ``aparam`` field.

    Readers of ABACUS, LAMMPS and DeltaSpin VASP data call this automatically
    when the file they read carries on-site U values. Call it explicitly to
    read such values back from a DeePMD data set.
    """
    for dtype in MAG_DATA_TYPES:
        if dtype.name != "hubbard_u":
            continue
        dpdata.System.register_data_type(dtype)
        dpdata.LabeledSystem.register_data_type(dtype)


def register_hubbard_u_if_stored(folder):
    """Register ``hubbard_u`` when a DeePMD data set stores an ``aparam`` file.

    Parameters
    ----------
    folder : str or os.PathLike
        DeePMD ``npy`` or ``raw`` data set directory.

    Returns
    -------
    bool
        Whether the alias was registered.
    """
    if not isinstance(folder, (str, os.PathLike)) or not os.path.isdir(folder):
        return False
    claimed = any(
        dtype.deepmd_name == "aparam"
        for dtype in (*dpdata.System.DTYPES, *dpdata.LabeledSystem.DTYPES)
    )
    if claimed:
        # The data set owner registered the generic ``aparam`` field itself;
        # do not shadow that choice with the on-site U alias.
        return False
    stored = os.path.isdir(os.path.join(folder, "set.000")) and os.path.isfile(
        os.path.join(folder, "set.000", "aparam.npy")
    )
    stored = stored or os.path.isfile(os.path.join(folder, "aparam.raw"))
    if not stored:
        return False
    register_hubbard_u()
    return True


def register_mag_data(data=None):
    """Register the magnetic data types on ``System`` and ``LabeledSystem``.

    Parameters
    ----------
    data : dict, optional
        When given, only the data types whose key is present in ``data`` are
        registered, which is what a reader that just loaded a file needs.
        Without it ``spins`` and ``force_mags`` are registered, which is what a
        reader of an existing data set needs to recognize those keys. The
        ``hubbard_u`` alias is only registered when ``data`` carries it; use
        :func:`register_hubbard_u` to read one back from a DeePMD data set.
    """
    for dtype in MAG_DATA_TYPES:
        if dtype.name == "hubbard_u" and data is None:
            # Never claim the generic ``aparam`` field on behalf of every data
            # set; the ABACUS/LAMMPS/VASP readers register it when needed.
            continue
        if data is not None and dtype.name not in data:
            continue
        dpdata.System.register_data_type(dtype)
        dpdata.LabeledSystem.register_data_type(dtype)

"""Read and write the DeltaSpin moment keys of a VASP INCAR.

A DeltaSpin calculation keeps the initial moments in ``MAGMOM`` and the
constrained targets in ``M_DELTASPIN``. ``DELTASPIN_ATOMS`` selects the
constrained atoms and ``DELTASPIN_COMPONENTS`` the constrained components,
both as space separated 0/1 lists; ``LDELTASPIN`` switches the constraint on.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from dpdata.formats.vasp.poscar import from_system_data as poscar_from_system_data
from dpdata.utils import open_file

MAGMOM = "MAGMOM"
M_DELTASPIN = "M_DELTASPIN"
DELTASPIN_ATOMS = "DELTASPIN_ATOMS"
DELTASPIN_COMPONENTS = "DELTASPIN_COMPONENTS"
LDELTASPIN = "LDELTASPIN"

# keys this writer owns, in the order they are appended to an INCAR
DELTASPIN_KEYS = (
    LDELTASPIN,
    DELTASPIN_ATOMS,
    DELTASPIN_COMPONENTS,
    MAGMOM,
    M_DELTASPIN,
)


def incar_path(file_name):
    """Return the INCAR that belongs to a POSCAR/CONTCAR path."""
    directory = os.path.dirname(file_name)
    return os.path.join(directory, "INCAR")


def read_incar(file_name):
    """Read the INCAR next to a POSCAR/CONTCAR as a key/value mapping."""
    values = {}
    path = incar_path(file_name)
    if not os.path.isfile(path):
        warnings.warn(f"no INCAR next to {file_name}; the moments are not read")
        return values
    with open_file(path) as fp:
        text = fp.read()
    for line in text.splitlines():
        stripped = line.split("!")[0].strip()
        if not stripped or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip().upper()] = value.strip()
    return values


def moments_from_incar(values, natoms):
    """Return the moments of a DeltaSpin INCAR as an ``(natoms, 3)`` array.

    ``M_DELTASPIN`` holds the constrained targets and is preferred over the
    initial guess in ``MAGMOM``. A collinear list of one value per atom is
    interpreted as a moment along z.
    """
    for key in (M_DELTASPIN, MAGMOM):
        if key not in values:
            continue
        try:
            numbers = [float(value) for value in values[key].split()]
        except ValueError:
            warnings.warn(f"cannot read the numbers of {key} in INCAR")
            continue
        if len(numbers) == 3 * natoms:
            return np.array(numbers).reshape(natoms, 3)
        if len(numbers) == natoms:
            moments = np.zeros([natoms, 3])
            moments[:, 2] = numbers
            return moments
        warnings.warn(
            f"{key} lists {len(numbers)} values, which matches neither "
            f"{natoms} nor {3 * natoms} atoms"
        )
    return None


def format_values(values):
    """Format one flat list of numbers as an INCAR value."""
    return " ".join(f"{value:.10f}" for value in values)


def update_incar(file_name, updates):
    """Replace the given keys of the INCAR next to a POSCAR/CONTCAR.

    Keys that are absent from the INCAR are appended, and the continuation
    lines of a replaced multi-line value are removed.

    Parameters
    ----------
    file_name : str
        POSCAR/CONTCAR path; its INCAR is updated.
    updates : dict
        Mapping of INCAR key to the value that is written for it.
    """
    path = incar_path(file_name)
    lines = []
    if os.path.isfile(path):
        with open_file(path) as fp:
            lines = fp.read().splitlines()

    kept = []
    replaced = set()
    for index, line in enumerate(lines):
        stripped = line.split("!")[0].strip()
        key = stripped.partition("=")[0].strip().upper() if "=" in stripped else None
        if key in updates:
            kept.append(f"{key} = {updates[key]}")
            replaced.add(key)
            # drop the continuation lines of a multi-line value
            while index + 1 < len(lines):
                following = lines[index + 1]
                if (
                    not following.strip()
                    or "=" in following
                    or following.lstrip().startswith("!")
                ):
                    break
                index += 1
            continue
        kept.append(line)

    for key in DELTASPIN_KEYS:
        if key in updates and key not in replaced:
            kept.append(f"{key} = {updates[key]}")

    with open_file(path, "w") as fp:
        fp.write("\n".join(kept).rstrip("\n") + "\n")

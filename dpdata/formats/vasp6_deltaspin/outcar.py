"""Read the magnetic observables of a VASP 6 DeltaSpin OUTCAR.

The DeltaSpin build of VASP 6 constrains the local moments of selected atoms
and prints, once per ionic step,

* ``DeltaSpin final status`` with the SCF and moment constraint convergence,
* ``DeltaSpin final constrained observables`` with the moment of every
  constrained atom, and
* ``Magnetic force (eV/uB)`` with the magnetic force of every atom.

The tables list the moments of constrained atoms only, so atoms that take part
in the calculation without a moment constraint have no moment label here. The
constrained components are given per atom by the ``cx cy cz`` mask columns of
the two tables.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from dpdata.formats.vasp.outcar import get_frames as _get_vasp_frames

_MOMENT_HEADER = ("ion", "elem", "cx", "cy", "cz", "Mx", "My", "Mz")
_FORCE_HEADER = ("ion", "elem", "cx", "cy", "cz", "MFx", "MFy", "MFz")
_OBSERVABLES_TITLE = "DeltaSpin final constrained observables"
_FORCE_TITLE = "Magnetic force (eV/uB)"
_STATUS_TITLE = "DeltaSpin final status"

_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?"
_MOMENT_DEF_RE = re.compile(r"DELTASPIN_MOMENT_DEF\s*=\s*(\d+)")
_CONSTRAINT_RE = re.compile(r"Moment constraint\s*:\s*(reached|not reached|unavailable)")
_SCF_RE = re.compile(r"SCF convergence\s*:\s*(reached|not reached)")
_MAX_ERROR_RE = re.compile(rf"Maximum moment error\s*:\s*({_NUMBER})")
_TOLERANCE_RE = re.compile(rf"Requested tolerance\s*:\s*({_NUMBER})")


def _to_float(text: str) -> float:
    """Convert a Fortran formatted number, tolerating ``D`` exponents."""
    return float(text.replace("D", "E").replace("d", "e"))


@dataclass
class DeltaSpinObservables:
    """Magnetic observables of one ionic step.

    Attributes
    ----------
    moment_def : int
        Value of ``DELTASPIN_MOMENT_DEF``; it selects how the moments are
        defined (0: PAW projector, 1: PAW-POU, 2: PAW onsite all-l).
    moment_constraint_reached : bool or None
        Whether the moment constraint converged; ``None`` when the OUTCAR has
        no ``DeltaSpin final status`` block.
    scf_converged : bool or None
        Whether the SCF loop converged, when the OUTCAR reports it.
    max_moment_error : float or None
        Largest constrained-component error of the step, in uB.
    tolerance : float or None
        Requested moment tolerance, in uB.
    moments : dict[int, tuple[float, float, float]]
        Moment of each constrained atom, keyed by its one-based ion index.
    masks : dict[int, tuple[int, int, int]]
        ``cx cy cz`` constraint mask of each atom listed in the moment table.
    magnetic_forces : dict[int, tuple[float, float, float]]
        Magnetic force of every atom in the magnetic force table.
    """

    moment_def: int
    moment_constraint_reached: bool | None = None
    scf_converged: bool | None = None
    max_moment_error: float | None = None
    tolerance: float | None = None
    moments: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    masks: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    magnetic_forces: dict[int, tuple[float, float, float]] = field(
        default_factory=dict
    )


def _find_header(lines, header, start=0):
    """Return the index of the table header, or ``None`` when it is absent."""
    for idx in range(start, len(lines)):
        fields = lines[idx].split()
        if fields[: len(header)] == list(header):
            return idx
    return None


def _iter_table_rows(lines, header_index):
    """Yield the split rows between the dashed rule and the next title."""
    idx = header_index + 1
    while idx < len(lines) and lines[idx].strip() and set(lines[idx].strip()) != {"-"}:
        idx += 1
    if idx >= len(lines) or set(lines[idx].strip()) != {"-"}:
        return
    idx += 1
    while idx < len(lines):
        fields = lines[idx].split()
        if len(fields) < 2 or not fields[0].isdigit():
            return
        yield fields
        idx += 1


def _parse_status(lines):
    """Read the DeltaSpin final status block, when the OUTCAR has one."""
    status = {}
    for idx, line in enumerate(lines):
        if _STATUS_TITLE not in line:
            continue
        block = lines[idx : idx + 8]
        for entry in block:
            match = _CONSTRAINT_RE.search(entry)
            if match:
                status["moment_constraint_reached"] = match.group(1) == "reached"
            match = _SCF_RE.search(entry)
            if match:
                status["scf_converged"] = match.group(1) == "reached"
            match = _MAX_ERROR_RE.search(entry)
            if match:
                status["max_moment_error"] = _to_float(match.group(1))
            match = _TOLERANCE_RE.search(entry)
            if match:
                status["tolerance"] = _to_float(match.group(1))
        break
    return status


def parse_deltaspin_block(lines):
    """Parse the DeltaSpin observables of one ionic-step block.

    Parameters
    ----------
    lines : list[str]
        Lines of one OUTCAR ionic step.

    Returns
    -------
    DeltaSpinObservables or None
        The observables, or ``None`` when the block has no DeltaSpin tables.
    """
    title_index = next(
        (idx for idx, line in enumerate(lines) if _OBSERVABLES_TITLE in line), None
    )
    if title_index is None:
        return None

    moment_def = 0
    for line in lines[title_index : title_index + 4]:
        match = _MOMENT_DEF_RE.search(line)
        if match:
            moment_def = int(match.group(1))
            break

    moment_header = _find_header(lines, _MOMENT_HEADER, title_index)
    if moment_header is None:
        raise ValueError(
            "the DeltaSpin observables of this OUTCAR do not use the "
            "'ion elem cx cy cz ...' table header"
        )

    moments = {}
    masks = {}
    for fields in _iter_table_rows(lines, moment_header):
        ion = int(fields[0])
        masks[ion] = tuple(int(value) for value in fields[2:5])
        moments[ion] = tuple(_to_float(value) for value in fields[5:8])

    force_title = next(
        (idx for idx, line in enumerate(lines) if _FORCE_TITLE in line), None
    )
    if force_title is None:
        raise ValueError(
            f"the OUTCAR has no '{_FORCE_TITLE}' table for the DeltaSpin "
            "observables it reports"
        )
    force_header = _find_header(lines, _FORCE_HEADER, force_title)
    if force_header is None:
        raise ValueError(
            "the DeltaSpin magnetic force table does not use the "
            "'ion elem cx cy cz ...' table header"
        )
    magnetic_forces = {}
    for fields in _iter_table_rows(lines, force_header):
        ion = int(fields[0])
        magnetic_forces[ion] = tuple(_to_float(value) for value in fields[5:8])

    return DeltaSpinObservables(
        moment_def=moment_def,
        moments=moments,
        masks=masks,
        magnetic_forces=magnetic_forces,
        **_parse_status(lines),
    )


def _check_moments(observables, ntot):
    """Reject structurally inconsistent tables before they reach a data set."""
    for ion in list(observables.moments) + list(observables.magnetic_forces):
        if not 1 <= ion <= ntot:
            raise ValueError(
                f"DeltaSpin observables list ion {ion}, but the OUTCAR has "
                f"{ntot} atoms"
            )
    for ion, moment in observables.moments.items():
        if not any(observables.masks[ion]):
            warnings.warn(
                f"DeltaSpin moment table lists unconstrained atom {ion}; its "
                "moment is not part of the constraint"
            )
        if not np.isfinite(moment).all():
            raise ValueError(f"DeltaSpin moment of ion {ion} is not finite")


def get_frames(fname, begin=0, step=1, convergence_check=True):
    """Read the labeled ionic steps of a DeltaSpin OUTCAR.

    The cells, coordinates, energies, forces and virials come from the regular
    VASP OUTCAR reader, so both readers accept exactly the same ionic steps.

    Parameters
    ----------
    fname : str
        VASP ``OUTCAR`` file.
    begin : int, default=0
        Index of the first ionic step to load.
    step : int, default=1
        Load every ``step``-th ionic step.
    convergence_check : bool, default=True
        Drop the ionic steps whose moment constraint did not converge. When
        disabled they are kept and their moments and magnetic forces are zero.

    Returns
    -------
    tuple
        ``(atom_names, atom_numbs, atom_types, cells, coords, energies,
        forces, virials, spins, force_mags)``.
    """
    steps = []

    def _observe(step_index, block):
        steps.append(parse_deltaspin_block(block))

    (
        atom_names,
        atom_numbs,
        atom_types,
        cells,
        coords,
        energies,
        forces,
        virials,
    ) = _get_vasp_frames(
        fname,
        begin=begin,
        step=step,
        convergence_check=convergence_check,
        block_observer=_observe,
    )

    ntot = sum(atom_numbs)
    if len(steps) != len(cells):
        raise ValueError(
            "internal error: the DeltaSpin and the VASP reader disagree on the "
            f"number of ionic steps ({len(steps)} != {len(cells)})"
        )

    keep = []
    unconstrained = []
    for index, observables in enumerate(steps):
        if observables is None or observables.moment_constraint_reached is False:
            if convergence_check:
                continue
        if observables is not None:
            _check_moments(observables, ntot)
            unconstrained.append(
                sorted(set(range(1, ntot + 1)) - set(observables.moments))
            )
        keep.append(index)

    if convergence_check:
        dropped = len(steps) - len(keep)
        if dropped:
            warnings.warn(
                f"{dropped} of {len(steps)} ionic steps have no converged "
                "DeltaSpin moment constraint and are not collected."
            )
    if len(keep) == 0:
        raise ValueError(
            "no ionic step of this OUTCAR carries a converged DeltaSpin "
            "moment constraint; check DELTASPIN_TOL and the electronic SCF"
        )

    if unconstrained and any(unconstrained):
        atoms = sorted({atom for step in unconstrained for atom in step})
        warnings.warn(
            f"atoms {atoms} are not moment constrained, so their spins are "
            "not reported by DeltaSpin and are stored as zero"
        )

    all_spins = np.zeros([len(keep), ntot, 3])
    all_force_mags = np.zeros([len(keep), ntot, 3])
    for frame, index in enumerate(keep):
        observables = steps[index]
        if observables is None:
            continue
        for ion, moment in observables.moments.items():
            all_spins[frame, ion - 1] = moment
        for ion, force in observables.magnetic_forces.items():
            all_force_mags[frame, ion - 1] = force

    return (
        atom_names,
        atom_numbs,
        atom_types,
        cells[keep],
        coords[keep],
        energies[keep],
        forces[keep],
        None if virials is None else virials[keep],
        all_spins,
        all_force_mags,
    )

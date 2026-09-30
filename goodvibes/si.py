"""A per-structure Supporting Information table.

``si_rows`` gives, for each ``ThermoResult``, what an SI table reports:
the electronic energy (and the single point when one was applied), ZPE,
H, T·S, G and qh-G, the number of imaginary modes and their values, the
lowest real frequencies, the vibrational scale factors and their source,
the point group and symmetry number and their source, the level of theory
and the temperature. ``write_si`` writes the table as CSV, Markdown or
LaTeX, with an appendix of Cartesian coordinates (in the same Markdown or
LaTeX file, or next to a CSV as ``<name>_coordinates.xyz``).
"""
from __future__ import annotations

import csv
import io
import os
from typing import Any, Dict, List, Optional, Sequence

from .constants import hartree_factor, canonical_units

__all__ = ["si_rows", "si_xyz", "write_si", "SI_COLUMNS"]

#: Columns of ``si_rows``, in order; energies in the chosen units.
SI_COLUMNS = (
    "structure", "level_of_theory", "temperature", "E", "E_sp", "ZPE", "H", "TS", "qh_TS", "G", "qh_G",
    "n_imag", "imaginary", "lowest", "freq_scale_factor", "zpe_scale_factor", "scale_factor_source",
    "point_group", "symmetry_number", "symmetry_source",
)
_HEADERS = {
    "structure": "Structure", "level_of_theory": "Level of theory", "temperature": "T (K)",
    "E": "E", "E_sp": "E (SP)", "ZPE": "ZPE", "H": "H", "TS": "T·S", "qh_TS": "T·qh-S", "G": "G",
    "qh_G": "qh-G", "n_imag": "n_imag", "imaginary": "Imaginary (cm⁻¹)", "lowest": "Lowest (cm⁻¹)",
    "freq_scale_factor": "Scale factor", "zpe_scale_factor": "ZPE scale factor",
    "scale_factor_source": "Scale factor source", "point_group": "Point group",
    "symmetry_number": "σ", "symmetry_source": "σ source",
}
_ENERGIES = ("E", "E_sp", "ZPE", "H", "TS", "qh_TS", "G", "qh_G")


def _freqs(values, n=None) -> str:
    vals = sorted(float(v) for v in (values or []))
    if n is not None:
        vals = vals[:n]
    return ", ".join(f"{v:.1f}" for v in vals)


def si_rows(results: Sequence[Any], *, units: str = "hartree", n_lowest: int = 3) -> List[Dict[str, Any]]:
    """One row per result (``ThermoResult`` from ``compute_thermo`` /
    ``compute_batch``); columns ``SI_COLUMNS``, energies in ``units``
    (absolute, Hartree by default), T·S and T·qh-S at the result's
    temperature, ``imaginary`` and ``lowest`` (the ``n_lowest`` lowest real
    modes) as comma-separated cm⁻¹. A structure without frequencies has
    None for the thermal quantities."""
    f = hartree_factor(units)
    rows = []
    for r in results:
        T = r.temperature
        e_sp = r.sp_energy if getattr(r, "spc_applied", False) else None
        row = {
            "structure": r.name, "level_of_theory": r.level_of_theory or None, "temperature": T,
            "E": r.scf_energy, "E_sp": e_sp, "ZPE": r.zpe, "H": r.enthalpy,
            "TS": T * r.entropy if (T is not None and r.entropy is not None) else None,
            "qh_TS": T * r.qh_entropy if (T is not None and r.qh_entropy is not None) else None,
            "G": r.gibbs_free_energy, "qh_G": r.qh_gibbs_free_energy,
            "n_imag": r.n_imag, "imaginary": _freqs(r.im_frequency_wn) or None,
            "lowest": _freqs(r.frequency_wn, n_lowest) or None,
            "freq_scale_factor": r.freq_scale_factor, "zpe_scale_factor": r.zpe_scale_factor,
            "scale_factor_source": r.scale_factor_source, "point_group": r.point_group or None,
            "symmetry_number": r.symmno, "symmetry_source": r.symmetry_source,
        }
        for key in _ENERGIES:
            if row[key] is not None:
                row[key] = row[key] * f
        rows.append(row)
    return rows


def si_xyz(results: Sequence[Any], *, units: str = "hartree") -> str:
    """The structures as a multi-frame .xyz: the atom count, a comment line
    with the name, level of theory, E and qh-G (in ``units``), then one line
    per atom. Structures without coordinates are skipped."""
    f = hartree_factor(units)
    unit = canonical_units(units)
    out = []
    for r in results:
        qc = r.qcdata
        atoms = list(getattr(qc, "atom_types", None) or [])
        coords = list(getattr(qc, "cartesians", None) or [])
        if not atoms or len(atoms) != len(coords):
            continue
        parts = [r.name]
        if r.level_of_theory:
            parts.append(r.level_of_theory)
        if r.scf_energy is not None:
            parts.append(f"E = {r.scf_energy * f:.6f} {unit}")
        if r.qh_gibbs_free_energy is not None:
            parts.append(f"qh-G = {r.qh_gibbs_free_energy * f:.6f} {unit}")
        out.append(str(len(atoms)))
        out.append("  ".join(parts))
        out.extend(f"{a:<2} {x:14.8f} {y:14.8f} {z:14.8f}" for a, (x, y, z) in zip(atoms, coords))
    return "\n".join(out) + ("\n" if out else "")


def _fmt(v, decimals: Optional[int]) -> str:
    """An energy with ``decimals`` places; another number in its short form."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}" if decimals is not None else f"{v:g}"
    return str(v)


def _columns(rows: List[dict]) -> List[str]:
    """SI_COLUMNS without those empty in every row (no single points, no
    imaginary modes, ...)."""
    return [c for c in SI_COLUMNS if any(r.get(c) not in (None, "") for r in rows)]


def _markdown(rows, decimals, units) -> str:
    cols = _columns(rows)
    head = [_HEADERS[c] + (f" ({canonical_units(units)})" if c in _ENERGIES else "") for c in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c), decimals if c in _ENERGIES else None) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("·", r"$\cdot$"), ("σ", r"$\sigma$"), ("⁻¹", r"$^{-1}$"), ("—", "--")):
        text = text.replace(a, b)
    return text


def _latex(rows, decimals, units) -> str:
    cols = _columns(rows)
    head = [_latex_escape(_HEADERS[c] + (f" ({canonical_units(units)})" if c in _ENERGIES else "")) for c in cols]
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(_latex_escape(_fmt(r.get(c), decimals if c in _ENERGIES else None)) for c in cols)
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_si(results: Sequence[Any], path, *, units: str = "hartree", decimals: int = 6,
             n_lowest: int = 3, coordinates: bool = True) -> List[str]:
    """Write the SI table for ``results`` and return the paths written.

    The format follows the extension: ``.md`` (a Markdown table, then a
    "Cartesian coordinates" section with one block per structure),
    ``.tex`` (a booktabs tabular, then the coordinates in a verbatim
    block), ``.csv`` (the table, all columns; the coordinates go to
    ``<stem>_coordinates.xyz``) or ``.xyz`` (the coordinates only).
    ``coordinates=False`` leaves them out.
    """
    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    rows = si_rows(results, units=units, n_lowest=n_lowest)
    xyz = si_xyz(results, units=units) if coordinates else ""
    written = [path]
    if ext == ".xyz":
        text = si_xyz(results, units=units)
    elif ext in (".md", ".markdown"):
        text = _markdown(rows, decimals, units)
        if xyz:
            text += "\n## Cartesian coordinates (Å)\n\n```text\n" + xyz + "```\n"
    elif ext == ".tex":
        text = _latex(rows, decimals, units)
        if xyz:
            text += "\n\\begin{verbatim}\n" + xyz + "\\end{verbatim}\n"
    elif ext in (".csv", ".tsv"):
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(SI_COLUMNS), delimiter="\t" if ext == ".tsv" else ",",
                                lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in r.items()})
        text = buf.getvalue()
        if xyz:
            side = os.path.splitext(path)[0] + "_coordinates.xyz"
            with open(side, "w", encoding="utf-8") as fh:
                fh.write(xyz)
            written.append(side)
    else:
        raise ValueError(f"write_si: unknown format {ext or '(none)'!r}; use .md, .tex, .csv, .tsv or .xyz")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return written

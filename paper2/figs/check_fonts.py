"""Fail the build if any font in the PDF would be rejected by arXiv.

    python paper/figs/check_fonts.py paper/main.pdf

arXiv requires every font to carry an embedded font programme and rejects
Type 3 fonts. matplotlib's PDF backend defaults to ``pdf.fonttype: 3``, which
emits Type 3 with no embedded programme, so a figure generated without an
explicit ``pdf.fonttype: 42`` silently poisons an otherwise clean XeLaTeX
build. This walks every font object in the file, not only the ones reachable
from page resources, because figures arrive as Form XObjects with their own
resource dictionaries.

Exit code 0 = clean, 1 = at least one font would be rejected.
"""
from __future__ import annotations

import sys
from pathlib import Path


def audit(pdf_path: Path) -> list[tuple[str, str, str]]:
    import pikepdf

    pdf = pikepdf.open(pdf_path)
    problems: list[tuple[str, str, str]] = []
    n_fonts = 0
    for obj in pdf.objects:
        if not isinstance(obj, pikepdf.Dictionary):
            continue
        if obj.get("/Type") != "/Font":
            continue
        n_fonts += 1
        subtype = str(obj.get("/Subtype"))
        name = str(obj.get("/BaseFont", "<unnamed>"))
        desc = obj.get("/FontDescriptor")
        if subtype == "/Type0" and "/DescendantFonts" in obj:
            desc = obj["/DescendantFonts"][0].get("/FontDescriptor")
        embedded = desc is not None and any(
            k in desc for k in ("/FontFile", "/FontFile2", "/FontFile3")
        )
        if subtype == "/Type3":
            problems.append((name, subtype, "Type 3: no embedded font programme"))
        elif not embedded:
            problems.append((name, subtype, "no /FontFile in the descriptor"))
    audit.n_fonts = n_fonts  # type: ignore[attr-defined]
    return problems


def offending_figures(figs_dir: Path) -> list[str]:
    """Which included figure carries a Type 3 font.

    Naming the font is not enough to act on: every figure in the manuscript
    uses the same typeface, so ``Inter-Bold [/Type3]`` does not say which file
    to regenerate. This checks each figure PDF individually.
    """
    try:
        import pikepdf
    except ImportError:
        return []
    bad = []
    for path in sorted(figs_dir.glob("*.pdf")):
        try:
            with pikepdf.open(path) as pdf:
                if any(isinstance(o, pikepdf.Dictionary)
                       and o.get("/Type") == "/Font"
                       and str(o.get("/Subtype")) == "/Type3"
                       for o in pdf.objects):
                    bad.append(path.name)
        except Exception:  # noqa: BLE001 - a bad figure must not mask the report
            continue
    return bad


def main() -> int:
    pdf_path = Path(sys.argv[1] if len(sys.argv) > 1 else "main.pdf")
    if not pdf_path.exists():
        print(f"check_fonts: {pdf_path} not found", file=sys.stderr)
        return 1
    try:
        problems = audit(pdf_path)
    except ImportError:
        print("check_fonts: pikepdf not installed, skipping (pip install pikepdf)")
        return 0
    n = getattr(audit, "n_fonts", 0)
    if problems:
        print(f"check_fonts: {len(problems)} of {n} font objects would be "
              f"rejected by arXiv:", file=sys.stderr)
        for name, subtype, why in sorted(set(problems)):
            print(f"    {name}  [{subtype}]  {why}", file=sys.stderr)
        culprits = offending_figures(pdf_path.parent / "figs")
        if culprits:
            print("\n    the offending figure(s):", file=sys.stderr)
            for f in culprits:
                print(f"      {f}", file=sys.stderr)
            print("\n    These are written by the audit run, not by the paper "
                  "build, so re-running make here will not clear them: they "
                  "carry whatever font type the run that produced them used. "
                  "molsanity.viz.style.save_vector now pins pdf.fonttype=42 at "
                  "save time, so the next sweep regenerates them clean.",
                  file=sys.stderr)
        else:
            print("    fix: set pdf.fonttype=42 wherever the figure is "
                  "generated", file=sys.stderr)
        return 1
    print(f"check_fonts: {n} font objects, all embedded, no Type 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

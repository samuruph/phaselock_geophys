"""Write the category breakdown as a formatted .xlsx.

A CSV of ninety cells is a wall of digits. The same numbers with the red-to-green scale
applied in the sheet are readable at a glance and stay sortable and filterable, which a
PNG is not -- so this exists alongside the figure rather than instead of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..metrics.geophys import STATISTICS

# Matches the figure's clipping, so a cell that reads green there reads green here.
SCALE_MIN, SCALE_MID, SCALE_MAX = 30.0, 60.0, 90.0


def write_category_workbook(
    cells: Sequence[Any], path: Path, key: str = "family"
) -> Path:
    """One sheet of ``mean ± sd (max)``, colour-scaled, plus the raw numbers.

    Two sheets on purpose. The first is for reading -- merged source blocks, one row per
    statistic, a three-colour scale about the 50% chance line. The second is the long-form
    table one row per cell, which is what you want for a pivot or a plot and what the CSV
    used to be.
    """
    from openpyxl import Workbook
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Border, Font, Side
    from openpyxl.utils import get_column_letter

    from . import palette

    statistics = list(STATISTICS)
    # Baseline first, matching the figure: the row you measure against belongs above the
    # rows being measured.
    sources = palette.ordered_sources({c.source for c in cells}, baselines_first=True)
    categories = sorted({c.category for c in cells})
    lookup = {(c.source, c.statistic, c.category): c for c in cells}

    book = Workbook()
    sheet = book.active
    sheet.title = f"by {key}"

    bold = Font(bold=True)
    centre = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thick = Side(style="medium")

    sheet.cell(row=1, column=1, value="source").font = bold
    sheet.cell(row=1, column=2, value="statistic").font = bold
    for index, category in enumerate(categories):
        head = sheet.cell(row=1, column=3 + index, value=category)
        head.font, head.alignment = bold, centre
    average_column = 3 + len(categories)
    for offset, name in enumerate(("M.Avg.", "M.Avg. sd")):
        head = sheet.cell(row=1, column=average_column + offset, value=name)
        head.font, head.alignment = bold, centre
    # The mean drives the colour scale, so the max sits in its own block to the right
    # rather than being buried in a string the scale cannot read.
    best_column = average_column + 3
    for index, category in enumerate(categories):
        head = sheet.cell(row=1, column=best_column + index,
                          value=f"{category} (best cell)")
        head.font, head.alignment = bold, centre

    row = 2
    for source in sources:
        present = [s for s in statistics
                   if any((source, s, c) in lookup for c in categories)]
        if not present:
            continue
        first = row
        for statistic in present:
            sheet.cell(row=row, column=1, value=source if row == first else None)
            sheet.cell(row=row, column=2, value=statistic)
            across = []
            for index, category in enumerate(categories):
                cell = lookup.get((source, statistic, category))
                if cell is None:
                    continue
                across.append(100 * cell.mean)
                mean = sheet.cell(row=row, column=3 + index, value=round(100 * cell.mean, 1))
                mean.alignment = centre
                mean.number_format = "0.0"
                spread = sheet.cell(row=row, column=best_column + index,
                                    value=round(100 * cell.best, 1))
                spread.alignment = centre
                spread.number_format = "0.0"
            if across:
                mean_of_means = sum(across) / len(across)
                variance = sum((v - mean_of_means) ** 2 for v in across) / len(across)
                for offset, value in enumerate((mean_of_means, variance**0.5)):
                    average = sheet.cell(row=row, column=average_column + offset,
                                         value=round(value, 1))
                    average.alignment = centre
                    average.number_format = "0.0"
            row += 1
        if row > first + 1:
            sheet.merge_cells(start_row=first, start_column=1, end_row=row - 1, end_column=1)
        sheet.cell(row=first, column=1).alignment = centre
        sheet.cell(row=first, column=1).font = bold
        for column in range(1, best_column + len(categories)):
            sheet.cell(row=row - 1, column=column).border = Border(bottom=thick)

    span = f"C2:{get_column_letter(average_column)}{row - 1}"
    sheet.conditional_formatting.add(span, ColorScaleRule(
        start_type="num", start_value=SCALE_MIN, start_color="D73027",
        mid_type="num", mid_value=SCALE_MID, mid_color="FFFFBF",
        end_type="num", end_value=SCALE_MAX, end_color="1A9850",
    ))
    sheet.freeze_panes = "C2"
    sheet.column_dimensions["A"].width = 18
    sheet.column_dimensions["B"].width = 14
    for index in range(best_column + len(categories) - 3):
        sheet.column_dimensions[get_column_letter(3 + index)].width = 13

    long = book.create_sheet("all cells")
    fields = list(cells[0].flatten())
    for index, name in enumerate(fields, start=1):
        long.cell(row=1, column=index, value=name).font = bold
    for r, cell in enumerate(cells, start=2):
        for index, name in enumerate(fields, start=1):
            long.cell(row=r, column=index, value=cell.flatten()[name])
    long.freeze_panes = "A2"

    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    return path


def write_summary_workbook(summaries: Sequence[Any], path: Path) -> Path:
    """The aggregate table as a sheet: sources down, signals across.

    The same numbers as the summary figure, but sortable and filterable, and with the
    spread and the best cell in their own columns so the colour scale can read the mean
    without a string in the way.
    """
    from openpyxl import Workbook
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Border, Font, Side
    from openpyxl.utils import get_column_letter

    from . import palette

    summaries = [s for s in summaries if s.kind in ("phi", "coupling")]
    if not summaries:
        raise ValueError("no phi or coupling summaries to write")

    sources = palette.ordered_sources({s.source for s in summaries}, baselines_first=True)
    order = tuple(STATISTICS) + ("alignment", "erosion")
    statistics = [n for n in order if any(s.statistic == n for s in summaries)]
    lookup = {(s.source, s.statistic): s for s in summaries}

    book = Workbook()
    sheet = book.active
    sheet.title = "overall"

    bold = Font(bold=True)
    centre = Alignment(horizontal="center", vertical="center", wrap_text=True)

    sheet.cell(row=1, column=1, value="source").font = bold
    blocks = (("mean", "mean"), ("s.d.", "std"), ("best cell", "best"))
    for group, (label, _) in enumerate(blocks):
        for index, name in enumerate(statistics):
            head = sheet.cell(row=1, column=2 + group * (len(statistics) + 1) + index,
                              value=f"{name} {label}")
            head.font, head.alignment = bold, centre

    for row, source in enumerate(sources, start=2):
        sheet.cell(row=row, column=1, value=source).font = bold
        for group, (_, attribute) in enumerate(blocks):
            for index, name in enumerate(statistics):
                item = lookup.get((source, name))
                if item is None:
                    continue
                cell = sheet.cell(
                    row=row, column=2 + group * (len(statistics) + 1) + index,
                    value=round(100 * getattr(item, attribute), 1),
                )
                cell.alignment, cell.number_format = centre, "0.0"
        if source in set(palette.BASELINE_SOURCES):
            for column in range(1, 2 + 3 * (len(statistics) + 1)):
                sheet.cell(row=row, column=column).border = Border(
                    bottom=Side(style="medium"))

    last = len(sources) + 1
    for group in (0, 2):  # colour the mean and the best-cell blocks, not the spread
        first = get_column_letter(2 + group * (len(statistics) + 1))
        final = get_column_letter(1 + group * (len(statistics) + 1) + len(statistics))
        sheet.conditional_formatting.add(f"{first}2:{final}{last}", ColorScaleRule(
            start_type="num", start_value=SCALE_MIN, start_color="D73027",
            mid_type="num", mid_value=SCALE_MID, mid_color="FFFFBF",
            end_type="num", end_value=SCALE_MAX, end_color="1A9850",
        ))
    sheet.freeze_panes = "B2"
    sheet.column_dimensions["A"].width = 20
    for index in range(3 * (len(statistics) + 1)):
        sheet.column_dimensions[get_column_letter(2 + index)].width = 12

    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    return path

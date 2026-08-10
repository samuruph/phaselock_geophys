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

    order = ["hidden_states", "latent", "x0_hat", "velocity", "dinov2"]
    statistics = list(STATISTICS)
    sources = [s for s in order if any(c.source == s for c in cells)]
    sources += sorted({c.source for c in cells} - set(order))
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
    # The mean drives the colour scale, so the max sits in its own block to the right
    # rather than being buried in a string the scale cannot read.
    for index, category in enumerate(categories):
        head = sheet.cell(row=1, column=3 + len(categories) + 1 + index,
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
            for index, category in enumerate(categories):
                cell = lookup.get((source, statistic, category))
                if cell is None:
                    continue
                mean = sheet.cell(row=row, column=3 + index, value=round(100 * cell.mean, 1))
                mean.alignment = centre
                mean.number_format = "0.0"
                spread = sheet.cell(row=row, column=3 + len(categories) + 1 + index,
                                    value=round(100 * cell.best, 1))
                spread.alignment = centre
                spread.number_format = "0.0"
            row += 1
        if row > first + 1:
            sheet.merge_cells(start_row=first, start_column=1, end_row=row - 1, end_column=1)
        sheet.cell(row=first, column=1).alignment = centre
        sheet.cell(row=first, column=1).font = bold
        for column in range(1, 3 + 2 * len(categories) + 1):
            sheet.cell(row=row - 1, column=column).border = Border(bottom=thick)

    span = f"C2:{get_column_letter(2 + len(categories))}{row - 1}"
    sheet.conditional_formatting.add(span, ColorScaleRule(
        start_type="num", start_value=SCALE_MIN, start_color="D73027",
        mid_type="num", mid_value=SCALE_MID, mid_color="FFFFBF",
        end_type="num", end_value=SCALE_MAX, end_color="1A9850",
    ))
    sheet.freeze_panes = "C2"
    sheet.column_dimensions["A"].width = 18
    sheet.column_dimensions["B"].width = 14
    for index in range(2 * len(categories) + 1):
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

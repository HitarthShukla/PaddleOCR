from pathlib import Path
import csv
import json
import re
import html
from html.parser import HTMLParser

from paddleocr import PPStructureV3


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

OCR_CONFIDENCE = 0.20

# How much of an OCR box must overlap a table cell
# before we consider it a match.
MIN_OVERLAP_RATIO = 0.10

# Slight tolerance for OCR boxes sitting just outside
# predicted cell boundaries.
CELL_PADDING = 3


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    # Remove HTML tags such as:
    # <div ...>, <img ...>, </div>, etc.
    text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities
    text = html.unescape(text)

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    text = text.replace("\\:", ":")
    text = text.replace("\\@", "@")

    return text.strip()

def normalize_for_compare(text):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean_text(text).lower()
    )


# ============================================================
# GEOMETRY
# ============================================================

def box_to_xyxy(box):

    if hasattr(box, "tolist"):
        box = box.tolist()

    # Polygon:
    # [[x,y], [x,y], [x,y], [x,y]]
    if (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and isinstance(box[0], (list, tuple))
    ):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]

        return (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    # [x1,y1,x2,y2]
    if len(box) == 4:
        return tuple(float(v) for v in box)

    # 8-point polygon
    if len(box) == 8:
        xs = [float(box[i]) for i in range(0, 8, 2)]
        ys = [float(box[i]) for i in range(1, 8, 2)]

        return (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    raise ValueError(f"Unsupported box: {box}")


def box_area(box):

    x1, y1, x2, y2 = box

    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0

    return (x2 - x1) * (y2 - y1)


def center_of(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )

def cluster_1d(values, k, max_iterations=50):
    """
    Simple 1D k-means clustering.
    Returns clusters sorted by their center.
    """
    if not values:
        return []

    values = [float(v) for v in values]

    if len(values) <= k:
        return [
            {
                "center": v,
                "values": [v]
            }
            for v in sorted(values)
        ]

    # Initial centers using evenly spaced values.
    sorted_values = sorted(values)

    centers = []
    for i in range(k):
        index = round(
            i * (len(sorted_values) - 1) / (k - 1)
        )
        centers.append(sorted_values[index])

    for _ in range(max_iterations):

        clusters = [[] for _ in range(k)]

        for value in values:

            nearest = min(
                range(k),
                key=lambda i: abs(value - centers[i])
            )

            clusters[nearest].append(value)

        new_centers = []

        for i, cluster in enumerate(clusters):

            if cluster:
                new_centers.append(
                    sum(cluster) / len(cluster)
                )
            else:
                new_centers.append(centers[i])

        if all(
            abs(a - b) < 0.01
            for a, b in zip(centers, new_centers)
        ):
            break

        centers = new_centers

    result = []

    for center, cluster in zip(centers, clusters):

        if cluster:
            result.append({
                "center": center,
                "values": cluster
            })

    result.sort(
        key=lambda x: x["center"]
    )

    return result


def detect_column_centers(
    cell_boxes,
    column_count,
    ocr_items=None
):
    """
    Determine visual column centers.

    Prefer table-cell geometry when available.  With
    use_wired_table_cells_trans_to_html=True, PP-StructureV3 can use
    RT-DETR wired cell detection geometry directly, which is the
    reliable source for this wired bank-statement layout.

    OCR X geometry is only a fallback when no usable cell boxes exist.
    """

    if cell_boxes:
        centers = [
            center_of(box)[0]
            for box in cell_boxes
        ]

        if len(centers) >= column_count:
            clusters = cluster_1d(
                centers,
                column_count
            )

            if len(clusters) == column_count:
                return [
                    cluster["center"]
                    for cluster in clusters
                ]

    if ocr_items:
        # Paddle's rec_boxes are text-line boxes.  Their left edge is
        # much more stable than the box center for right-aligned
        # numeric columns and long Details text.
        centers = [
            float(item["box"][0])
            for item in ocr_items
            if item.get("text")
        ]

        if len(centers) >= column_count:
            clusters = cluster_1d(
                centers,
                column_count
            )

            if len(clusters) == column_count:
                return [
                    cluster["center"]
                    for cluster in clusters
                ]

    return []

def _is_standalone_amount(text):
    """Return True only for an OCR item that is just a monetary amount."""
    text = clean_text(text)
    return bool(re.fullmatch(
        r"[€£$]?\s*\d+(?:,\d{3})*(?:\.\d{2})",
        text
    ))



def assign_items_to_columns(
    items,
    column_centers
):
    """
    Assign OCR items to the nearest visual column.
    """

    columns = [
        []
        for _ in column_centers
    ]

    for item in items:

        cx, _ = item["center"]

        column_index = min(
            range(len(column_centers)),
            key=lambda i:
                abs(cx - column_centers[i])
        )

        columns[column_index].append(item)

    return columns


def cluster_ocr_rows(
    ocr_items,
    row_tolerance=None
):
    """
    Group OCR items into visual rows using OCR box CENTER-Y only.

    For bank statements, adjacent rows are tightly packed but their
    text centers are still separated.  Using box overlap as a row
    criterion is dangerous because OCR boxes can be taller than the
    visible glyphs and overlap the next physical row.  That was the
    source of values such as two consecutive balances ending up in
    one cell.

    This function deliberately treats Y-center as the row signal and
    never lets a tall OCR box "pull" the next row into the current one.
    """

    if not ocr_items:
        return []

    items = sorted(
        ocr_items,
        key=lambda x: (
            x["center"][1],
            x["center"][0]
        )
    )

    if row_tolerance is None:
        heights = sorted(
            max(1.0, x["box"][3] - x["box"][1])
            for x in items
        )
        median_height = heights[len(heights) // 2]

        # OCR boxes on one printed line normally have nearly identical
        # center-Y values.  Adjacent bank-statement lines are farther
        # apart.  Keep this deliberately below the line pitch.
        row_tolerance = max(
            2.5,
            median_height * 0.50
        )

    rows = []
    current = [items[0]]
    current_center = items[0]["center"][1]

    for item in items[1:]:
        cy = item["center"][1]

        # Compare against the current row center only.
        # IMPORTANT: no vertical-overlap test here.
        if abs(cy - current_center) <= row_tolerance:
            current.append(item)

            # Robust center update: median is less likely than an average
            # to drift when one OCR box is slightly mislocalized.
            ys = sorted(x["center"][1] for x in current)
            current_center = ys[len(ys) // 2]
        else:
            current.sort(key=lambda x: x["center"][0])
            rows.append(current)
            current = [item]
            current_center = cy

    current.sort(key=lambda x: x["center"][0])
    rows.append(current)

    return rows


def looks_like_bank_statement(html_grid, ocr_items=None):
    """Robust bank-transaction detector; HTML header need not be first row."""
    if not html_grid:
        return False

    if _is_bank_statement_summary(html_grid):
        return False

    text = " ".join(
        clean_text(cell.get("text", ""))
        for row in html_grid
        for cell in row
    ).lower()
    normalized = normalize_for_compare(text)

    html_ok = (
        "date" in normalized
        and ("details" in normalized or "description" in normalized)
        and "balance" in normalized
        and ("withdrawn" in normalized or "moneyout" in normalized)
        and ("paidin" in normalized or "moneyin" in normalized)
    )
    if html_ok:
        return True

    if ocr_items:
        ocr_text = " ".join(clean_text(x.get("text", "")) for x in ocr_items).lower()
        on = normalize_for_compare(ocr_text)
        return (
            "date" in on
            and "balance" in on
            and ("withdrawn" in on or "moneyout" in on)
            and ("paidin" in on or "moneyin" in on)
        )
    return False


def _normalize_header_token(text):
    return normalize_for_compare(text)


def _is_bank_date(text):
    """
    Permanent TSB transaction dates in these PDFs are OCR'd as e.g. 03JUN25.
    Keep this deliberately narrow so detail text containing dates is not
    mistaken for the Date column.
    """
    text = clean_text(text)
    return bool(re.fullmatch(r"\d{2}[A-Z]{3}\d{2}", text))


def _is_bank_statement_summary(html_grid):
    """
    The Revolut 'Balance summary' table is structurally different from the
    transaction table.  Its HTML structure is already the useful source of
    truth, including the two-line 'Closing balance' header.
    """
    if not html_grid:
        return False

    header = " ".join(
        clean_text(cell.get("text", ""))
        for cell in html_grid[0]
    ).lower()

    normalized = normalize_for_compare(header)

    return (
        "product" in normalized
        and "openingbalance" in normalized
        and "moneyout" in normalized
        and "moneyin" in normalized
        and "closingbalance" in normalized
    )


def _find_bank_header_row(ocr_items):
    """Locate the bank header row anywhere in the OCR result."""
    rows = cluster_ocr_rows(ocr_items)
    if not rows:
        return None, None, rows

    best_idx = None
    best_score = -1
    for idx, row in enumerate(rows):
        toks = [_normalize_header_token(x.get("text", "")) for x in row]
        score = 0
        score += 1 if "date" in toks else 0
        score += 1 if ("details" in toks or "description" in toks) else 0
        score += 1 if ("withdrawn" in toks or "moneyout" in toks) else 0
        score += 1 if ("balance" in toks) else 0
        score += 1 if ("paidin" in toks or ("paid" in toks and "in" in toks) or "moneyin" in toks) else 0
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is not None and best_score >= 3:
        return best_idx, rows[best_idx], rows
    return None, None, rows


def _numeric_anchor_candidates(ocr_items):
    return [
        (float(item["box"][2]), item)
        for item in ocr_items
        if _is_standalone_amount(item.get("text", ""))
    ]


def _cluster_amount_columns(amount_x):
    """Return 2 or 3 robust X clusters for standalone monetary values."""
    if len(amount_x) < 2:
        return []

    xs = sorted(float(x) for x in amount_x)
    c2 = cluster_1d(xs, 2) if len(xs) >= 4 else []
    c3 = cluster_1d(xs, 3) if len(xs) >= 9 else []

    def quality(clusters):
        if len(clusters) < 2:
            return -1.0
        centers = [c["center"] for c in clusters]
        gaps = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
        spreads = []
        for c in clusters:
            vals = c["values"]
            mean = c["center"]
            if vals:
                spreads.append(sum((v-mean)**2 for v in vals) / len(vals))
        spread = (sum(spreads) / len(spreads)) ** 0.5 if spreads else 0.0
        return min(gaps) / max(1.0, spread)

    # Prefer three clusters only when the middle cluster is genuinely
    # separated. Otherwise two-cluster geometry is the safer interpretation.
    q2 = quality(c2)
    q3 = quality(c3) if len(c3) == 3 else -1.0
    if len(c3) == 3 and q3 > max(6.0, q2 * 1.15):
        return [c["center"] for c in c3]
    if len(c2) == 2:
        return [c["center"] for c in c2]
    if len(c3) == 3:
        return [c["center"] for c in c3]
    return []


def _infer_bank_numeric_anchors(ocr_items):
    """Infer numeric-column anchors without trusting malformed cell geometry.

    When the OCR header is present, its X centers are the primary source of
    truth.  This preserves the behavior that was already good on most pages.
    Only when the header cannot be found do we infer numeric columns from the
    right edges of repeated standalone money values.
    """
    header_idx, header_items, rows = _find_bank_header_row(ocr_items)

    found = {}
    if header_items:
        for item in header_items:
            norm = _normalize_header_token(item.get("text", ""))
            x = float(item["center"][0])
            if norm in ("withdrawn", "moneyout"):
                found["withdrawn"] = x
            elif norm in ("paidin", "moneyin"):
                found["paidin"] = x
            elif norm == "balance":
                found["balance"] = x
            elif norm == "date":
                found["date"] = x
            elif norm in ("details", "description"):
                found["details"] = x

        if "paidin" not in found:
            parts = [
                x for x in header_items
                if _normalize_header_token(x.get("text", "")) in ("paid", "in")
            ]
            if len(parts) >= 2:
                found["paidin"] = sum(float(x["center"][0]) for x in parts[:2]) / 2.0

        if all(k in found for k in ("withdrawn", "paidin", "balance")):
            numeric = [found["withdrawn"], found["paidin"], found["balance"]]
            if all(a < b for a, b in zip(numeric, numeric[1:])):
                return [
                    found.get("date", 0.0),
                    found.get("details", 0.0),
                    *numeric,
                ], rows, header_idx

    # Header missing: infer only the numeric columns from standalone money
    # boxes. Their right edge is stable despite varying digit counts.
    amount_x = [x for x, _ in _numeric_anchor_candidates(ocr_items)]
    observed = _cluster_amount_columns(amount_x)
    if len(observed) == 3:
        return [found.get("date", 0.0), found.get("details", 0.0), *observed], rows, header_idx

    if len(observed) == 2:
        if all(k in found for k in ("paidin", "balance")):
            numeric = [observed[0], found["paidin"], found["balance"]]
            if not (numeric[0] < numeric[1] < numeric[2]):
                numeric = [observed[0], (observed[0] + observed[1]) / 2.0, observed[1]]
        elif all(k in found for k in ("withdrawn", "balance")):
            numeric = [found["withdrawn"], (observed[0] + observed[1]) / 2.0, observed[1]]
            if not (numeric[0] < numeric[1] < numeric[2]):
                numeric = [observed[0], (observed[0] + observed[1]) / 2.0, observed[1]]
        else:
            numeric = [observed[0], (observed[0] + observed[1]) / 2.0, observed[1]]
        return [found.get("date", 0.0), found.get("details", 0.0), *numeric], rows, header_idx

    return None, rows, header_idx


def _extract_embedded_bank_date(text):
    """Extract a transaction date embedded in an OCR text line.

    Supports both Permanent TSB's compact form (03JUN25) and the
    human-readable form used by Revolut-style statements (9 May 2025).
    The date is only extracted when it is the only date-like token in the
    OCR line; the remainder stays in Details/Description.
    """
    text = clean_text(text)

    patterns = (
        r"(?<!\d)(\d{2}[A-Z]{3}\d{2})(?!\d)",
        r"(?<!\d)(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})(?!\d)",
        r"(?<!\d)(\d{1,2}[/-](?:\d{1,2})[/-]\d{2,4})(?!\d)",
    )

    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if len(matches) != 1:
            continue
        m = matches[0]
        date_text = clean_text(m.group(1))
        remainder = clean_text(text[:m.start()] + " " + text[m.end():])
        return date_text, remainder

    return None, text


def _assign_bank_row_semantically(row_items, anchors):
    """Initial geometric assignment of one bank row.

    The numeric anchors describe the three physical numeric positions from
    left to right.  A later orientation-consistency pass may reverse the
    numeric columns (and/or the row order) when the OCR coordinate system
    for a page is mirrored.
    """
    columns = [[] for _ in range(5)]
    numeric_anchors = anchors[2:5]
    for item in row_items:
        text = clean_text(item.get("text", ""))
        if not text:
            continue
        if _is_bank_date(text):
            columns[0].append(item)
            continue

        embedded_date, remainder = _extract_embedded_bank_date(text)
        if embedded_date:
            date_item = dict(item)
            date_item["text"] = embedded_date
            columns[0].append(date_item)
            if not remainder:
                continue
            item = dict(item)
            item["text"] = remainder
            text = remainder

        if _is_standalone_amount(text):
            cx = float(item["box"][2])
            target = min(range(3), key=lambda i: abs(cx - numeric_anchors[i]))
            columns[2 + target].append(item)
        else:
            columns[1].append(item)
    for col in columns:
        col.sort(key=lambda item: (item["center"][1], item["center"][0]))
    return columns


def _clone_bank_columns(columns):
    return [list(col) for col in columns]


def _flip_bank_numeric_columns(rows):
    """Mirror only Withdrawn/Paid In/Balance, preserving Date/Details."""
    flipped = []
    for columns in rows:
        c = _clone_bank_columns(columns)
        c[2], c[4] = c[4], c[2]
        flipped.append(c)
    return flipped


def _bank_amount_value(items):
    """Return a single numeric amount from a column, else None."""
    values = [
        clean_text(item.get("text", ""))
        for item in items
        if _is_standalone_amount(item.get("text", ""))
    ]
    if len(values) != 1:
        return None
    try:
        return float(values[0].replace(",", ""))
    except ValueError:
        return None


def _bank_date_key(text):
    """Parse common bank statement date formats to a comparable tuple."""
    text = clean_text(text)
    if _is_bank_date(text):
        try:
            return (int(text[5:7]) + 2000,
                    ("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split().index(text[2:5]) + 1),
                    int(text[:2]))
        except (ValueError, IndexError):
            return None

    m = re.fullmatch(
        r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        return (int(m.group(3)), months[m.group(2)[:3].lower()], int(m.group(1)))

    return None


def _bank_orientation_score(rows):
    """Score chronological balance continuity for a candidate orientation."""
    if len(rows) < 2:
        return 0.0, 0

    exact = 0
    comparisons = 0
    errors = []

    for previous, current in zip(rows, rows[1:]):
        prev_balance = _bank_amount_value(previous[4])
        curr_balance = _bank_amount_value(current[4])
        if prev_balance is None or curr_balance is None:
            continue

        withdrawn = _bank_amount_value(current[2]) or 0.0
        paid_in = _bank_amount_value(current[3]) or 0.0
        expected = prev_balance - withdrawn + paid_in
        error = abs(expected - curr_balance)
        comparisons += 1

        # Bank statements are cents-precise. Allow a small OCR/rounding
        # tolerance, but reward exact balance continuity very strongly.
        if error <= 0.051:
            exact += 1
            errors.append(0.0)
        else:
            errors.append(error)

    if comparisons == 0:
        return 0.0, 0

    mean_error = sum(errors) / len(errors)
    score = exact * 100.0 - mean_error
    return score, comparisons


def _orient_bank_rows(rows):
    """Correct rare per-page mirror/flip errors using statement arithmetic.

    Paddle's OCR coordinates can occasionally be transformed for a page even
    though the rendered PDF is visually normal.  Rather than hardcoding a
    page number, test the four plausible orientations:
      1. normal rows + normal numeric order
      2. normal rows + mirrored numeric order
      3. reversed rows + normal numeric order
      4. reversed rows + mirrored numeric order

    Only apply an alternative when it has several balance-to-transaction
    comparisons and materially beats the unmodified orientation.
    """
    candidates = []
    base = rows

    variants = (
        ("normal", base),
        ("numeric_flip", _flip_bank_numeric_columns(base)),
        ("row_flip", list(reversed(base))),
        ("both_flip", list(reversed(_flip_bank_numeric_columns(base)))),
    )

    for name, candidate in variants:
        score, comparisons = _bank_orientation_score(candidate)
        candidates.append((score, comparisons, name, candidate))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_comparisons, best_name, best_rows = candidates[0]
    normal_score, normal_comparisons, _, _ = candidates[0] if candidates[0][2] == "normal" else next(
        item for item in candidates if item[2] == "normal"
    )

    if best_name != "normal" and best_comparisons >= 3:
        # Require a meaningful improvement, not a single lucky balance match.
        improvement = best_score - normal_score
        if improvement >= 150.0 or (best_score >= 250.0 and improvement >= 50.0):
            print(
                f"Bank OCR orientation correction: {best_name} "
                f"({best_comparisons} balance checks)"
            )
            return best_rows

    return rows


def _bank_header_from_html(html_grid):
    text = " ".join(clean_text(cell.get("text", "")) for row in (html_grid or []) for cell in row).lower()
    if "description" in text or "money out" in text or "money in" in text:
        return ["Date", "Description", "Money out", "Money in", "Balance"]
    return ["Date", "Details", "Withdrawn", "Paid In", "Balance"]


def build_bank_statement_from_ocr(cell_boxes, ocr_items, column_count=5, html_grid=None):
    if not ocr_items or column_count != 5:
        return [], 0, len(ocr_items)

    anchors, visual_rows, header_idx = _infer_bank_numeric_anchors(ocr_items)
    if anchors is None:
        return [], 0, len(ocr_items)

    print("Bank OCR numeric anchors: " + ", ".join(f"{x:.1f}" for x in anchors[2:5]))

    assigned_rows = []
    for idx, row_items in enumerate(visual_rows):
        # Never blindly drop row 0: header can be at the bottom or missing.
        if header_idx is not None and idx == header_idx:
            continue
        columns = _assign_bank_row_semantically(row_items, anchors)
        assigned_rows.append(columns)

    assigned_rows = _orient_bank_rows(assigned_rows)

    grid = [_bank_header_from_html(html_grid)]
    for columns in assigned_rows:
        row = [cell_text(col) for col in columns]
        if any(row):
            grid.append(row)

    return grid, len(ocr_items), 0


def build_grid_from_ocr(
    cell_boxes,
    ocr_items,
    column_count
):
    """
    Reconstruct a table directly from OCR geometry.

    This is used when Paddle's HTML table structure is
    clearly unreliable.
    """

    if not ocr_items:
        return [], 0, 0

    column_centers = detect_column_centers(
        cell_boxes,
        column_count,
        ocr_items
    )

    if len(column_centers) != column_count:

        print(
            "WARNING: Could not reliably detect "
            "column centers."
        )

        return [], 0, len(ocr_items)

    visual_rows = cluster_ocr_rows(
        ocr_items
    )

    grid = []

    assigned_count = 0

    for row_items in visual_rows:

        columns = assign_items_to_columns(
            row_items,
            column_centers
        )

        row = []

        for column_items in columns:

            text = cell_text(
                column_items
            )

            row.append(text)

            if column_items:
                assigned_count += len(
                    column_items
                )

        grid.append(row)

    unmatched = (
        len(ocr_items)
        - assigned_count
    )

    return (
        grid,
        assigned_count,
        unmatched
    )

def point_inside(point, box):

    x, y = point
    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


# ============================================================
# HTML TABLE PARSER
# ============================================================

class TableParser(HTMLParser):

    def __init__(self):

        super().__init__()

        self.rows = []
        self.current_row = None
        self.current_cell = None
        self.current_cell_text = []

    def handle_starttag(self, tag, attrs):

        tag = tag.lower()

        if tag == "tr":

            self.current_row = []

        elif tag in ("td", "th"):

            attrs_dict = dict(attrs)

            self.current_cell = {
                "text": "",
                "colspan": int(
                    attrs_dict.get("colspan", "1")
                ),
                "rowspan": int(
                    attrs_dict.get("rowspan", "1")
                ),
            }

            self.current_cell_text = []

    def handle_data(self, data):

        if self.current_cell is not None:

            self.current_cell_text.append(data)

    def handle_endtag(self, tag):

        tag = tag.lower()

        if tag in ("td", "th"):

            if self.current_cell is not None:

                self.current_cell["text"] = clean_text(
                    "".join(self.current_cell_text)
                )

                self.current_row.append(
                    self.current_cell
                )

            self.current_cell = None
            self.current_cell_text = []

        elif tag == "tr":

            if self.current_row is not None:

                self.rows.append(
                    self.current_row
                )

            self.current_row = None


# ============================================================
# PARSE PADDLE HTML
# ============================================================

def parse_html_structure(html):

    if not html:
        return []

    parser = TableParser()

    parser.feed(html)

    return parser.rows


# ============================================================
# EXPAND COLSPAN / ROWSPAN INTO A REAL GRID
# ============================================================

def expand_html_grid(parsed_rows):

    grid = []

    # Track cells occupied by rowspans.
    occupied = {}

    for r, source_row in enumerate(parsed_rows):

        row = []

        col = 0

        for cell in source_row:

            # Skip positions occupied by previous rowspan.
            while (r, col) in occupied:

                row.append(occupied[(r, col)])

                col += 1

            colspan = max(
                1,
                cell.get("colspan", 1)
            )

            rowspan = max(
                1,
                cell.get("rowspan", 1)
            )

            cell_obj = {
                "text": cell.get("text", ""),
                "html_row": r,
                "html_col": col,
                "colspan": colspan,
                "rowspan": rowspan,
            }

            for rr in range(
                r,
                r + rowspan
            ):

                for cc in range(
                    col,
                    col + colspan
                ):

                    occupied[(rr, cc)] = cell_obj

            row.append(cell_obj)

            # Add colspan positions.
            for extra in range(1, colspan):

                row.append(cell_obj)

            col += colspan

        # Fill remaining rowspan positions.
        while True:

            positions = [
                k
                for k in occupied
                if k[0] == r
                and k[1] >= len(row)
            ]

            if not positions:
                break

            next_col = min(
                positions,
                key=lambda x: x[1]
            )

            row.append(
                occupied[next_col]
            )

        grid.append(row)

    # Determine maximum width.
    max_cols = max(
        (len(row) for row in grid),
        default=0
    )

    for row in grid:

        while len(row) < max_cols:

            row.append({
                "text": "",
                "html_row": len(grid),
                "html_col": len(row),
                "colspan": 1,
                "rowspan": 1,
            })

    return grid


# ============================================================
# EXTRACT CELL BOXES
# ============================================================

def extract_cell_boxes(table_result):

    cell_boxes = table_result.get(
        "cell_box_list",
        None
    )

    if cell_boxes is None:
        cell_boxes = table_result.get(
            "cell_boxes",
            None
        )

    if cell_boxes is None or len(cell_boxes) == 0:
        return []

    result = []

    for box in cell_boxes:

        try:
            result.append(
                box_to_xyxy(box)
            )
        except Exception:
            continue

    return result


# ============================================================
# EXTRACT OCR
# ============================================================

def extract_table_ocr(table_result):

    ocr = table_result.get(
        "table_ocr_pred",
        {}
    )

    if not isinstance(ocr, dict):
        return []

    texts = ocr.get(
        "rec_texts",
        []
    )

    scores = ocr.get(
        "rec_scores",
        []
    )

    # --------------------------------------------------------
    # Paddle versions may return NumPy arrays here.
    # NEVER use:
    #
    #     if not boxes:
    #
    # because NumPy arrays cannot be evaluated that way.
    # --------------------------------------------------------

    boxes = ocr.get(
        "rec_boxes",
        None
    )

    if boxes is None or len(boxes) == 0:

        boxes = ocr.get(
            "rec_polys",
            None
        )

    if boxes is None or len(boxes) == 0:
        return []

    items = []

    for i, text in enumerate(texts):

        text = clean_text(text)

        if not text:
            continue

        score = (
            float(scores[i])
            if i < len(scores)
            else 1.0
        )

        if score < OCR_CONFIDENCE:
            continue

        if i >= len(boxes):
            continue

        try:

            box = box_to_xyxy(
                boxes[i]
            )

        except Exception:

            continue

        items.append({
            "text": text,
            "score": score,
            "box": box,
            "center": center_of(box),
        })

    return items

# ============================================================
# ASSIGN OCR TO PADDLE CELLS
# ============================================================

def assign_ocr_to_cells(
    cell_boxes,
    ocr_items
):

    if len(cell_boxes) == 0:

        return [], 0, len(ocr_items)

    cells = [
        []
        for _ in cell_boxes
    ]

    unmatched = []

    for item in ocr_items:

        ocr_box = item["box"]
        ocr_area = box_area(ocr_box)

        if ocr_area <= 0:
            unmatched.append(item)
            continue

        best_index = None
        best_score = 0.0

        cx, cy = item["center"]

        for i, cell in enumerate(cell_boxes):

            # Small tolerance around cell.
            x1, y1, x2, y2 = cell

            padded_cell = (
                x1 - CELL_PADDING,
                y1 - CELL_PADDING,
                x2 + CELL_PADDING,
                y2 + CELL_PADDING,
            )

            overlap = intersection_area(
                ocr_box,
                padded_cell
            )

            if overlap <= 0:
                continue

            # Main score: fraction of OCR box contained
            # inside this cell.
            overlap_ratio = (
                overlap / ocr_area
            )

            # Bonus if OCR center is inside cell.
            center_bonus = (
                0.25
                if point_inside(
                    (cx, cy),
                    padded_cell
                )
                else 0.0
            )

            score = (
                overlap_ratio
                + center_bonus
            )

            if score > best_score:

                best_score = score
                best_index = i

        if (
            best_index is None
            or best_score < MIN_OVERLAP_RATIO
        ):

            unmatched.append(item)

        else:

            cells[best_index].append(item)

    return (
        cells,
        len(ocr_items) - len(unmatched),
        len(unmatched),
    )


# ============================================================
# CLEAN DUPLICATE OCR
# ============================================================

def deduplicate_items(items):

    if not items:
        return []

    items = sorted(
        items,
        key=lambda x: (
            x["center"][1],
            x["center"][0]
        )
    )

    result = []

    for item in items:

        norm = normalize_for_compare(
            item["text"]
        )

        duplicate = False

        for previous in result:

            prev_norm = normalize_for_compare(
                previous["text"]
            )

            if norm == prev_norm:
                duplicate = True
                break

        if not duplicate:

            result.append(item)

    return result


# ============================================================
# CELL TEXT
# ============================================================

def cell_text(items):

    items = deduplicate_items(items)

    if not items:
        return ""

    # Reading order.
    items.sort(
        key=lambda x: (
            x["center"][1],
            x["center"][0]
        )
    )

    pieces = []

    for item in items:

        text = clean_text(
            item["text"]
        )

        if not text:
            continue

        if pieces:

            previous = normalize_for_compare(
                pieces[-1]
            )

            current = normalize_for_compare(
                text
            )

            if (
                current == previous
                or current in previous
                or previous in current
            ):

                # Keep the longer version.
                if len(current) > len(previous):

                    pieces[-1] = text

                continue

        pieces.append(text)

    return clean_text(
        " ".join(pieces)
    )


# ============================================================
# MAP CELL BOXES TO HTML GRID
# ============================================================

def map_boxes_to_html_grid(
    html_grid,
    cell_boxes
):

    if not html_grid:
        return []

    rows = len(html_grid)
    cols = max(len(row) for row in html_grid)

    if not cell_boxes:
        return None

    # --------------------------------------------------------
    # If counts match exactly, direct mapping is safest.
    # --------------------------------------------------------

    if len(cell_boxes) == rows * cols:

        return [
            cell_boxes[
                r * cols:
                (r + 1) * cols
            ]
            for r in range(rows)
        ]

    # --------------------------------------------------------
    # FALLBACK
    #
    # HTML may contain rowspan/colspan cells, meaning:
    #
    #     HTML grid slots != actual detected cell boxes
    #
    # We therefore map detected boxes by their actual
    # spatial positions instead of comparing pixels to
    # row/column numbers.
    # --------------------------------------------------------

    result = [
        [None for _ in range(cols)]
        for _ in range(rows)
    ]

    # --------------------------------------------------------
    # Calculate box centers.
    # --------------------------------------------------------

    box_info = []

    for box in cell_boxes:

        try:

            cx, cy = center_of(box)

            box_info.append({
                "box": box,
                "cx": cx,
                "cy": cy,
            })

        except Exception:

            continue

    if not box_info:
        return None

    # --------------------------------------------------------
    # Cluster detected boxes into visual rows.
    #
    # Boxes belonging to the same physical table row have
    # similar Y centers.
    # --------------------------------------------------------

    box_info.sort(
        key=lambda x: x["cy"]
    )

    # Estimate typical cell height.
    heights = []

    for item in box_info:

        x1, y1, x2, y2 = item["box"]

        h = max(
            1.0,
            y2 - y1
        )

        heights.append(h)

    heights.sort()

    median_height = heights[
        len(heights) // 2
    ]

    row_tolerance = max(
        8.0,
        median_height * 0.45
    )

    visual_rows = []

    for item in box_info:

        placed = False

        for visual_row in visual_rows:

            avg_y = sum(
                x["cy"]
                for x in visual_row
            ) / len(visual_row)

            if abs(
                item["cy"] - avg_y
            ) <= row_tolerance:

                visual_row.append(item)
                placed = True
                break

        if not placed:

            visual_rows.append(
                [item]
            )

    # --------------------------------------------------------
    # Sort each detected visual row left-to-right.
    # --------------------------------------------------------

    for visual_row in visual_rows:

        visual_row.sort(
            key=lambda x: x["cx"]
        )

    # --------------------------------------------------------
    # Sort visual rows top-to-bottom.
    # --------------------------------------------------------

    visual_rows.sort(
        key=lambda row:
        sum(x["cy"] for x in row) / len(row)
    )

    # --------------------------------------------------------
    # Map visual rows to HTML rows.
    #
    # The number of detected physical rows may be smaller
    # than HTML rows because of merged cells.
    # --------------------------------------------------------

    html_row_count = rows
    visual_row_count = len(visual_rows)

    if visual_row_count == 0:
        return None

    # Usually they correspond closely.
    #
    # Use proportional mapping rather than assuming an exact
    # count.
    for vr, visual_row in enumerate(visual_rows):

        if visual_row_count == 1:
            target_r = 0
        else:
            target_r = round(
                vr *
                (html_row_count - 1) /
                (visual_row_count - 1)
            )

        target_r = max(
            0,
            min(
                rows - 1,
                target_r
            )
        )

        # ----------------------------------------------------
        # Determine available HTML columns in this row.
        #
        # Do not duplicate rowspan cells here.
        # ----------------------------------------------------

        available_cols = list(
            range(cols)
        )

        # Sort by actual X position.
        for item, c in zip(
            visual_row,
            available_cols
        ):

            result[
                target_r
            ][
                c
            ] = item["box"]

    # --------------------------------------------------------
    # Some HTML rows may still contain None.
    #
    # Instead of crashing, leave them without a box.
    # build_final_grid() will fall back to Paddle's HTML text.
    # --------------------------------------------------------

    return result


# ============================================================
# NORMALIZE HTML GRID
# ============================================================

def trim_structurally_empty_columns(
    html_grid
):
    """
    Paddle can occasionally emit phantom trailing columns.

    Example:
        real table = 5 columns
        Paddle HTML = 6 columns
        6th column contains no HTML text.

    Keep the HTML structure only as a source of text/header
    metadata; OCR geometry is used for the actual placement.
    """
    if not html_grid:
        return html_grid

    max_cols = max(
        len(row)
        for row in html_grid
    )

    keep_cols = max_cols

    while keep_cols > 1:
        has_text = False

        for row in html_grid:
            if keep_cols - 1 < len(row):
                if clean_text(
                    row[keep_cols - 1].get(
                        "text",
                        ""
                    )
                ):
                    has_text = True
                    break

        if has_text:
            break

        keep_cols -= 1

    if keep_cols == max_cols:
        return html_grid

    return [
        list(row[:keep_cols])
        for row in html_grid
    ]


# ============================================================
# BUILD FINAL GRID
# ============================================================

def build_final_grid(
    html_grid,
    cell_boxes,
    ocr_items
):
    if not html_grid:
        return [], 0, len(ocr_items)

    rows = len(html_grid)
    cols = max(
        len(row)
        for row in html_grid
    )

    # ========================================================
    # SPECIAL-CASE STRUCTURED SUMMARY TABLES
    # ========================================================
    #
    # sample2.pdf table 2 is a 3-row Revolut balance summary.  OCR sees
    # "Closing" and "balance" as two visual lines, so a naive row-count
    # comparison incorrectly decides that Paddle HTML is suspicious and
    # manufactures a fifth row.
    #
    # The HTML structure is exactly the representation we want for this
    # table, including the two-line "Closing balance" header.
    if _is_bank_statement_summary(html_grid):
        grid = [
            [
                clean_text(cell.get("text", ""))
                for cell in row
            ]
            for row in html_grid
        ]
        return grid, 0, len(ocr_items)

    # ========================================================
    # DETECT WHETHER PADDLE'S HTML STRUCTURE IS SUSPICIOUS
    # ========================================================
    visual_rows = cluster_ocr_rows(
        ocr_items
    )

    visual_row_count = len(
        visual_rows
    )

    html_is_suspicious = False

    if visual_row_count > 0:
        # Paddle often gets the row structure approximately right while
        # still putting OCR geometry into the wrong cells.
        if rows > visual_row_count * 1.35:
            html_is_suspicious = True

        # Long transaction tables: one OCR visual row generally represents
        # one printed transaction line.
        if (
            rows >= 30
            and visual_row_count >= rows * 0.75
        ):
            html_is_suspicious = True

        if visual_row_count > max(1, rows) * 1.5:
            html_is_suspicious = True

    # Bank statements always use the semantic OCR reconstruction.
    if looks_like_bank_statement(html_grid, ocr_items):
        html_is_suspicious = True

    if rows >= 30 and len(ocr_items) < rows * 1.5:
        html_is_suspicious = True

    # ========================================================
    # OCR-DRIVEN FALLBACK
    # ========================================================
    if html_is_suspicious:
        print(
            "WARNING: Paddle HTML structure appears "
            "over-expanded."
        )

        print(
            f"Using OCR geometry reconstruction: "
            f"{visual_row_count} visual rows"
        )

        fallback_cols = cols

        header_text = " ".join(
            clean_text(cell.get("text", ""))
            for cell in html_grid[0]
        ).lower() if html_grid else ""

        bank_markers = (
            "date",
            "details",
            "balance",
        )

        if all(marker in header_text for marker in bank_markers):
            fallback_cols = 5

        if looks_like_bank_statement(html_grid, ocr_items):
            grid, assigned, unmatched = (
                build_bank_statement_from_ocr(
                    cell_boxes,
                    ocr_items,
                    5,
                    html_grid=html_grid,
                )
            )
        else:
            grid, assigned, unmatched = (
                build_grid_from_ocr(
                    cell_boxes,
                    ocr_items,
                    fallback_cols
                )
            )

        if grid:
            return (
                grid,
                assigned,
                unmatched
            )

    # ========================================================
    # NORMAL PADDLE HTML PATH
    # ========================================================
    if len(cell_boxes) == rows * cols:
        box_grid = [
            cell_boxes[
                r * cols:
                (r + 1) * cols
            ]
            for r in range(rows)
        ]
    else:
        box_grid = map_boxes_to_html_grid(
            html_grid,
            cell_boxes
        )

    if box_grid is None:
        grid = []
        for row in html_grid:
            grid.append([
                clean_text(
                    cell.get(
                        "text",
                        ""
                    )
                )
                for cell in row
            ])

        return (
            grid,
            0,
            len(ocr_items)
        )

    valid_boxes = []
    valid_positions = []

    for r in range(rows):
        for c in range(cols):
            box = box_grid[r][c]

            if box is not None:
                valid_boxes.append(box)
                valid_positions.append(
                    (r, c)
                )

    assigned, assigned_count, unmatched = (
        assign_ocr_to_cells(
            valid_boxes,
            ocr_items
        )
    )

    grid = [
        [
            clean_text(
                html_grid[r][c].get(
                    "text",
                    ""
                )
            )
            for c in range(cols)
        ]
        for r in range(rows)
    ]

    for i, (r, c) in enumerate(
        valid_positions
    ):
        text = cell_text(
            assigned[i]
        )

        if text:
            grid[r][c] = text

    return (
        grid,
        assigned_count,
        unmatched,
    )

# ============================================================
# HEADERS
# ============================================================

def determine_headers(grid):

    if not grid:
        return []

    # The first row is Paddle's header row.
    return [
        clean_text(x)
        for x in grid[0]
    ]


# ============================================================
# MARKDOWN
# ============================================================

def markdown_escape(text):

    text = str(text)

    return (
        text
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def to_markdown(grid):

    if not grid:
        return ""

    column_count = max(
        len(row)
        for row in grid
    )

    normalized = []

    for row in grid:

        row = list(row)

        while len(row) < column_count:
            row.append("")

        normalized.append([
            clean_text(x)
            for x in row[:column_count]
        ])

    # ========================================================
    # ESCAPE MARKDOWN
    # ========================================================

    def escape(text):

        return (
            str(text)
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\n", " ")
        )

    escaped_grid = [
        [
            escape(cell)
            for cell in row
        ]
        for row in normalized
    ]

    # ========================================================
    # CALCULATE REAL COLUMN WIDTH
    #
    # IMPORTANT:
    # Use the COMPLETE cell contents.
    # NOT the longest word.
    # ========================================================

    widths = []

    for col in range(column_count):

        width = 3

        for row in escaped_grid:

            width = max(
                width,
                len(row[col])
            )

        widths.append(width)

    # ========================================================
    # FORMAT ROW
    # ========================================================

    def format_row(row):

        cells = []

        for i, text in enumerate(row):

            cells.append(
                text.ljust(
                    widths[i]
                )
            )

        return (
            "| "
            + " | ".join(cells)
            + " |"
        )

    # ========================================================
    # BUILD MARKDOWN
    # ========================================================

    output = []

    # Header
    output.append(
        format_row(
            escaped_grid[0]
        )
    )

    # Separator
    output.append(
        "| "
        + " | ".join(
            "-" * width
            for width in widths
        )
        + " |"
    )

    # Data
    for row in escaped_grid[1:]:

        output.append(
            format_row(row)
        )

    return "\n".join(output)


# ============================================================
# CSV
# ============================================================

def save_csv(
    path,
    grid
):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerows(grid)


# ============================================================
# JSON
# ============================================================

def save_json(
    path,
    grid
):

    if not grid:
        records = []

    else:

        headers = grid[0]

        records = []

        for row in grid[1:]:

            record = {}

            for i, header in enumerate(headers):

                key = (
                    header
                    if header
                    else f"column_{i + 1}"
                )

                value = (
                    row[i]
                    if i < len(row)
                    else ""
                )

                # Prevent duplicate JSON keys.
                original_key = key
                counter = 2

                while key in record:

                    key = (
                        f"{original_key}_{counter}"
                    )

                    counter += 1

                record[key] = value

            records.append(record)

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# PROCESS TABLE
# ============================================================

def process_table(
    table_result,
    page_number,
    table_number
):

    print()
    print(
        f"TABLE {table_number}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    html = table_result.get(
        "pred_html",
        ""
    )

    parsed_html = parse_html_structure(
        html
    )

    html_grid = expand_html_grid(
        parsed_html
    )

    # Remove phantom trailing columns produced by malformed HTML.
    html_grid = trim_structurally_empty_columns(
        html_grid
    )

    html_rows = len(html_grid)

    html_cols = (
        max(
            len(row)
            for row in html_grid
        )
        if html_grid
        else 0
    )

    print(
        f"Paddle HTML structure: "
        f"{html_rows} rows × "
        f"{html_cols} columns"
    )

    # --------------------------------------------------------
    # CELL BOXES
    # --------------------------------------------------------

    cell_boxes = extract_cell_boxes(
        table_result
    )

    print(
        f"Detected cells: "
        f"{len(cell_boxes)}"
    )

    # --------------------------------------------------------
    # OCR
    #
    # IMPORTANT:
    # This MUST happen before using ocr_items.
    # --------------------------------------------------------

    ocr_items = extract_table_ocr(
        table_result
    )

    print(
        f"OCR items: "
        f"{len(ocr_items)}"
    )

    # --------------------------------------------------------
    # OCR VISUAL ROW COUNT
    # --------------------------------------------------------

    ocr_visual_rows = len(
        cluster_ocr_rows(
            ocr_items
        )
    )

    print(
        f"OCR visual rows: "
        f"{ocr_visual_rows}"
    )

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    grid, assigned, unmatched = (
        build_final_grid(
            html_grid,
            cell_boxes,
            ocr_items
        )
    )

    print(
        f"OCR assigned: "
        f"{assigned}"
    )

    print(
        f"OCR unmatched: "
        f"{unmatched}"
    )

    if not grid:

        print(
            "ERROR: Could not construct table grid."
        )

        return

    # --------------------------------------------------------
    # GUARANTEE RECTANGULAR OUTPUT
    # --------------------------------------------------------

    column_count = max(
        len(row)
        for row in grid
    )

    for row in grid:

        while len(row) < column_count:

            row.append("")

    print(
        f"Final dimensions: "
        f"{len(grid)} rows × "
        f"{column_count} columns"
    )

    print()

    # --------------------------------------------------------
    # MARKDOWN
    # --------------------------------------------------------

    markdown = to_markdown(
        grid
    )

    print(markdown)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = (
        f"page_{page_number}_"
        f"table_{table_number}"
    )

    md_path = (
        OUTPUT_DIR /
        f"{stem}.md"
    )

    csv_path = (
        OUTPUT_DIR /
        f"{stem}.csv"
    )

    json_path = (
        OUTPUT_DIR /
        f"{stem}.json"
    )

    md_path.write_text(
        markdown,
        encoding="utf-8"
    )

    save_csv(
        csv_path,
        grid
    )

    save_json(
        json_path,
        grid
    )

    print()
    print("Saved:")
    print(
        f"  Markdown: {md_path}"
    )
    print(
        f"  CSV:      {csv_path}"
    )
    print(
        f"  JSON:     {json_path}"
    )


# ============================================================
# MAIN
# ============================================================

print()
print(
    "Initializing PP-StructureV3..."
)
print("=" * 70)

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

pdfs = sorted(
    INPUT_DIR.glob("*.pdf")
)

if not pdfs:

    raise FileNotFoundError(
        "No PDF files found in input/"
    )


for pdf_path in pdfs:

    print()
    print("=" * 70)

    print(
        f"Processing: "
        f"{pdf_path}"
    )

    print("=" * 70)

    results = pipeline.predict(
        str(pdf_path)
    )

    for page_number, result in enumerate(
        results,
        start=1
    ):

        print()
        print(
            f"PAGE {page_number}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Result object → dict
        # ----------------------------------------------------

        if isinstance(result, dict):

            result_data = result

        elif hasattr(result, "json"):

            result_data = result.json

        else:

            print(
                "Unsupported result type:",
                type(result)
            )

            continue

        if "res" in result_data:

            data = result_data["res"]

        else:

            data = result_data

        # ----------------------------------------------------
        # THIS IS THE IMPORTANT PART.
        #
        # PPStructureV3 exposes tables through
        # table_res_list.
        # ----------------------------------------------------

        table_results = data.get(
            "table_res_list",
            []
        )

        print(
            f"Tables detected: "
            f"{len(table_results)}"
        )

        if not table_results:

            print(
                "No tables found on this page."
            )

            continue

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        for table_number, table_result in enumerate(
            table_results,
            start=1
        ):

            process_table(
                table_result,
                page_number,
                table_number
            )


print()
print("=" * 70)
print("DONE")
print("=" * 70)
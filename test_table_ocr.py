from pathlib import Path
import csv
import json
import re
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
    column_count
):
    """
    Determine actual visual column centers from Paddle's
    detected cell boxes.

    This ignores the number of HTML rows completely.
    """

    if not cell_boxes:
        return []

    centers = []

    for box in cell_boxes:

        cx, _ = center_of(box)

        centers.append(cx)

    clusters = cluster_1d(
        centers,
        column_count
    )

    return [
        cluster["center"]
        for cluster in clusters
    ]


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
    Group OCR items into visual text rows based on Y center.

    Unlike Paddle's HTML rows, these rows represent what is
    actually visible on the page.
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

        heights = []

        for item in items:

            x1, y1, x2, y2 = item["box"]

            heights.append(
                max(1.0, y2 - y1)
            )

        heights.sort()

        median_height = heights[
            len(heights) // 2
        ]

        # OCR text on the same visual line normally has
        # very similar Y centers.
        row_tolerance = max(
            4.0,
            median_height * 0.65
        )

    rows = []

    for item in items:

        _, cy = item["center"]

        placed = False

        for row in rows:

            average_y = sum(
                x["center"][1]
                for x in row
            ) / len(row)

            if abs(cy - average_y) <= row_tolerance:

                row.append(item)
                placed = True
                break

        if not placed:

            rows.append([item])

    # Top -> bottom.
    rows.sort(
        key=lambda row:
            sum(
                x["center"][1]
                for x in row
            ) / len(row)
    )

    # Left -> right inside each row.
    for row in rows:

        row.sort(
            key=lambda x:
                x["center"][0]
        )

    return rows

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
        column_count
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
    # DETECT WHETHER PADDLE'S HTML STRUCTURE IS SUSPICIOUS
    # ========================================================

    # Estimate how many actual visual rows exist from OCR.
    visual_rows = cluster_ocr_rows(
        ocr_items
    )

    visual_row_count = len(
        visual_rows
    )

    html_is_suspicious = False

    if visual_row_count > 0:

        # Example:
        #
        # Paddle HTML: 39 rows
        # OCR visual rows: ~21
        #
        # That's clearly bogus.
        if rows > visual_row_count * 1.35:

            html_is_suspicious = True

    # Also flag extremely large empty HTML grids.
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

        grid, assigned, unmatched = (
            build_grid_from_ocr(
                cell_boxes,
                ocr_items,
                cols
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
    enable_mkldnn=False,
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
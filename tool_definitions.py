"""
Tool definitions + sandbox executor implementations for atomic visual reasoning.

Compatible with:
- datakit Tool/Function schema (types.py)
- CoF dataset: image_zoom_in_tool signature preserved exactly
- CodeDance dataset: all 17 atomic operations covered

Tool result conventions:
  - Image-returning tools : {"image_path": "<path>"}   (CoF-compatible)
  - Text-returning tools  : {"result": "<printed output>"}
  - On error              : {"error": "<traceback>", "stdout": "<partial output>"}

Quick usage:
    from tool_definitions import call_tool
    result = call_tool("image_draw_arrow_tool",
                       {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
                       image_path="input.jpg",
                       output_path="output.jpg")
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _func(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "args_format": "Format the arguments as a JSON object.",
        },
    }


# ---------------------------------------------------------------------------
# Shared parameter fragments
# ---------------------------------------------------------------------------

_BBOX_2D = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 4,
    "maxItems": 4,
    "description": (
        "Bounding box as [x1, y1, x2, y2] in pixel coordinates, "
        "where (x1, y1) is the top-left and (x2, y2) is the bottom-right corner."
    ),
}

_COLOR_RGB = {
    "type": "array",
    "items": {"type": "integer", "minimum": 0, "maximum": 255},
    "minItems": 3,
    "maxItems": 3,
    "description": "RGB color as [R, G, B], e.g. [255, 0, 0] for red.",
}

_LINE_WIDTH = {
    "type": "integer",
    "minimum": 1,
    "description": "Line thickness in pixels.",
    "default": 2,
}

# ---------------------------------------------------------------------------
# 1. Visual – image manipulation
# ---------------------------------------------------------------------------

# 1-1. image_zoom_in_tool  (CoF-compatible — do NOT change signature)
IMAGE_ZOOM_IN_TOOL = _func(
    name="image_zoom_in_tool",
    description=(
        "Zoom in on a specific region of an image by cropping it to a bounding box "
        "(bbox_2d) and upscaling so the shorter side is at least 400 px (max 4×). "
        "Use this when an object or text region is too small to inspect at the original "
        "resolution. The optional label is for your own bookkeeping and does not affect "
        "the output image."
    ),
    properties={
        "bbox_2d": _BBOX_2D,
        "label": {
            "type": "string",
            "description": (
                "Short descriptor for the cropped region (e.g. 'legend', 'axis label'). "
                "Optional — does not alter the output."
            ),
        },
    },
    required=["bbox_2d"],
)

# 1-2. image_crop_tool  (crop only, no upscaling)
IMAGE_CROP_TOOL = _func(
    name="image_crop_tool",
    description=(
        "Crop a rectangular region from the image and return it at its original "
        "resolution without any rescaling. Use this when you need to isolate a "
        "sub-region for further analysis but do not need zoom-in upscaling."
    ),
    properties={
        "bbox_2d": _BBOX_2D,
    },
    required=["bbox_2d"],
)

# 1-3. image_draw_reference_line_tool
IMAGE_DRAW_REFERENCE_LINE_TOOL = _func(
    name="image_draw_reference_line_tool",
    description=(
        "Draw a straight reference or guide line on the image between two pixel "
        "coordinates (x1, y1) → (x2, y2). Primarily used to mark threshold or "
        "benchmark values on charts so that data points can be visually compared "
        "against a fixed level. "
        "Example: draw a horizontal line at y=303 from x=0 to x=800 in red to mark "
        "the 1 000 unit threshold."
    ),
    properties={
        "x1": {"type": "number", "description": "Start x-coordinate (pixels from left edge)."},
        "y1": {"type": "number", "description": "Start y-coordinate (pixels from top edge)."},
        "x2": {"type": "number", "description": "End x-coordinate."},
        "y2": {"type": "number", "description": "End y-coordinate."},
        "color": {
            **_COLOR_RGB,
            "default": [255, 0, 0],
            "description": "RGB color of the line. Default red [255, 0, 0].",
        },
        "width": {
            **_LINE_WIDTH,
            "default": 2,
            "description": "Thickness of the line in pixels. Default 2.",
        },
    },
    required=["x1", "y1", "x2", "y2"],
)

# 1-4. image_draw_bounding_box_tool
IMAGE_DRAW_BOUNDING_BOX_TOOL = _func(
    name="image_draw_bounding_box_tool",
    description=(
        "Draw a rectangular bounding box outline on the image to highlight or annotate "
        "a region of interest. Optionally render a short text label near the top-left "
        "corner of the box. "
        "Example: bbox_2d=[50, 80, 300, 250], color=[0, 255, 0], label='car'."
    ),
    properties={
        "bbox_2d": _BBOX_2D,
        "color": {
            **_COLOR_RGB,
            "default": [0, 255, 0],
            "description": "RGB outline color. Default green [0, 255, 0].",
        },
        "width": {
            **_LINE_WIDTH,
            "default": 2,
            "description": "Outline thickness in pixels. Default 2.",
        },
        "label": {
            "type": "string",
            "description": "Optional text label drawn near the top-left corner of the box.",
        },
    },
    required=["bbox_2d"],
)

# 1-5. image_annotate_points_tool
IMAGE_ANNOTATE_POINTS_TOOL = _func(
    name="image_annotate_points_tool",
    description=(
        "Draw numbered circle markers at a list of (x, y) pixel coordinates. "
        "Each point is rendered as three concentric circles (blue outer, green middle, "
        "red filled inner) with its 1-based index printed beside it. Use for counting "
        "and localizing objects in an image. "
        "Example: points=[[120, 340], [200, 180], [450, 90]] marks three locations "
        "labeled 1, 2, 3."
    ),
    properties={
        "points": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "[x, y] pixel coordinate of one point.",
            },
            "description": "Ordered list of [x, y] pixel coordinates to annotate.",
        },
        "radius": {
            "type": "integer",
            "minimum": 1,
            "description": "Radius of the innermost (red) circle in pixels. Default 6.",
            "default": 6,
        },
    },
    required=["points"],
)

# 1-6. image_get_pixel_color_tool
IMAGE_GET_PIXEL_COLOR_TOOL = _func(
    name="image_get_pixel_color_tool",
    description=(
        "Read and return the RGB color value of a single pixel at coordinate (x, y). "
        "Useful for identifying colors, reading chart legend swatches, or sampling a "
        "region's dominant hue before performing color-based analysis. "
        "Returns: {\"result\": \"[R, G, B]\"}."
    ),
    properties={
        "x": {"type": "integer", "description": "Pixel x-coordinate (column index, 0 = left edge)."},
        "y": {"type": "integer", "description": "Pixel y-coordinate (row index, 0 = top edge)."},
    },
    required=["x", "y"],
)

# 1-7. image_draw_text_tool
IMAGE_DRAW_TEXT_TOOL = _func(
    name="image_draw_text_tool",
    description=(
        "Render a text string directly onto the image at pixel position (x, y). "
        "The top-left corner of the text block is placed at (x, y). "
        "Use for labeling regions, adding value annotations on charts, or marking "
        "identified objects. "
        "Example: x=50, y=30, text='Peak: 1.8k', color=[255, 255, 0], font_size=18."
    ),
    properties={
        "x": {"type": "integer", "description": "Left edge x-coordinate of the text block."},
        "y": {"type": "integer", "description": "Top edge y-coordinate of the text block."},
        "text": {"type": "string", "description": "The string to render on the image."},
        "color": {
            **_COLOR_RGB,
            "default": [255, 255, 255],
            "description": "RGB text color. Default white [255, 255, 255].",
        },
        "font_size": {
            "type": "integer",
            "minimum": 6,
            "description": "Font size in points. Default 16.",
            "default": 16,
        },
    },
    required=["x", "y", "text"],
)

# 1-8. image_draw_arrow_tool
IMAGE_DRAW_ARROW_TOOL = _func(
    name="image_draw_arrow_tool",
    description=(
        "Draw an arrow from tail (x1, y1) to head (x2, y2) on the image. "
        "The arrowhead is placed at (x2, y2). Uses cv2.arrowedLine internally. "
        "Use to point to or highlight a specific location, measurement, or object "
        "in an image. "
        "Example: x1=400, y1=50, x2=400, y2=200, color=[255, 0, 0], thickness=3 "
        "draws a downward red arrow."
    ),
    properties={
        "x1": {"type": "number", "description": "Tail x-coordinate of the arrow (start point)."},
        "y1": {"type": "number", "description": "Tail y-coordinate of the arrow (start point)."},
        "x2": {"type": "number", "description": "Head x-coordinate of the arrow (arrowhead end)."},
        "y2": {"type": "number", "description": "Head y-coordinate of the arrow (arrowhead end)."},
        "color": {
            **_COLOR_RGB,
            "default": [0, 0, 255],
            "description": "RGB color of the arrow. Default blue [0, 0, 255].",
        },
        "thickness": {
            "type": "integer",
            "minimum": 1,
            "description": "Shaft thickness in pixels. Default 2.",
            "default": 2,
        },
        "tip_length": {
            "type": "number",
            "minimum": 0.05,
            "maximum": 0.9,
            "description": (
                "Arrowhead length as a fraction of the total arrow length. "
                "0.1 = small tip, 0.3 = large tip. Default 0.3."
            ),
            "default": 0.3,
        },
    },
    required=["x1", "y1", "x2", "y2"],
)

# 1-9. image_rotate_tool
IMAGE_ROTATE_TOOL = _func(
    name="image_rotate_tool",
    description=(
        "Rotate the image by a fixed angle (90, 180, or 270 degrees counter-clockwise) "
        "to correct its orientation. Use when a photo or scan is sideways or upside-down. "
        "The output canvas is resized to fit the rotated image (no cropping)."
    ),
    properties={
        "angle": {
            "type": "integer",
            "enum": [90, 180, 270],
            "description": (
                "Counter-clockwise rotation in degrees. "
                "90 = landscape-left → upright, "
                "180 = upside-down → upright, "
                "270 = landscape-right → upright."
            ),
        },
    },
    required=["angle"],
)

# ---------------------------------------------------------------------------
# 2. Math – computation
# ---------------------------------------------------------------------------

# 2-1. python_arithmetic_tool  (pure Python, no external libs)
PYTHON_ARITHMETIC_TOOL = _func(
    name="python_arithmetic_tool",
    description=(
        "Execute a pure-Python arithmetic script (no external libraries) and return "
        "the printed output. Suitable for integer/float arithmetic, set-theory counts, "
        "combinatorics, ratio and percentage calculations, or any computation that does "
        "not require numpy or math. "
        "Rules: (1) Use only Python built-ins (int, float, list, dict, sum, abs, etc.). "
        "(2) End with at least one print() call to output the answer. "
        "Example: code='total = 3*4 + 12//5\\nprint(total)'."
    ),
    properties={
        "code": {
            "type": "string",
            "description": (
                "Complete Python script using only built-in operations. "
                "Must contain at least one print() statement."
            ),
        },
    },
    required=["code"],
)

# 2-2. python_numerical_compute_tool  (numpy / math)
PYTHON_NUMERICAL_COMPUTE_TOOL = _func(
    name="python_numerical_compute_tool",
    description=(
        "Execute a Python script that uses numpy or the standard math module for "
        "numerical computation and return the printed output. Suitable for array "
        "operations, trigonometry (sin/cos/tan), logarithms, statistical aggregates "
        "(mean, std, percentile), and matrix algebra. "
        "Rules: (1) import numpy as np and/or import math as needed. "
        "(2) End with print() to output the answer. "
        "Example: code='import numpy as np\\nprint(np.mean([3, 7, 5, 12]))'."
    ),
    properties={
        "code": {
            "type": "string",
            "description": (
                "Complete Python script that may import numpy and/or math. "
                "Must contain at least one print() statement."
            ),
        },
    },
    required=["code"],
)

# 2-3. python_symbolic_math_tool  (sympy)
PYTHON_SYMBOLIC_MATH_TOOL = _func(
    name="python_symbolic_math_tool",
    description=(
        "Execute a Python script using sympy for symbolic mathematics and return the "
        "printed output. Suitable for algebraic equation solving, differentiation, "
        "integration, complex number arithmetic, and exact simplification without "
        "floating-point error. "
        "Rules: (1) import sympy or use 'from sympy import ...'. "
        "(2) Use sympy.solve(), sympy.diff(), sympy.integrate(), sympy.simplify(), etc. "
        "(3) End with print() to output the answer. "
        "Example: code='from sympy import symbols, solve\\n"
        "x = symbols(\"x\")\\nprint(solve(x**2 - 4, x))'."
    ),
    properties={
        "code": {
            "type": "string",
            "description": (
                "Complete Python script that imports and uses sympy. "
                "Must contain at least one print() statement."
            ),
        },
    },
    required=["code"],
)

# 2-4. python_geometry_compute_tool
PYTHON_GEOMETRY_COMPUTE_TOOL = _func(
    name="python_geometry_compute_tool",
    description=(
        "Execute a Python script for geometric computation and return the printed "
        "output. Suitable for distances between points, polygon areas and perimeters, "
        "angles between lines, circle properties, 3-D volumes, and coordinate geometry. "
        "May use math, numpy, or sympy as needed. "
        "Rules: import whatever is needed; end with print(). "
        "Example: compute distance from (0,0) to (3,4): "
        "code='import math\\nprint(math.sqrt(3**2 + 4**2))'."
    ),
    properties={
        "code": {
            "type": "string",
            "description": (
                "Complete Python script performing geometric calculations. "
                "Must contain at least one print() statement."
            ),
        },
    },
    required=["code"],
)

# 2-5. python_scientific_compute_tool  (scipy)
PYTHON_SCIENTIFIC_COMPUTE_TOOL = _func(
    name="python_scientific_compute_tool",
    description=(
        "Execute a Python script that uses scipy for scientific computation and return "
        "the printed output. Suitable for optimization (minimize/maximize), numerical "
        "integration, interpolation (interp1d), linear algebra (linalg), and statistical "
        "tests (ttest, chi2). "
        "Rules: (1) import scipy.* as needed, also numpy if required. "
        "(2) End with print(). "
        "Example: code='from scipy.optimize import minimize\\nimport numpy as np\\n"
        "res = minimize(lambda x: (x-3)**2, x0=0)\\nprint(res.x)'."
    ),
    properties={
        "code": {
            "type": "string",
            "description": (
                "Complete Python script that imports and uses scipy. "
                "Must contain at least one print() statement."
            ),
        },
    },
    required=["code"],
)

# 2-6. python_fraction_arithmetic_tool
PYTHON_FRACTION_ARITHMETIC_TOOL = _func(
    name="python_fraction_arithmetic_tool",
    description=(
        "Execute a Python script using the fractions.Fraction class for exact rational "
        "arithmetic and return the printed output. Avoids all floating-point rounding "
        "error. Suitable for fraction addition/subtraction/multiplication/division, "
        "mixed-number conversions, and exact ratio comparisons. "
        "Rules: (1) from fractions import Fraction. "
        "(2) Fraction objects can be constructed from strings: Fraction('3/4'). "
        "(3) End with print(). "
        "Example: code='from fractions import Fraction\\n"
        "print(Fraction(1,3) + Fraction(1,6))'  →  1/2."
    ),
    properties={
        "code": {
            "type": "string",
            "description": (
                "Complete Python script that uses fractions.Fraction. "
                "Must contain at least one print() statement."
            ),
        },
    },
    required=["code"],
)

# ---------------------------------------------------------------------------
# 3. Chart / Data
# ---------------------------------------------------------------------------

# 3-1. chart_scale_compute_tool
CHART_SCALE_COMPUTE_TOOL = _func(
    name="chart_scale_compute_tool",
    description=(
        "Convert pixel coordinates to chart axis values using linear interpolation "
        "from known calibration (anchor) points. No code required — pass structured "
        "parameters only. "
        "How to use: "
        "(1) Identify at least two pixel positions on a chart axis where you know the "
        "    true data value (e.g. read from axis tick labels). "
        "(2) Pass those as calibration_points: [[pixel, value], ...]. "
        "(3) Pass the pixel(s) you want to convert as query_pixels. "
        "Example: axis has pixel 133 = value 100 and pixel 421 = value 0; "
        "query_pixels=[280] → returns the interpolated data value at pixel 280. "
        "Works for both x-axes (horizontal) and y-axes (vertical)."
    ),
    properties={
        "calibration_points": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "[pixel_coordinate, chart_axis_value]",
            },
            "minItems": 2,
            "description": (
                "At least two [pixel, chart_value] anchor pairs that define the axis "
                "scale. Pixel values must be monotonically increasing or decreasing "
                "across the list. "
                "Example: [[133, 100], [277, 50], [421, 0]] for a y-axis where "
                "pixel 133 = 100, pixel 277 = 50, pixel 421 = 0."
            ),
        },
        "query_pixels": {
            "type": "array",
            "items": {"type": "number"},
            "description": (
                "One or more pixel coordinates to convert to chart values. "
                "Example: [305, 348] to read two data points off the axis."
            ),
        },
    },
    required=["calibration_points", "query_pixels"],
)

# 3-2. chart_visualization_tool
CHART_VISUALIZATION_TOOL = _func(
    name="chart_visualization_tool",
    description=(
        "Execute a Python script using matplotlib to create a chart or plot, and "
        "return the rendered image. Use this to re-plot data extracted from an image "
        "chart for visual verification, or to produce supplementary visualizations. "
        "Rules: "
        "(1) import matplotlib.pyplot as plt at the top. "
        "(2) Do NOT call plt.show() — the figure is captured automatically. "
        "(3) Use plt.figure(), plt.plot(), plt.bar(), plt.title(), etc. as needed. "
        "Example: code='import matplotlib.pyplot as plt\\n"
        "plt.bar([\"A\",\"B\",\"C\"], [3, 7, 2])\\nplt.title(\"Count\")'."
    ),
    properties={
        "code": {
            "type": "string",
            "description": (
                "Complete Python script using matplotlib. "
                "Must not call plt.show(). The last active figure is saved automatically."
            ),
        },
    },
    required=["code"],
)

# 3-3. data_tabulate_tool
DATA_TABULATE_TOOL = _func(
    name="data_tabulate_tool",
    description=(
        "Execute a Python script using pandas to organize, aggregate, or analyze "
        "tabular data and return the printed result. Suitable for grouping, filtering, "
        "sorting, pivot tables, and descriptive statistics on manually transcribed data. "
        "Rules: "
        "(1) import pandas as pd. "
        "(2) Construct DataFrames from Python dicts or lists — do NOT read files. "
        "(3) End with print(df) or print(result). "
        "Example: code='import pandas as pd\\n"
        "df = pd.DataFrame({\"name\":[\"A\",\"B\"],\"score\":[80,95]})\\n"
        "print(df.sort_values(\"score\", ascending=False))'."
    ),
    properties={
        "code": {
            "type": "string",
            "description": (
                "Complete Python script using pandas. "
                "Must construct data from literals (not files) and print the result."
            ),
        },
    },
    required=["code"],
)

# ---------------------------------------------------------------------------
# Exports – schema
# ---------------------------------------------------------------------------

ALL_TOOLS: list[dict] = [
    # Visual
    IMAGE_ZOOM_IN_TOOL,               # CoF-compatible
    IMAGE_CROP_TOOL,
    IMAGE_DRAW_REFERENCE_LINE_TOOL,
    IMAGE_DRAW_BOUNDING_BOX_TOOL,
    IMAGE_ANNOTATE_POINTS_TOOL,
    IMAGE_GET_PIXEL_COLOR_TOOL,
    IMAGE_DRAW_TEXT_TOOL,
    IMAGE_DRAW_ARROW_TOOL,
    IMAGE_ROTATE_TOOL,
    # Math
    PYTHON_ARITHMETIC_TOOL,
    PYTHON_NUMERICAL_COMPUTE_TOOL,
    PYTHON_SYMBOLIC_MATH_TOOL,
    PYTHON_GEOMETRY_COMPUTE_TOOL,
    PYTHON_SCIENTIFIC_COMPUTE_TOOL,
    PYTHON_FRACTION_ARITHMETIC_TOOL,
    # Chart / Data
    CHART_SCALE_COMPUTE_TOOL,
    CHART_VISUALIZATION_TOOL,
    DATA_TABULATE_TOOL,
]

TOOL_BY_NAME: dict[str, dict] = {
    t["function"]["name"]: t for t in ALL_TOOLS
}

# ---------------------------------------------------------------------------
# Sandbox Executor Implementations
# ---------------------------------------------------------------------------
# Each exec_* function accepts exactly the parameters from its schema, plus
# optional image_path / output_path for image tools.
# Heavy imports (cv2, PIL, numpy, …) are deferred inside each function so
# that importing this module in schema-only contexts has no side-effects.
# ---------------------------------------------------------------------------

import contextlib
import io
import textwrap
import traceback

_DEFAULT_INPUT  = "input_image.jpg"
_DEFAULT_OUTPUT = "output_image.jpg"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _load_image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _save_image(img, out: str = _DEFAULT_OUTPUT) -> str:
    img.save(out)
    return out


def _pil_to_cv(img) -> "np.ndarray":
    import cv2, numpy as np
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr) -> "Image.Image":
    import cv2
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _draw_arrow_pil(img, x1: int, y1: int, x2: int, y2: int,
                    color: tuple, thickness: int, tip_length: float):
    """Pure-PIL arrow: shaft via draw.line + filled polygon arrowhead."""
    import math
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)
    # Arrowhead
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return img
    tip = length * tip_length
    # Unit vectors
    ux, uy = dx / length, dy / length
    # Perpendicular
    px, py = -uy, ux
    hw = max(2, thickness * 2)
    base_x = x2 - ux * tip
    base_y = y2 - uy * tip
    poly = [
        (x2, y2),
        (int(base_x + px * hw), int(base_y + py * hw)),
        (int(base_x - px * hw), int(base_y - py * hw)),
    ]
    draw.polygon(poly, fill=color)
    return img


def _exec_code(code: str, extra_ns: dict | None = None) -> dict:
    buf = io.StringIO()
    ns: dict = {"__builtins__": __builtins__}
    if extra_ns:
        ns.update(extra_ns)
    try:
        with contextlib.redirect_stdout(buf):
            exec(textwrap.dedent(code), ns)
        return {"result": buf.getvalue().strip()}
    except Exception:
        return {"error": traceback.format_exc(), "stdout": buf.getvalue()}


def _load_font(size: int):
    from PIL import ImageFont
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Image tools ──────────────────────────────────────────────────────────────

def exec_image_zoom_in_tool(
    bbox_2d: list[float],
    label: str | None = None,
    image_path: str = _DEFAULT_INPUT,
    output_path: str = _DEFAULT_OUTPUT,
) -> dict:
    img = _load_image(image_path)
    x1, y1, x2, y2 = [int(v) for v in bbox_2d]
    x1, x2 = sorted([max(0, x1), min(img.width,  x2)])
    y1, y2 = sorted([max(0, y1), min(img.height, y2)])
    crop = img.crop((x1, y1, x2, y2))
    w, h = crop.size
    if min(w, h) == 0:
        return {"error": "bbox_2d has zero area after clamping to image bounds."}
    # Upscale so the shorter side is ≥ 400 px, capped at 4×
    scale = min(4.0, max(1.0, 400 / min(w, h)))
    from PIL import Image
    zoomed = crop.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return {"image_path": _save_image(zoomed, output_path)}


def exec_image_crop_tool(
    bbox_2d: list[float],
    image_path: str = _DEFAULT_INPUT,
    output_path: str = _DEFAULT_OUTPUT,
) -> dict:
    img = _load_image(image_path)
    x1, y1, x2, y2 = [int(v) for v in bbox_2d]
    x1, x2 = sorted([max(0, x1), min(img.width,  x2)])
    y1, y2 = sorted([max(0, y1), min(img.height, y2)])
    return {"image_path": _save_image(img.crop((x1, y1, x2, y2)), output_path)}


def exec_image_draw_reference_line_tool(
    x1: float, y1: float, x2: float, y2: float,
    color: list[int] | None = None,
    width: int = 2,
    image_path: str = _DEFAULT_INPUT,
    output_path: str = _DEFAULT_OUTPUT,
) -> dict:
    from PIL import ImageDraw
    if color is None:
        color = [255, 0, 0]
    img = _load_image(image_path)
    draw = ImageDraw.Draw(img)
    draw.line([(int(x1), int(y1)), (int(x2), int(y2))], fill=tuple(color), width=width)
    return {"image_path": _save_image(img, output_path)}


def exec_image_draw_bounding_box_tool(
    bbox_2d: list[float],
    color: list[int] | None = None,
    width: int = 2,
    label: str | None = None,
    image_path: str = _DEFAULT_INPUT,
    output_path: str = _DEFAULT_OUTPUT,
) -> dict:
    from PIL import ImageDraw
    if color is None:
        color = [0, 255, 0]
    img = _load_image(image_path)
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(v) for v in bbox_2d]
    draw.rectangle([x1, y1, x2, y2], outline=tuple(color), width=width)
    if label:
        font = _load_font(14)
        draw.text((x1 + 3, y1 + 3), label, fill=tuple(color), font=font)
    return {"image_path": _save_image(img, output_path)}


def exec_image_annotate_points_tool(
    points: list[list[float]],
    radius: int = 6,
    image_path: str = _DEFAULT_INPUT,
    output_path: str = _DEFAULT_OUTPUT,
) -> dict:
    img = _load_image(image_path)
    try:
        import cv2
        arr = _pil_to_cv(img)
        for idx, pt in enumerate(points, start=1):
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(arr, (cx, cy), radius + 6, (255,   0,   0), 2, cv2.LINE_AA)
            cv2.circle(arr, (cx, cy), radius + 3, (  0, 255,   0), 2, cv2.LINE_AA)
            cv2.circle(arr, (cx, cy), radius,     (  0,   0, 255), -1)
            tx, ty = cx + radius + 4, cy - radius - 2
            for ddx, ddy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                cv2.putText(arr, str(idx), (tx + ddx, ty + ddy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(arr, str(idx), (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        img = _cv_to_pil(arr)
    except ImportError:
        # PIL fallback: concentric ellipses + text label
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        font = _load_font(max(10, radius * 2))
        for idx, pt in enumerate(points, start=1):
            cx, cy = int(pt[0]), int(pt[1])
            draw.ellipse([cx-radius-6, cy-radius-6, cx+radius+6, cy+radius+6],
                         outline=(255, 0, 0), width=2)
            draw.ellipse([cx-radius-3, cy-radius-3, cx+radius+3, cy+radius+3],
                         outline=(0, 255, 0), width=2)
            draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius],
                         fill=(0, 0, 255))
            # black shadow then white text
            for ddx, ddy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                draw.text((cx + radius + 4 + ddx, cy - radius - 2 + ddy),
                          str(idx), fill=(0, 0, 0), font=font)
            draw.text((cx + radius + 4, cy - radius - 2), str(idx),
                      fill=(255, 255, 255), font=font)
    return {"image_path": _save_image(img, output_path)}


def exec_image_get_pixel_color_tool(
    x: int, y: int,
    image_path: str = _DEFAULT_INPUT,
) -> dict:
    img = _load_image(image_path)
    x = min(max(0, int(x)), img.width  - 1)
    y = min(max(0, int(y)), img.height - 1)
    rgb = img.getpixel((x, y))[:3]
    r, g, b = rgb
    return {"result": f"[{r}, {g}, {b}]"}


def exec_image_draw_text_tool(
    x: int, y: int, text: str,
    color: list[int] | None = None,
    font_size: int = 16,
    image_path: str = _DEFAULT_INPUT,
    output_path: str = _DEFAULT_OUTPUT,
) -> dict:
    from PIL import ImageDraw
    if color is None:
        color = [255, 255, 255]
    img = _load_image(image_path)
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    draw.text((int(x), int(y)), text, fill=tuple(color), font=font)
    return {"image_path": _save_image(img, output_path)}


def exec_image_draw_arrow_tool(
    x1: float, y1: float, x2: float, y2: float,
    color: list[int] | None = None,
    thickness: int = 2,
    tip_length: float = 0.3,
    image_path: str = _DEFAULT_INPUT,
    output_path: str = _DEFAULT_OUTPUT,
) -> dict:
    if color is None:
        color = [0, 0, 255]
    img = _load_image(image_path)
    try:
        import cv2
        arr = _pil_to_cv(img)
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.arrowedLine(
            arr,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            bgr,
            thickness=thickness,
            line_type=cv2.LINE_AA,
            tipLength=tip_length,
        )
        img = _cv_to_pil(arr)
    except ImportError:
        img = _draw_arrow_pil(img, int(x1), int(y1), int(x2), int(y2),
                              tuple(color), thickness, tip_length)
    return {"image_path": _save_image(img, output_path)}


def exec_image_rotate_tool(
    angle: int,
    image_path: str = _DEFAULT_INPUT,
    output_path: str = _DEFAULT_OUTPUT,
) -> dict:
    img = _load_image(image_path)
    rotated = img.rotate(angle, expand=True)
    return {"image_path": _save_image(rotated, output_path)}


# ── Code-execution tools ─────────────────────────────────────────────────────

def exec_python_arithmetic_tool(code: str) -> dict:
    return _exec_code(code)


def exec_python_numerical_compute_tool(code: str) -> dict:
    import numpy as _np, math as _math
    return _exec_code(code, {"numpy": _np, "np": _np, "math": _math})


def exec_python_symbolic_math_tool(code: str) -> dict:
    try:
        import sympy as _sp
    except ImportError:
        return {"error": "sympy is not installed. Run: pip install sympy"}
    return _exec_code(code, {"sympy": _sp})


def exec_python_geometry_compute_tool(code: str) -> dict:
    import math as _math
    extra: dict = {"math": _math}
    try:
        import numpy as _np
        extra.update({"numpy": _np, "np": _np})
    except ImportError:
        pass
    try:
        import sympy as _sp
        extra["sympy"] = _sp
    except ImportError:
        pass
    return _exec_code(code, extra)


def exec_python_scientific_compute_tool(code: str) -> dict:
    try:
        import scipy as _sci
    except ImportError:
        return {"error": "scipy is not installed. Run: pip install scipy"}
    import numpy as _np
    return _exec_code(code, {"scipy": _sci, "numpy": _np, "np": _np})


def exec_python_fraction_arithmetic_tool(code: str) -> dict:
    from fractions import Fraction as _Fraction
    return _exec_code(code, {"Fraction": _Fraction})


# ── Chart / Data tools ───────────────────────────────────────────────────────

def exec_chart_scale_compute_tool(
    calibration_points: list[list[float]],
    query_pixels: list[float],
) -> dict:
    import numpy as _np
    pixels = [p[0] for p in calibration_points]
    values = [p[1] for p in calibration_points]
    results = [float(_np.interp(q, pixels, values)) for q in query_pixels]
    lines = [f"pixel {q} → {v:.6g}" for q, v in zip(query_pixels, results)]
    return {"result": "\n".join(lines)}


def exec_chart_visualization_tool(
    code: str,
    output_path: str = _DEFAULT_OUTPUT,
) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"error": "matplotlib is not installed. Run: pip install matplotlib"}

    buf = io.StringIO()
    ns = {"__builtins__": __builtins__, "matplotlib": matplotlib, "plt": plt}
    try:
        with contextlib.redirect_stdout(buf):
            exec(textwrap.dedent(code), ns)
        fignums = plt.get_fignums()
        if not fignums:
            return {"error": "No matplotlib figure was created by the script.",
                    "stdout": buf.getvalue()}
        plt.figure(fignums[-1]).savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close("all")
        return {"image_path": output_path}
    except Exception:
        plt.close("all")
        return {"error": traceback.format_exc(), "stdout": buf.getvalue()}


def exec_data_tabulate_tool(code: str) -> dict:
    import pandas as _pd
    return _exec_code(code, {"pd": _pd, "pandas": _pd})


# ── Dispatcher ───────────────────────────────────────────────────────────────

_EXECUTORS: dict[str, object] = {
    "image_zoom_in_tool":              exec_image_zoom_in_tool,
    "image_crop_tool":                 exec_image_crop_tool,
    "image_draw_reference_line_tool":  exec_image_draw_reference_line_tool,
    "image_draw_bounding_box_tool":    exec_image_draw_bounding_box_tool,
    "image_annotate_points_tool":      exec_image_annotate_points_tool,
    "image_get_pixel_color_tool":      exec_image_get_pixel_color_tool,
    "image_draw_text_tool":            exec_image_draw_text_tool,
    "image_draw_arrow_tool":           exec_image_draw_arrow_tool,
    "image_rotate_tool":               exec_image_rotate_tool,
    "python_arithmetic_tool":          exec_python_arithmetic_tool,
    "python_numerical_compute_tool":   exec_python_numerical_compute_tool,
    "python_symbolic_math_tool":       exec_python_symbolic_math_tool,
    "python_geometry_compute_tool":    exec_python_geometry_compute_tool,
    "python_scientific_compute_tool":  exec_python_scientific_compute_tool,
    "python_fraction_arithmetic_tool": exec_python_fraction_arithmetic_tool,
    "chart_scale_compute_tool":        exec_chart_scale_compute_tool,
    "chart_visualization_tool":        exec_chart_visualization_tool,
    "data_tabulate_tool":              exec_data_tabulate_tool,
}


def call_tool(
    tool_name: str,
    arguments: dict,
    image_path: str = _DEFAULT_INPUT,
    output_path: str = _DEFAULT_OUTPUT,
) -> dict:
    """
    Dispatch a VLM tool call to the appropriate sandbox executor.

    Args:
        tool_name:   Name matching a key in ALL_TOOLS / _EXECUTORS.
        arguments:   Parameter dict exactly as the VLM produced it.
        image_path:  Current working image path (image tools only).
        output_path: Destination path for image tool output.

    Returns:
        Image tools:   {"image_path": "<saved path>"}
        Compute tools: {"result": "<printed output>"}
        On error:      {"error": "<traceback>", "stdout": "<partial>"}
    """
    import inspect
    fn = _EXECUTORS.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name!r}. Available: {list(_EXECUTORS)}"}

    sig = inspect.signature(fn)
    kwargs = dict(arguments)
    if "image_path"  in sig.parameters and "image_path"  not in kwargs:
        kwargs["image_path"]  = image_path
    if "output_path" in sig.parameters and "output_path" not in kwargs:
        kwargs["output_path"] = output_path

    try:
        return fn(**kwargs)
    except TypeError as e:
        return {"error": f"Argument mismatch for {tool_name!r}: {e}"}
    except Exception:
        return {"error": traceback.format_exc()}


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    print(json.dumps(ALL_TOOLS, indent=2, ensure_ascii=False))

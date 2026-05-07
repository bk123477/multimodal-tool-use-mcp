#!/usr/bin/env python3
"""
Visual Reasoning Tools — MCP Server
====================================
Exposes all 18 atomic visual-reasoning tools as MCP tools via FastMCP.

Run (stdio, for Claude Desktop / MCP clients):
    python mcp_server.py

Run (HTTP/SSE, for remote access):
    python mcp_server.py --transport sse --port 8000

Tool result conventions
-----------------------
- Image tools  → [TextContent(path info), ImageContent(base64 JPEG)]
- Compute tools → TextContent(printed result)
- Errors        → TextContent("ERROR: ...")
"""

from __future__ import annotations

import base64
import os
import sys

from fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

# Ensure tool_definitions.py is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tool_definitions as _td

# ---------------------------------------------------------------------------
mcp = FastMCP(
    "visual-reasoning-tools",
    instructions=(
        "Tools for atomic visual reasoning on images. "
        "Image tools read from image_path and write to output_path. "
        "Pass output_path from a previous image tool as image_path for chained edits. "
        "Compute tools accept a Python code string and return printed output."
    ),
)
# ---------------------------------------------------------------------------


# ── Internal helpers ─────────────────────────────────────────────────────────

def _img_response(result: dict) -> list[TextContent | ImageContent]:
    if "error" in result:
        return [TextContent(type="text", text=f"ERROR: {result['error']}")]
    path = result["image_path"]
    if not os.path.exists(path):
        return [TextContent(type="text", text=f"ERROR: output image not found at '{path}'")]
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return [
        TextContent(type="text", text=f"image_path: {path}"),
        ImageContent(type="image", data=b64, mimeType=mime),
    ]


def _txt_response(result: dict) -> str:
    if "error" in result:
        return f"ERROR: {result['error']}"
    return result.get("result", "")


# ── 1. Image tools ───────────────────────────────────────────────────────────

@mcp.tool()
def image_zoom_in_tool(
    bbox_2d: list[float],
    image_path: str = "input_image.jpg",
    output_path: str = "output_image.jpg",
    label: str = "",
) -> list[TextContent | ImageContent]:
    """
    Zoom in on a region of the image by cropping to bbox_2d and upscaling.
    The shorter side of the crop is scaled to ≥ 400 px (max 4×, LANCZOS).

    Args:
        bbox_2d: [x1, y1, x2, y2] bounding box in pixel coordinates.
        image_path: Path of the source image to read.
        output_path: Path to save the zoomed result.
        label: Optional region label for bookkeeping (does not affect output).
    """
    result = _td.exec_image_zoom_in_tool(
        bbox_2d=bbox_2d,
        label=label or None,
        image_path=image_path,
        output_path=output_path,
    )
    return _img_response(result)


@mcp.tool()
def image_crop_tool(
    bbox_2d: list[float],
    image_path: str = "input_image.jpg",
    output_path: str = "output_image.jpg",
) -> list[TextContent | ImageContent]:
    """
    Crop a rectangular region from the image at its original resolution (no upscaling).

    Args:
        bbox_2d: [x1, y1, x2, y2] bounding box in pixel coordinates.
        image_path: Path of the source image to read.
        output_path: Path to save the cropped result.
    """
    result = _td.exec_image_crop_tool(
        bbox_2d=bbox_2d,
        image_path=image_path,
        output_path=output_path,
    )
    return _img_response(result)


@mcp.tool()
def image_draw_reference_line_tool(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_path: str = "input_image.jpg",
    output_path: str = "output_image.jpg",
    color: list[int] = None,
    width: int = 2,
) -> list[TextContent | ImageContent]:
    """
    Draw a straight reference line from (x1, y1) to (x2, y2) on the image.
    Useful for marking threshold values on charts.

    Args:
        x1: Start x-coordinate (pixels from left).
        y1: Start y-coordinate (pixels from top).
        x2: End x-coordinate.
        y2: End y-coordinate.
        image_path: Path of the source image to read.
        output_path: Path to save the annotated result.
        color: RGB color as [R, G, B]. Default red [255, 0, 0].
        width: Line thickness in pixels. Default 2.
    """
    result = _td.exec_image_draw_reference_line_tool(
        x1=x1, y1=y1, x2=x2, y2=y2,
        color=color or [255, 0, 0],
        width=width,
        image_path=image_path,
        output_path=output_path,
    )
    return _img_response(result)


@mcp.tool()
def image_draw_bounding_box_tool(
    bbox_2d: list[float],
    image_path: str = "input_image.jpg",
    output_path: str = "output_image.jpg",
    color: list[int] = None,
    width: int = 2,
    label: str = "",
) -> list[TextContent | ImageContent]:
    """
    Draw a rectangular bounding box outline on the image.

    Args:
        bbox_2d: [x1, y1, x2, y2] bounding box in pixel coordinates.
        image_path: Path of the source image to read.
        output_path: Path to save the annotated result.
        color: RGB outline color. Default green [0, 255, 0].
        width: Outline thickness in pixels. Default 2.
        label: Optional text label drawn near the top-left corner.
    """
    result = _td.exec_image_draw_bounding_box_tool(
        bbox_2d=bbox_2d,
        color=color or [0, 255, 0],
        width=width,
        label=label or None,
        image_path=image_path,
        output_path=output_path,
    )
    return _img_response(result)


@mcp.tool()
def image_annotate_points_tool(
    points: list[list[float]],
    image_path: str = "input_image.jpg",
    output_path: str = "output_image.jpg",
    radius: int = 6,
) -> list[TextContent | ImageContent]:
    """
    Draw numbered circle markers at pixel coordinates. Each point gets three
    concentric circles (blue/green/red) with a 1-based index label beside it.

    Args:
        points: List of [x, y] pixel coordinates to annotate.
        image_path: Path of the source image to read.
        output_path: Path to save the annotated result.
        radius: Radius of the innermost circle in pixels. Default 6.
    """
    result = _td.exec_image_annotate_points_tool(
        points=points,
        radius=radius,
        image_path=image_path,
        output_path=output_path,
    )
    return _img_response(result)


@mcp.tool()
def image_get_pixel_color_tool(
    x: int,
    y: int,
    image_path: str = "input_image.jpg",
) -> str:
    """
    Read the RGB color of a single pixel at (x, y).
    Returns a string like "[R, G, B]".

    Args:
        x: Pixel x-coordinate (column, 0 = left edge).
        y: Pixel y-coordinate (row, 0 = top edge).
        image_path: Path of the source image to read.
    """
    result = _td.exec_image_get_pixel_color_tool(x=x, y=y, image_path=image_path)
    return _txt_response(result)


@mcp.tool()
def image_draw_text_tool(
    x: int,
    y: int,
    text: str,
    image_path: str = "input_image.jpg",
    output_path: str = "output_image.jpg",
    color: list[int] = None,
    font_size: int = 16,
) -> list[TextContent | ImageContent]:
    """
    Render a text annotation onto the image at pixel position (x, y).
    The top-left corner of the text block is placed at (x, y).

    Args:
        x: Left edge x-coordinate of the text.
        y: Top edge y-coordinate of the text.
        text: String to render on the image.
        image_path: Path of the source image to read.
        output_path: Path to save the annotated result.
        color: RGB text color. Default white [255, 255, 255].
        font_size: Font size in points. Default 16.
    """
    result = _td.exec_image_draw_text_tool(
        x=x, y=y, text=text,
        color=color or [255, 255, 255],
        font_size=font_size,
        image_path=image_path,
        output_path=output_path,
    )
    return _img_response(result)


@mcp.tool()
def image_draw_arrow_tool(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_path: str = "input_image.jpg",
    output_path: str = "output_image.jpg",
    color: list[int] = None,
    thickness: int = 2,
    tip_length: float = 0.3,
) -> list[TextContent | ImageContent]:
    """
    Draw an arrow from tail (x1, y1) to head (x2, y2). The arrowhead is placed
    at (x2, y2). Uses cv2.arrowedLine when available, falls back to PIL polygon.

    Args:
        x1: Tail x-coordinate (arrow start).
        y1: Tail y-coordinate (arrow start).
        x2: Head x-coordinate (arrowhead end).
        y2: Head y-coordinate (arrowhead end).
        image_path: Path of the source image to read.
        output_path: Path to save the annotated result.
        color: RGB color. Default blue [0, 0, 255].
        thickness: Shaft thickness in pixels. Default 2.
        tip_length: Arrowhead as fraction of total length (0.05–0.9). Default 0.3.
    """
    result = _td.exec_image_draw_arrow_tool(
        x1=x1, y1=y1, x2=x2, y2=y2,
        color=color or [0, 0, 255],
        thickness=thickness,
        tip_length=tip_length,
        image_path=image_path,
        output_path=output_path,
    )
    return _img_response(result)


@mcp.tool()
def image_rotate_tool(
    angle: int,
    image_path: str = "input_image.jpg",
    output_path: str = "output_image.jpg",
) -> list[TextContent | ImageContent]:
    """
    Rotate the image counter-clockwise by 90, 180, or 270 degrees.
    Canvas is resized to fit the rotated image (no cropping).

    Args:
        angle: Rotation angle — must be 90, 180, or 270.
        image_path: Path of the source image to read.
        output_path: Path to save the rotated result.
    """
    if angle not in (90, 180, 270):
        return [TextContent(type="text", text="ERROR: angle must be 90, 180, or 270.")]
    result = _td.exec_image_rotate_tool(
        angle=angle,
        image_path=image_path,
        output_path=output_path,
    )
    return _img_response(result)


# ── 2. Compute tools ─────────────────────────────────────────────────────────

@mcp.tool()
def python_arithmetic_tool(code: str) -> str:
    """
    Execute a pure-Python arithmetic script (no external libraries) and return
    printed output. Use for integer/float arithmetic, combinatorics, set counts,
    ratio/percentage calculations.

    Rules:
      - Use only Python built-ins (int, float, list, dict, sum, abs, …).
      - Must contain at least one print() statement.

    Args:
        code: Complete Python script. Example: "total = 3*4 + 12//5\\nprint(total)"
    """
    return _txt_response(_td.exec_python_arithmetic_tool(code))


@mcp.tool()
def python_numerical_compute_tool(code: str) -> str:
    """
    Execute a Python script using numpy or the math module for numerical
    computation. Use for array operations, trigonometry, logarithms, statistics.

    Rules:
      - import numpy as np  and/or  import math  as needed.
      - Must contain at least one print() statement.

    Args:
        code: Complete Python script. Example: "import numpy as np\\nprint(np.mean([3,7,5,12]))"
    """
    return _txt_response(_td.exec_python_numerical_compute_tool(code))


@mcp.tool()
def python_symbolic_math_tool(code: str) -> str:
    """
    Execute a Python script using sympy for symbolic mathematics. Use for
    equation solving, differentiation, integration, exact simplification.

    Rules:
      - from sympy import ...  or  import sympy.
      - Must contain at least one print() statement.

    Args:
        code: Complete Python script using sympy.
              Example: "from sympy import symbols, solve\\nx=symbols('x')\\nprint(solve(x**2-4,x))"
    """
    return _txt_response(_td.exec_python_symbolic_math_tool(code))


@mcp.tool()
def python_geometry_compute_tool(code: str) -> str:
    """
    Execute a Python script for geometric computation: distances, areas,
    perimeters, angles, volumes, coordinate geometry. May use math, numpy, sympy.

    Rules:
      - Import whatever is needed.
      - Must contain at least one print() statement.

    Args:
        code: Complete Python script. Example: "import math\\nprint(math.sqrt(3**2+4**2))"
    """
    return _txt_response(_td.exec_python_geometry_compute_tool(code))


@mcp.tool()
def python_scientific_compute_tool(code: str) -> str:
    """
    Execute a Python script using scipy for scientific computation: optimization,
    numerical integration, interpolation, linear algebra, statistical tests.

    Rules:
      - import scipy.*  and  import numpy  as needed.
      - Must contain at least one print() statement.

    Args:
        code: Complete Python script using scipy.
    """
    return _txt_response(_td.exec_python_scientific_compute_tool(code))


@mcp.tool()
def python_fraction_arithmetic_tool(code: str) -> str:
    """
    Execute a Python script using fractions.Fraction for exact rational arithmetic.
    Avoids all floating-point rounding error. Use for fraction add/sub/mul/div and
    exact ratio comparisons.

    Rules:
      - from fractions import Fraction.
      - Must contain at least one print() statement.

    Args:
        code: Complete Python script. Example: "from fractions import Fraction\\nprint(Fraction(1,3)+Fraction(1,6))"
    """
    return _txt_response(_td.exec_python_fraction_arithmetic_tool(code))


# ── 3. Chart / Data tools ────────────────────────────────────────────────────

@mcp.tool()
def chart_scale_compute_tool(
    calibration_points: list[list[float]],
    query_pixels: list[float],
) -> str:
    """
    Convert pixel coordinates to chart axis values via linear interpolation.
    No code needed — pass structured parameters only.

    How to use:
      1. Identify ≥2 pixel positions on a chart axis where you know the true value.
      2. Pass as calibration_points: [[pixel, value], …].
      3. Pass pixels to convert as query_pixels.

    Example:
      calibration_points=[[133, 100], [421, 0]], query_pixels=[280]
      → "pixel 280 → 48.9583"

    Args:
        calibration_points: List of [pixel_coordinate, chart_axis_value] anchor pairs.
        query_pixels: Pixel coordinates to convert to chart values.
    """
    return _txt_response(_td.exec_chart_scale_compute_tool(
        calibration_points=calibration_points,
        query_pixels=query_pixels,
    ))


@mcp.tool()
def chart_visualization_tool(
    code: str,
    output_path: str = "output_image.jpg",
) -> list[TextContent | ImageContent]:
    """
    Execute a matplotlib script and return the rendered chart as an image.
    Use to re-plot extracted data for visual verification.

    Rules:
      - import matplotlib.pyplot as plt  at the top.
      - Do NOT call plt.show() — the figure is auto-captured.
      - Use plt.figure(), plt.plot(), plt.bar(), etc. as needed.

    Args:
        code: Complete matplotlib script. Must not call plt.show().
        output_path: Path to save the rendered chart image.
    """
    result = _td.exec_chart_visualization_tool(code=code, output_path=output_path)
    return _img_response(result)


@mcp.tool()
def data_tabulate_tool(code: str) -> str:
    """
    Execute a pandas script to organize, aggregate, or analyze tabular data
    and return the printed result. Use for grouping, filtering, pivot tables,
    and descriptive statistics.

    Rules:
      - import pandas as pd.
      - Construct DataFrames from Python dicts/lists — do NOT read files.
      - Must contain at least one print() statement.

    Args:
        code: Complete pandas script.
              Example: "import pandas as pd\\ndf=pd.DataFrame({'a':[1,2],'b':[3,4]})\\nprint(df)"
    """
    return _txt_response(_td.exec_data_tabulate_tool(code))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visual Reasoning Tools MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"],
                        default="stdio", help="Transport protocol (default: stdio)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for SSE/HTTP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE/HTTP (default: 8000)")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)

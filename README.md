# Visual Reasoning Tools

VLM이 이미지를 보고 추론하는 과정에서 호출할 수 있는 19개의 atomic tool 모음입니다.
PIL/OpenCV/NumPy/Pandas 등을 이용해 이미지 조작과 수식 계산을 실제로 실행합니다.

---

## 파일 구조

```
.
├── tool_definitions.py   # 스키마(VLM용) + 실제 executor 구현
├── mcp_server.py         # MCP 서버 (FastMCP 기반)
└── README.md
```

| 파일 | 역할 |
|---|---|
| `tool_definitions.py` | `ALL_TOOLS` (VLM에 전달하는 JSON 스키마) + `exec_*()` 함수 + `call_tool()` 디스패처 |
| `mcp_server.py` | `@mcp.tool()` 19개로 외부 MCP 클라이언트에 노출 |

---

## 설치

```bash
# 필수
pip install "mcp[cli]" fastmcp pillow pandas numpy

# 선택 (미설치 시 해당 tool은 설치 안내 에러 반환)
pip install sympy scipy matplotlib opencv-python
```

---

## Tool 목록

### 이미지 도구 (10종)

| Tool | 설명 |
|---|---|
| `image_zoom_in_tool` | bbox 영역을 크롭 후 업스케일 (짧은 변 ≥ 400px, 최대 4×) |
| `image_crop_tool` | bbox 영역을 원본 해상도 그대로 크롭 |
| `image_draw_reference_line_tool` | 두 좌표 사이에 기준선 그리기 (차트 임계값 표시 등) |
| `image_draw_bounding_box_tool` | 직사각형 바운딩 박스 + 선택적 텍스트 라벨 |
| `image_annotate_points_tool` | 좌표 목록에 번호 마커(동심원 3겹) 표시 |
| `image_get_pixel_color_tool` | 특정 픽셀의 RGB 색상값 반환 |
| `image_draw_text_tool` | 지정 위치에 텍스트 렌더링 |
| `image_draw_arrow_tool` | 두 좌표 사이에 화살표 그리기 (`cv2.arrowedLine`, 없으면 PIL 폴백) |
| `image_rotate_tool` | 90 / 180 / 270도 회전 (캔버스 자동 확장) |
| `image_merge_tool` | 여러 이미지를 좌우 또는 상하로 병합하고 정렬/간격/배경색 지정 |

### 수식 계산 도구 (6종)

| Tool | 설명 |
|---|---|
| `python_arithmetic_tool` | 순수 Python 산술 (외부 라이브러리 없음) |
| `python_numerical_compute_tool` | numpy / math 수치 계산 |
| `python_symbolic_math_tool` | sympy 기호 수학 (방정식 풀기, 미적분 등) |
| `python_geometry_compute_tool` | 기하 계산 (거리, 넓이, 각도 등) |
| `python_scientific_compute_tool` | scipy 과학 계산 (최적화, 보간, 통계 검정 등) |
| `python_fraction_arithmetic_tool` | `fractions.Fraction` 정확한 분수 연산 |

### 차트 / 데이터 도구 (3종)

| Tool | 설명 |
|---|---|
| `chart_scale_compute_tool` | 픽셀 좌표 → 차트 축 값 변환 (선형 보간, 코드 불필요) |
| `chart_visualization_tool` | matplotlib으로 차트를 그려 이미지로 반환 |
| `data_tabulate_tool` | pandas로 데이터 집계/분석 후 결과 텍스트 반환 |

---

## Tool 반환 형태

| 종류 | 반환값 |
|---|---|
| 이미지 도구 | `[TextContent("image_path: ..."), ImageContent(base64 JPEG)]` |
| 계산 도구 | `[TextContent("결과 텍스트")]` |
| 에러 발생 시 | `[TextContent("ERROR: ...")]` |
| 라이브러리 미설치 | `[TextContent("ERROR: sympy is not installed. Run: pip install sympy")]` |

---

## 사용 방법

### 방법 1 — `call_tool()` 직접 호출 (학습 데이터 생성 / 커스텀 추론 루프)

```python
from tool_definitions import call_tool, ALL_TOOLS

# VLM이 생성한 tool call을 그대로 넘기면 됨
result = call_tool(
    tool_name="image_draw_arrow_tool",
    arguments={"x1": 100, "y1": 50, "x2": 300, "y2": 280,
               "color": [255, 0, 0], "thickness": 3},
    image_path="input_image.jpg",   # 현재 working 이미지
    output_path="output_image.jpg"  # 결과 저장 경로
)
# {"image_path": "output_image.jpg"}
```

이미지 여러 장을 하나로 붙이고 싶다면:

```python
result = call_tool(
    tool_name="image_merge_tool",
    arguments={
        "image_paths": ["original.jpg", "tool_result.jpg"],
        "direction": "horizontal",
        "align": "center",
        "gap": 16,
        "background_color": [255, 255, 255],
    },
    output_path="merged.jpg",
)
# {"image_path": "merged.jpg"}
```

### 방법 2 — VLM 추론 루프에 연결

```python
import json
from tool_definitions import call_tool, ALL_TOOLS

def run_vlm_with_tools(vlm, image_path, question):
    messages = [{"role": "user", "content": question}]
    current_image = image_path

    while True:
        # ALL_TOOLS를 tools 파라미터로 전달 → VLM이 어떤 tool이 있는지 인식
        response = vlm(
            image=current_image,
            messages=messages,
            tools=ALL_TOOLS
        )

        if response.stop_reason == "end_turn":
            return response.content  # 최종 답변

        for tool_call in response.tool_calls:
            result = call_tool(
                tool_name=tool_call["name"],
                arguments=tool_call["arguments"],
                image_path=current_image,
                output_path="output_image.jpg",
            )

            # 이미지 도구라면 다음 턴에 수정된 이미지를 봄
            if "image_path" in result:
                current_image = result["image_path"]

            # tool 결과를 messages에 추가 후 VLM 재호출
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result),
            })
```

### 방법 3 — MCP 서버로 실행 (Claude Desktop / MCP 지원 클라이언트)

```bash
# stdio (기본, Claude Desktop용)
python mcp_server.py

# HTTP/SSE (원격 접근, EC2 등)
python mcp_server.py --transport sse --host 0.0.0.0 --port 8000

# Streamable HTTP
python mcp_server.py --transport streamable-http --port 8000
```

---

## MCP 클라이언트 설정

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "visual-reasoning": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/this/directory"
    }
  }
}
```

### 원격 서버 (SSE)

서버를 `--transport sse --port 8000`으로 띄운 후 클라이언트에서:

```python
from fastmcp import Client

async with Client("http://<EC2_IP>:8000/sse") as client:
    result = await client.call_tool("image_draw_arrow_tool", {
        "x1": 100, "y1": 50, "x2": 300, "y2": 280,
        "image_path": "input_image.jpg",
        "output_path": "output_image.jpg",
    })
```

### `mcp` CLI로 바로 테스트

```bash
# tool 목록 확인
mcp dev mcp_server.py

# 특정 tool 호출
mcp call mcp_server.py image_get_pixel_color_tool '{"x": 100, "y": 200, "image_path": "input_image.jpg"}'
```

---

## 이미지 경로 규칙

이미지 도구는 `image_path`를 읽고 `output_path`에 저장합니다.  
여러 도구를 연속으로 쓸 때는 앞 도구의 `output_path`를 다음 도구의 `image_path`로 넘기세요.

```
1단계: image_draw_reference_line_tool
        image_path="input_image.jpg"  →  output_path="step1.jpg"

2단계: image_draw_bounding_box_tool
        image_path="step1.jpg"        →  output_path="step2.jpg"

3단계: image_annotate_points_tool
        image_path="step2.jpg"        →  output_path="final.jpg"
```

기본값: `image_path="input_image.jpg"`, `output_path="output_image.jpg"`

---

## Tool별 파라미터 요약

### 이미지 도구

```
image_zoom_in_tool(bbox_2d, image_path, output_path, label="")
image_crop_tool(bbox_2d, image_path, output_path)
image_draw_reference_line_tool(x1, y1, x2, y2, image_path, output_path,
                               color=[255,0,0], width=2)
image_draw_bounding_box_tool(bbox_2d, image_path, output_path,
                             color=[0,255,0], width=2, label="")
image_annotate_points_tool(points, image_path, output_path, radius=6)
image_get_pixel_color_tool(x, y, image_path)          → "[R, G, B]"
image_draw_text_tool(x, y, text, image_path, output_path,
                     color=[255,255,255], font_size=16)
image_draw_arrow_tool(x1, y1, x2, y2, image_path, output_path,
                      color=[0,0,255], thickness=2, tip_length=0.3)
image_rotate_tool(angle, image_path, output_path)      # angle: 90|180|270
image_merge_tool(image_paths, output_path,
                 direction="horizontal", align="center",
                 gap=0, background_color=[255,255,255])
```

### 계산 도구 (`code` 문자열 필수, `print()` 포함)

```
python_arithmetic_tool(code)          # 순수 Python
python_numerical_compute_tool(code)   # numpy, math
python_symbolic_math_tool(code)       # sympy
python_geometry_compute_tool(code)    # math, numpy, sympy
python_scientific_compute_tool(code)  # scipy, numpy
python_fraction_arithmetic_tool(code) # fractions.Fraction
```

### 차트 / 데이터

```
chart_scale_compute_tool(calibration_points, query_pixels)
  # calibration_points: [[pixel, value], ...]  (≥ 2개 필요)
  # query_pixels: [pixel, ...]

chart_visualization_tool(code, output_path)
  # matplotlib 스크립트, plt.show() 호출 금지

data_tabulate_tool(code)
  # pandas 스크립트, 파일 읽기 금지 (dict/list로 직접 생성)
```

---

## 주의사항

- **계산 도구의 `code`** 는 `exec()`로 실행됩니다. 신뢰할 수 없는 코드를 넣지 마세요.
- **`chart_visualization_tool`** 에서 `plt.show()` 를 호출하면 서버가 블로킹됩니다.
- **`chart_scale_compute_tool`** 은 코드가 필요 없는 유일한 계산 도구입니다.
- sympy / scipy / matplotlib / opencv-python이 없으면 해당 도구는 설치 안내 에러를 반환합니다. 다른 도구는 정상 동작합니다.

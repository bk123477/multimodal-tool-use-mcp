from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from fastmcp import Client as FastMCPClient


MCP_DIR = Path(__file__).resolve().parent
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

import tool_definitions as td


DEFAULT_SYSTEM_PREFIX = (
    "You are a helpful assistant.\n\n"
    "Solve the following problem step by step. # Tools\n"
    "You may call one or more functions to assist with the user query.\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_VLM_MODEL = "qwen/qwen3.5-397b-a17b"
DEFAULT_MAX_TOKENS = 30000
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_project_env(env_path: Path = Path("/home/minkih/.env")) -> None:
    if env_path.exists():
        load_dotenv(env_path)
    load_dotenv()


def compact_tools_for_prompt(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool["function"]
        compact.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    return compact


def tools_for_api(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    api_tools: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool["function"]
        api_tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        )
    return api_tools


def build_system_content(tools: list[dict[str, Any]]) -> str:
    tool_json = json.dumps(compact_tools_for_prompt(tools), ensure_ascii=False, indent=2)
    return f"{DEFAULT_SYSTEM_PREFIX}<tools>\n{tool_json}\n</tools>"


def tool_names(tools: Iterable[dict[str, Any]]) -> list[str]:
    return [tool["function"]["name"] for tool in tools]


def get_tools_by_names(names: Iterable[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for name in names:
        tool = td.TOOL_BY_NAME.get(name)
        if tool is not None and tool not in selected:
            selected.append(tool)
    return selected


def choose_tool_subset(
    required_tool_names: list[str],
    subset_size: int = 3,
    seed: int = 0,
) -> list[dict[str, Any]]:
    required = get_tools_by_names(required_tool_names)
    if not required:
        return list(td.ALL_TOOLS)

    rng = random.Random(seed)
    selected = list(required)
    pool = [tool for tool in td.ALL_TOOLS if tool not in selected]
    rng.shuffle(pool)
    selected.extend(pool[: max(0, subset_size - len(selected))])
    return selected


def build_tool_policy(has_prior_tool_call: bool, allowed_tools: list[str]) -> dict[str, Any]:
    if has_prior_tool_call:
        return {"tool_choice": "required", "allowed_tools": allowed_tools}
    return {"tool_choice": "auto", "allowed_tools": allowed_tools}


def strip_codedance_prompt_suffix(text: str) -> str:
    text = text.replace("<image>", "", 1)
    marker = "\nThink step-by-step within <think></think>."
    if marker in text:
        text = text.split(marker, 1)[0]
    return text.strip()


def extract_code_blocks(text: str) -> list[str]:
    pattern = re.compile(r"<code>\s*```(?:python)?\s*(.*?)```\s*</code>", re.DOTALL)
    blocks = [match.group(1).strip() for match in pattern.finditer(text)]
    if blocks:
        return blocks
    fallback = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
    return [match.group(1).strip() for match in fallback.finditer(text)]


def heuristic_tool_from_code(code: str) -> str | None:
    lower = code.lower()
    compact = re.sub(r"\s+", "", lower)

    if "matplotlib" in lower or "plt." in lower:
        return "chart_visualization_tool"
    if "pandas" in lower or "dataframe" in lower:
        return "data_tabulate_tool"
    if "sympy" in lower:
        return "python_symbolic_math_tool"
    if "scipy" in lower:
        return "python_scientific_compute_tool"
    if "fraction" in lower:
        return "python_fraction_arithmetic_tool"
    if ".crop(" in compact or "image.crop(" in compact:
        return "image_crop_tool"
    if ".rotate(" in compact or "transpose(" in compact:
        return "image_rotate_tool"
    if ".getpixel(" in compact or "getpixel(" in compact:
        return "image_get_pixel_color_tool"
    if "draw.line" in compact or ".line(" in compact:
        return "image_draw_reference_line_tool"
    if "draw.rectangle" in compact or ".rectangle(" in compact:
        return "image_draw_bounding_box_tool"
    if "draw.text" in compact or ".text(" in compact:
        return "image_draw_text_tool"
    if "arrowedline" in compact:
        return "image_draw_arrow_tool"
    if "draw.ellipse" in compact or ".ellipse(" in compact:
        return "image_annotate_points_tool"
    if "numpy" in lower or "np." in lower or "math." in lower:
        if any(word in lower for word in ("distance", "angle", "area", "perimeter", "geometry")):
            return "python_geometry_compute_tool"
        return "python_numerical_compute_tool"
    if "print(" in compact:
        return "python_arithmetic_tool"
    return None


def infer_tools_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    inferred: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for code in extract_code_blocks(str(message.get("content", ""))):
            name = heuristic_tool_from_code(code)
            if name and name not in inferred:
                inferred.append(name)
    return inferred


def classify_tool_with_openai(
    code: str,
    question: str,
    candidate_names: list[str] | None = None,
    model: str = "gpt-5-mini",
) -> str | None:
    from openai import OpenAI

    load_project_env()
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    candidates = candidate_names or tool_names(td.ALL_TOOLS)
    prompt = (
        "Select the single best replacement tool for this old python_executor code. "
        "Return only the exact tool name, nothing else.\n\n"
        f"Allowed tool names:\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
        f"Question:\n{question}\n\nPython code:\n{code}"
    )
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=200,
    )
    answer = response.output_text.strip()
    if answer in candidates:
        return answer
    for name in candidates:
        if name in answer:
            return name
    return None


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def datakit_content_item(item_type: str, value: Any, **extra: Any) -> dict[str, Any]:
    data = {"type": item_type, "value": value, "tool_call_id": extra.pop("tool_call_id", None)}
    data.update(extra)
    return data


def file_uri_relative(jsonl_path: Path, image_path: Path) -> str:
    rel = os.path.relpath(image_path, start=jsonl_path.parent)
    return f"file://{rel}"


def copy_input_image(src: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "0.png"
    shutil.copy2(src, dst)
    return dst


def extract_openrouter_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        args = function.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                args_obj = json.loads(args)
            except json.JSONDecodeError:
                args_obj = {"_raw_arguments": args}
        else:
            args_obj = args
        calls.append(
            {
                "id": raw.get("id") or f"call_{len(calls) + 1:03d}",
                "name": function.get("name"),
                "arguments": args_obj,
            }
        )
    return calls


def normalize_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    current_image: Path,
    output_path: Path,
) -> dict[str, Any]:
    fn = getattr(td, "_EXECUTORS", {}).get(tool_name)
    if fn is None:
        return dict(arguments)
    params = inspect.signature(fn).parameters
    normalized = dict(arguments)
    if "image_path" in params:
        normalized["image_path"] = str(current_image)
    if "output_path" in params:
        normalized["output_path"] = str(output_path)
    return normalized


class OpenRouterChatClient:
    def __init__(
        self,
        model: str = DEFAULT_VLM_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = 300,
    ) -> None:
        load_project_env()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing. Put it in /home/minkih/.env.")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def create(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools_for_api(tools),
            "tool_choice": tool_choice,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            OPENROUTER_BASE_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter error {exc.code}: {body[:2000]}") from exc
        return data["choices"][0]["message"]


class MCPToolRunner:
    def __init__(self, server_path: Path = MCP_DIR / "mcp_server.py", timeout: int = 120) -> None:
        self.server_path = server_path
        self.timeout = timeout

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with FastMCPClient(str(self.server_path), timeout=self.timeout) as client:
            result = await client.call_tool(name, arguments)
        text_parts = []
        image_paths = []
        for content in result.content:
            if getattr(content, "type", None) == "text":
                text = getattr(content, "text", "")
                text_parts.append(text)
                if text.startswith("image_path:"):
                    image_paths.append(text.split(":", 1)[1].strip())
        payload: dict[str, Any] = {
            "status": "error" if result.is_error else "success",
            "text": "\n".join(part for part in text_parts if part),
        }
        if image_paths:
            payload["image_path"] = image_paths[-1]
        if result.structured_content:
            payload["structured_content"] = result.structured_content
        return payload

    def call_sync(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self.call(name, arguments))


@dataclass
class ToolLoopResult:
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]
    output_images: list[Path]


def run_vlm_tool_loop(
    question: str,
    input_image_path: Path,
    output_image_dir: Path,
    tools: list[dict[str, Any]],
    tool_policy: dict[str, Any],
    vlm_client: OpenRouterChatClient,
    mcp_runner: MCPToolRunner,
    jsonl_path: Path,
    max_tool_rounds: int = 4,
) -> ToolLoopResult:
    output_images: list[Path] = []
    system_text = build_system_content(tools)
    datakit_messages = [
        {"role": "system", "content": [datakit_content_item("text", system_text)]},
        {
            "role": "user",
            "content": [
                datakit_content_item(
                    "image",
                    file_uri_relative(jsonl_path, input_image_path),
                    image_alias="input_image",
                ),
                datakit_content_item("text", question),
            ],
        },
    ]
    api_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(input_image_path)}},
                {"type": "text", "text": question},
            ],
        },
    ]

    current_image = input_image_path
    tool_choice = tool_policy["tool_choice"]
    tool_rounds = 0
    final_text = ""

    while True:
        message = vlm_client.create(api_messages, tools=tools, tool_choice=tool_choice)
        content = message.get("content") or ""
        calls = extract_openrouter_tool_calls(message)
        api_messages.append(message)

        if calls:
            assistant_items = []
            if content.strip():
                assistant_items.append(datakit_content_item("reasoning", content.strip()))
            for call in calls:
                call_id = call["id"]
                assistant_items.append(
                    datakit_content_item(
                        "tool_call",
                        json.dumps(
                            {"name": call["name"], "arguments": call["arguments"]},
                            ensure_ascii=False,
                        ),
                        tool_call_id=call_id,
                    )
                )
            datakit_messages.append({"role": "assistant", "content": assistant_items})

            for call in calls:
                tool_rounds += 1
                out_path = output_image_dir / f"{tool_rounds}.png"
                arguments = normalize_tool_arguments(
                    call["name"],
                    call["arguments"],
                    current_image=current_image,
                    output_path=out_path,
                )
                result = mcp_runner.call_sync(call["name"], arguments)
                if result.get("image_path"):
                    current_image = Path(result["image_path"])
                    output_images.append(current_image)
                    result["output_image_path"] = file_uri_relative(jsonl_path, current_image).removeprefix("file://")
                tool_result_json = json.dumps(result, ensure_ascii=False)
                datakit_messages.append(
                    {
                        "role": "tool",
                        "content": [
                            datakit_content_item(
                                "tool_result",
                                tool_result_json,
                                tool_call_id=call["id"],
                            )
                        ],
                    }
                )
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": tool_result_json,
                    }
                )
                if result.get("image_path"):
                    api_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_to_data_url(current_image)},
                                },
                                {
                                    "type": "text",
                                    "text": "Continue reasoning with the tool result image.",
                                },
                            ],
                        }
                    )
            if tool_rounds >= max_tool_rounds:
                tool_choice = "none"
            else:
                tool_choice = "auto"
            continue

        final_text = content.strip()
        datakit_messages.append(
            {"role": "assistant", "content": [datakit_content_item("text", final_text)]}
        )
        break

    return ToolLoopResult(
        messages=datakit_messages,
        metadata={
            "tool_call_turns": tool_rounds,
            "final_text": final_text,
            "vlm_model": vlm_client.model,
            "generated_at_unix": int(time.time()),
        },
        output_images=output_images,
    )

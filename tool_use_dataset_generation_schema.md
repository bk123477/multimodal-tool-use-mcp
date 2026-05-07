# Tool-Use Dataset Generation Schema

This document defines the shared dataset-generation pattern for visual tool-use SFT data.
Dataset-specific code should stay thin: it should normalize the source sample into this schema, then reuse the shared MCP/OpenRouter loop.

## Goal

For each sample, the generator receives:

- an input image
- a user question
- a set of allowed tools
- optional prior trajectory hints from the original dataset

The VLM reasons over the image and question, emits tool calls when useful, sends those calls to `/home/minkih/mcp/mcp_server.py`, receives real tool results, and continues until it produces a final answer.

## Runtime Components

- Tool schemas and executors: `/home/minkih/mcp/tool_definitions.py`
- MCP server: `/home/minkih/mcp/mcp_server.py`
- Shared generation helpers: `/home/minkih/mcp/tooluse_dataset_common.py`
- Dataset-specific generator example: `/home/minkih/CodeDance-SFT/src/generate_tooluse_dataset.py`

The MCP server is central and should not be duplicated under each dataset directory.

## Sample-Level Output Shape

Each JSONL row should contain:

```json
{
  "id": "codedance_00000",
  "messages": [],
  "metadata": {
    "source": "CodeDance-SFT",
    "conversion_mode": "codedance_openrouter_mcp_generation"
  },
  "tools": [],
  "tool_policy": {
    "tool_choice": "required",
    "allowed_tools": ["image_draw_reference_line_tool"]
  },
  "images": ["../images/part_0/0/0.png", "../images/part_0/0/1.png"]
}
```

`tools` is the actual tool list exposed to the model for this sample. `tool_policy` is control metadata for the generator, dataloader, or serving stack; it is not text the model learns directly.

## Messages

Use datakit-style typed content:

```json
{
  "role": "user",
  "content": [
    {
      "type": "image",
      "value": "file://../images/part_0/0/0.png",
      "tool_call_id": null,
      "image_alias": "input_image"
    },
    {
      "type": "text",
      "value": "Which country has its hours worked value between 1.8k and 1.9k?",
      "tool_call_id": null
    }
  ]
}
```

Assistant tool-call turns should be represented as:

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "reasoning",
      "value": "I need a visual threshold line...",
      "tool_call_id": null
    },
    {
      "type": "tool_call",
      "value": "{\"name\":\"image_draw_reference_line_tool\",\"arguments\":{\"x1\":38,\"y1\":303,\"x2\":864,\"y2\":303}}",
      "tool_call_id": "call_001"
    }
  ]
}
```

Tool results should be represented as:

```json
{
  "role": "tool",
  "content": [
    {
      "type": "tool_result",
      "value": "{\"status\":\"success\",\"image_path\":\"/abs/path/1.png\",\"output_image_path\":\"../images/part_0/0/1.png\"}",
      "tool_call_id": "call_001"
    }
  ]
}
```

## System Prompt

The system turn should begin with:

```text
You are a helpful assistant.

Solve the following problem step by step. # Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
```

Then append a `<tools>` block containing only `name`, `description`, and `parameters` for the tools exposed to the sample.

The same tools should also be sent through the model API `tools` parameter so the model can emit structured function calls.

## Tool Selection Policy

If the original sample has no tool call:

- expose all MCP tools, currently 19 tools from `ALL_TOOLS`
- set `tool_policy.tool_choice` to `"auto"`
- let the VLM decide whether a tool is needed

If the original sample has a tool call:

- infer the closest MCP atomic tool from the original trajectory
- build a small subset, usually 3 tools, including the inferred required tool
- set `tool_policy.tool_choice` to `"required"`
- pass `tool_choice="required"` to the model API for the first turn
- after the first tool turn, switch to `"auto"` so the model can continue or answer

The OpenAI/OpenRouter-style tool-choice values are sample-level runtime controls, not message contents.

## Image Layout

Dataset JSONL for CodeDance is written to:

```text
/home/minkih/CodeDance-SFT/generated_samples/sft_dataset_codedance.jsonl
```

Images are written to:

```text
/home/minkih/CodeDance-SFT/images/part_{index // bucket_size}/{index}/{image_number}.png
```

For each sample:

- `0.png` is the original input image
- `1.png`, `2.png`, ... are MCP image tool results

Inside JSONL, image values use `file://` with a path relative to the JSONL directory. Because the JSONL lives under `generated_samples`, CodeDance image values look like:

```text
file://../images/part_0/0/0.png
```

Allowed image URI prefixes are:

```python
("http://", "https://", "file://", "data:image/")
```

For persisted dataset rows, use `file://`.

## OpenRouter VLM

Default VLM:

```text
qwen/qwen3.5-397b-a17b
```

Default generation limit:

```text
max_tokens = 30000
```

The OpenRouter key is loaded from `/home/minkih/.env` as `OPENROUTER_API_KEY`.

## Optional OpenAI Classifier

When a dataset only has a generic old tool such as `python_executor`, use one of:

- heuristic parsing of the old code
- `gpt-5-mini` as a fallback classifier

For `gpt-5-mini`, use the Responses API and avoid unsupported chat-style knobs. The implemented classifier uses:

```python
client.responses.create(
    model="gpt-5-mini",
    input=prompt,
    max_output_tokens=200,
)
```

It intentionally does not set `temperature`.

## Dataset Adapter Checklist

For a new dataset, implement a small adapter that provides:

1. `question`: cleaned user question text
2. `input_image`: path to the original image only
3. `prior_tool_names`: inferred MCP tool names, or an empty list
4. `id` and source metadata
5. output image directory layout

Then call `run_vlm_tool_loop()` from `/home/minkih/mcp/tooluse_dataset_common.py`.

Keep dataset-specific parsing under the dataset directory; keep tool schemas, MCP execution, OpenRouter calls, datakit content helpers, and path helpers under `/home/minkih/mcp`.

# replicate-mcp

A machine learning model inference MCP server powered by [Replicate](https://replicate.com), built on [Dedalus](https://dedaluslabs.ai).

## Tools

| Tool | Description | Read/Write |
|------|-------------|-----------|
| `list_models` | List available ML models | Read |
| `get_model` | Get details for a specific model | Read |
| `list_versions` | List versions of a model | Read |
| `run_prediction` | Run a prediction (with optional polling for completion) | Write |
| `get_prediction` | Get prediction status and output | Read |
| `list_predictions` | List recent predictions | Read |
| `cancel_prediction` | Cancel a running prediction | Write |
| `list_collections` | List featured model collections | Read |

## Prediction Modes

`run_prediction` supports two modes:

- **By version**: Pass `version` (UUID) to run a specific model version
- **By model**: Pass `model_owner` + `model_name` to run the latest version of a model

Set `wait=True` (default) to poll until the prediction completes, or `wait=False` to return immediately and check status with `get_prediction`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `REPLICATE_API_TOKEN` | Yes | Replicate API token (`r8_...`) |
| `DEDALUS_API_KEY` | Yes | Dedalus platform API key |
| `DEDALUS_API_URL` | No | Dedalus API base URL |
| `DEDALUS_AS_URL` | No | Dedalus auth server URL (default: `https://as.dedaluslabs.ai`) |

## Quick Start

```bash
# Install dependencies
uv sync

# Set environment variables
cp .env.example .env
# Edit .env with your keys

# Run the server
uv run python src/main.py
```

## Testing

```bash
# Unit tests (no API key needed)
uv run pytest tests/test_tools.py

# Live connection tests (requires REPLICATE_API_TOKEN)
uv run pytest tests/test_connection_live.py
```

## Source Decision

**Decision: Build native (Python)**

No official Replicate MCP server exists. Community implementations are narrow (image-generation only). We build natively using `dedalus-mcp` to cover the full Replicate API.

## License

MIT

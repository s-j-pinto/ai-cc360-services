# Caregiver Insight API

FastAPI middleware that accepts caregiver screening data, sends a structured prompt to a local Ollama instance, and returns the generated hiring insight.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The default upstream is `http://192.168.86.250:11434`. Change `OLLAMA_BASE_URL` in `.env` when Ollama is hosted elsewhere. Ensure the model is available with `ollama pull qwen3.5:4b`.

## Endpoint

`POST /api/caregiver-insight`

Send the caregiver payload described in the request. The response is:

```json
{"aiGeneratedInsight":"..."}
```

Run tests with `pytest`.

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

The default upstream is `http://127.0.0.1:8080` for Ollama running locally with `OLLAMA_HOST=0.0.0.0:8080`. Change `OLLAMA_BASE_URL` in `.env` when Ollama is hosted elsewhere. The weekly summary request uses `OLLAMA_NUM_CTX` (default `8192`) to set Ollama's context window; increase it only if the machine has enough memory. Ensure the model is available with `ollama pull gemma4:e2b`.

Ollama generation usage (prompt/output token counts, completion reason, duration, and the complete generated response) is written to `fastapi.out.log`. Generated responses may contain sensitive client health information; restrict access to this file and remove it when it is no longer needed. Request prompts and source visit notes are not logged.

## Endpoint

`POST /api/caregiver-insight`

Send the caregiver payload described in the request. The response is:

```json
{"aiGeneratedInsight":"..."}
```

Run tests with `pytest`.

### Weekly notes summary

`POST /api/weekly-notes-summary` accepts the upstream request envelope. `visitNotes` is an array of objects containing `date`, `userName`, and `notesMessage`. `reportDate` is optional:

```bash
curl -X POST http://localhost:8000/api/weekly-notes-summary \
	-H "Content-Type: application/json" \
	-d '{
		"clientName": "Gunningham, John",
		"clientId": "204190",
		"visitNotes": [
			{
				"date": "09/17/2026 03:08 PM",
				"userName": "Pulido, Cristal",
				"notesMessage": "Visit 09/17/2026 07:00 AM: Client attended a community outing and Mass."
			}
		]
	}'
```

The endpoint returns the validated JSON summary object, including evidence-based weekly predictive health analytics for cognitive/behavioral, potential UTI-related, fall/mobility, and nutrition/hydration indicators. It does not diagnose conditions; indicators must be documented in the notes. Output is kept concise (approximately 2,000 tokens, with a default generation limit of 2,200). When provided, `reportDate` must use `YYYY-MM-DD` format. Legacy plain-text `visitNotes` input remains accepted.

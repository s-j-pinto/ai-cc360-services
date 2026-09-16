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

The default upstream is `http://127.0.0.1:8080` for Ollama running locally with `OLLAMA_HOST=0.0.0.0:8080`. Change `OLLAMA_BASE_URL` in `.env` when Ollama is hosted elsewhere. Ensure the model is available with `ollama pull qwen3.5:4b`.

## Endpoint

`POST /api/caregiver-insight`

Send the caregiver payload described in the request. The response is:

```json
{"aiGeneratedInsight":"..."}
```

Run tests with `pytest`.

### Weekly notes summary

`POST /api/weekly-notes-summary` with a JSON body containing `visitNotes` and `reportDate`:

```bash
curl -X POST http://localhost:8000/api/weekly-notes-summary \
	-H "Content-Type: application/json" \
	-d '{
		"reportDate": "2026-09-15",
		"visitNotes": "09/14/2026 11:55 AM\nEssel-Baidoo, Stella\nVisit 09/14/2026 08:00 AM: Dennis was up when I got there, lay his bed and cleaned his room and bathroom, swept and mopped the floors, took out trash, and took him for a neighborhood walk.\n\n09/10/2026 12:31 PM\nTapia Huerta, Teresa\nVisit 09/10/2026 09:00 AM: I purchased ingredients, cooked pasta with chicken, served Mr. Dennis, cleaned up, and finished early. Lisa relieved my shift early."
	}'
```

The endpoint returns the validated JSON summary object. `reportDate` must use `YYYY-MM-DD` format.

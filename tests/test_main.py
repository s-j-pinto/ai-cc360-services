import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import WeeklyNotesRequest, app, build_weekly_notes_prompt, get_ollama_client


class FakeOllamaClient:
    def __init__(self, response: httpx.Response):
        self.response = response
        self.request: httpx.Request | None = None
        self.closed = False

    async def post(self, url: str, **kwargs: object) -> httpx.Response:
        self.request = httpx.Request("POST", url, json=kwargs["json"])
        self.response.request = self.request
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def candidate_payload() -> dict[str, object]:
    return {
        "fullName": "Monica Selles",
        "yearsExperience": 0,
        "summary": "Not Experienced other than taking care of Mom for 6 months",
        "canUseHoyerLift": False,
        "hasDementiaExperience": False,
        "hasHospiceExperience": False,
        "hha": True,
        "hca": False,
        "availability": {"monday": ["Morning"]},
        "hasCar": "no",
        "validLicense": "yes",
        "interviewNotes": "Candidate seemed unpolite and not very punctual. Bad communication.",
        "candidateRating": "F",
    }


def weekly_notes_payload() -> dict[str, str]:
    return {
        "reportDate": "2026-09-15",
        "visitNotes": """09/14/2026 11:55 AM
Essel-Baidoo, Stella
Visit 09/14/2026 08:00 AM: Dennis was up when I got there, lay his bed and cleaned his room and bathroom, swept and mopped the floors, took out trash, and took him for a neighborhood walk.

09/10/2026 12:31 PM
Tapia Huerta, Teresa
Visit 09/10/2026 09:00 AM: Teresa Tapia Dennis Williams 9/10/26 9:00 AM-1:00 PM. I purchased ingredients, cooked pasta with chicken, served Mr. Dennis, cleaned up, and finished early. Lisa relieved my shift early.""",
    }


def test_builds_prompt_from_upstream_weekly_notes_envelope() -> None:
    request = WeeklyNotesRequest(
        clientName="Gunningham, John",
        clientId="204190",
        visitNotes=[
            {
                "date": "09/17/2026 03:08 PM",
                "userName": "Pulido, Cristal",
                "notesMessage": "Client attended a community outing and Mass.",
            }
        ],
    )

    prompt = build_weekly_notes_prompt(request)

    assert "Client name: Gunningham, John" in prompt
    assert "Client ID: 204190" in prompt
    assert '"userName": "Pulido, Cristal"' in prompt
    assert "Client attended a community outing and Mass." in prompt


@pytest.mark.asyncio
async def test_generates_insight_and_forwards_prompt() -> None:
    fake_client = FakeOllamaClient(httpx.Response(200, json={"response": "Summary\n\nRecommendation: Do not recommend."}))
    app.dependency_overrides[get_ollama_client] = lambda: fake_client
    try:
        with TestClient(app) as test_client:
            response = test_client.post("/api/caregiver-insight", json=candidate_payload())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"aiGeneratedInsight": "Summary\n\nRecommendation: Do not recommend."}
    assert fake_client.request is not None
    body = json.loads(fake_client.request.read())
    assert body["model"] == "qwen3.5:4b"
    assert "Monica Selles" in body["prompt"]
    assert body["stream"] is False
    assert fake_client.closed is True


def test_returns_bad_gateway_when_ollama_fails() -> None:
    fake_client = FakeOllamaClient(httpx.Response(503, json={"error": "unavailable"}))
    app.dependency_overrides[get_ollama_client] = lambda: fake_client
    try:
        with TestClient(app) as test_client:
            response = test_client.post("/api/caregiver-insight", json=candidate_payload())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {"detail": "Ollama request failed"}


def test_generates_valid_weekly_notes_summary() -> None:
    summary = {
        "clientName": "Dennis",
        "reportPeriod": {"startDate": "2026-09-10", "endDate": "2026-09-14"},
        "summary": {"overview": "Documented household and meal support.", "totalVisitsDocumented": 2, "primaryCareNeeds": ["Household support"]},
        "visitSummary": [],
        "careActivities": {"personalCare": [], "householdMaintenance": [], "nutrition": [], "mobilityAndCommunityEngagement": [], "otherActivities": []},
        "healthAndSafetyObservations": {"documentedPositiveObservations": [], "medicalConcernsReported": [], "fallsOrIncidentsReported": [], "medicationUpdatesReported": [], "changesInConditionReported": [], "notDocumented": ["Mood"]},
        "careConcernsAndFollowUp": [],
        "agencyOwnerReview": {"overallStatus": "Review documentation", "completedCareHighlights": [], "documentationGaps": [], "itemsToMonitor": [], "recommendedFollowUp": []},
        "documentationMetadata": {"sourceNoteCount": 2, "clientReadStatus": "Not documented", "reportGeneratedDate": "2026-09-15", "dataLimitations": []},
    }
    fake_client = FakeOllamaClient(httpx.Response(200, json={"response": json.dumps(summary)}))
    app.dependency_overrides[get_ollama_client] = lambda: fake_client
    try:
        with TestClient(app) as test_client:
            response = test_client.post("/api/weekly-notes-summary", json=weekly_notes_payload())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == summary
    assert fake_client.request is not None
    body = json.loads(fake_client.request.read())
    assert body["model"] == "qwen3.5:4b"
    assert "Dennis" in body["prompt"]
    assert "2026-09-15" in body["prompt"]
    assert body["stream"] is False
    assert body["format"]["type"] == "object"
    assert "careActivities" in body["format"]["properties"]
    assert body["options"]["num_predict"] == 16384
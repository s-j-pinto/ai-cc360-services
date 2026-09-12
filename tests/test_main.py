import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_ollama_client


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
from typing import Annotated, Any, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:8080"
    ollama_model: str = "qwen3.5:4b"
    ollama_timeout_seconds: float = 600.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
app = FastAPI(title="Caregiver Insight API", version="1.0.0")


class Availability(BaseModel):
    model_config = {"extra": "allow"}


class CaregiverProfile(BaseModel):
    fullName: str = Field(min_length=1)
    yearsExperience: float = Field(ge=0)
    summary: str | None = None
    canUseHoyerLift: bool = False
    hasDementiaExperience: bool = False
    hasHospiceExperience: bool = False
    hha: bool = False
    hca: bool = False
    availability: dict[str, list[str]] = Field(default_factory=dict)
    hasCar: str
    validLicense: str
    interviewNotes: str
    candidateRating: Literal["A", "B", "C", "D", "F"]


class InsightResponse(BaseModel):
    aiGeneratedInsight: str


def get_ollama_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=settings.ollama_timeout_seconds)


Client = Annotated[httpx.AsyncClient, Depends(get_ollama_client)]


def build_prompt(candidate: CaregiverProfile) -> str:
    summary = candidate.summary or "Not provided"
    return f"""You are an expert HR assistant for a home care agency. Your task is to analyze a caregiver candidate's profile and the notes from their phone screen to provide a single, combined insight containing a summary and a hiring recommendation.

Analyze the following information:

**Caregiver Profile:**
- Full Name: {candidate.fullName}
- Years of Experience: {candidate.yearsExperience}
- Experience Summary: {summary}
- Skills:
  - Hoyer Lift: {'Yes' if candidate.canUseHoyerLift else 'No'}
  - Dementia Experience: {'Yes' if candidate.hasDementiaExperience else 'No'}
  - Hospice Experience: {'Yes' if candidate.hasHospiceExperience else 'No'}
- Certifications:
  - HHA: {'Yes' if candidate.hha else 'No'}
  - HCA: {'Yes' if candidate.hca else 'No'}
- Transportation: Has car: {candidate.hasCar}, Valid License: {candidate.validLicense}

**Interviewer's Phone Screen Feedback:**
- Rating: {candidate.candidateRating} (A = Excellent, B = Good, C = Average, D = Below Average, F = Not Recommended)
- Notes:
{candidate.interviewNotes}

**Your Task:**

Generate a single string for the 'aiGeneratedInsight' field. This string must contain two parts:

1. **Summary:** First, write a concise professional summary of the candidate. Highlight their key strengths, potential weaknesses, and alignment with a caregiver role. The summary must be a maximum of 150 words.

2. **Recommendation:** After the summary, add two new lines (to create a blank line), then provide a clear, actionable hiring recommendation. Start this part with "Recommendation:". Choose from "Recommend for in-person interview," "Proceed with caution," or "Do not recommend." Justify your choice with 1-2 key reasons based on the provided data.

Example format:
[Summary of the candidate...]

Recommendation: [Your recommendation and justification...]"""


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/caregiver-insight", response_model=InsightResponse)
async def caregiver_insight(candidate: CaregiverProfile, client: Client) -> InsightResponse:
    request_body = {
        "model": settings.ollama_model,
        "prompt": build_prompt(candidate),
        "stream": False,
        "think": False,
    }
    try:
        response = await client.post(f"{settings.ollama_base_url}/api/generate", json=request_body)
        response.raise_for_status()
        ollama_result: dict[str, Any] = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise HTTPException(status_code=502, detail="Ollama request failed") from error
    finally:
        await client.aclose()

    insight = ollama_result.get("response")
    if not isinstance(insight, str) or not insight.strip():
        raise HTTPException(status_code=502, detail="Ollama returned an invalid response")
    return InsightResponse(aiGeneratedInsight=insight.strip())
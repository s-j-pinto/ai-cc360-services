import json
from typing import Annotated, Any, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:8080"
    ollama_model: str = "qwen3.5:4b"
    ollama_timeout_seconds: float = 300.0
    ollama_num_predict: int = 16384
    cors_allowed_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
app = FastAPI(title="CareConnect AI Services API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.cors_allowed_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class VisitNote(BaseModel):
    date: str = Field(min_length=1)
    userName: str = Field(min_length=1)
    notesMessage: str = Field(min_length=1)


class WeeklyNotesRequest(BaseModel):
    clientName: str | None = None
    clientId: str | None = None
    visitNotes: list[VisitNote] | str
    reportDate: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class ReportPeriod(BaseModel):
    startDate: str
    endDate: str


class Summary(BaseModel):
    overview: str
    totalVisitsDocumented: int = Field(ge=0)
    primaryCareNeeds: list[str]


class VisitSummary(BaseModel):
    date: str
    caregiver: str
    scheduledStartTime: str | None = None
    scheduledEndTime: str | None = None
    activitiesCompleted: list[str]
    clientObservations: list[str]
    shiftNotes: str
    concernsReported: list[str]
    followUpRequired: bool


class CareActivities(BaseModel):
    personalCare: list[str]
    householdMaintenance: list[str]
    nutrition: list[str]
    mobilityAndCommunityEngagement: list[str]
    otherActivities: list[str]


class HealthAndSafetyObservations(BaseModel):
    documentedPositiveObservations: list[str]
    medicalConcernsReported: list[str]
    fallsOrIncidentsReported: list[str]
    medicationUpdatesReported: list[str]
    changesInConditionReported: list[str]
    notDocumented: list[str]


class CareConcernAndFollowUp(BaseModel):
    concern: str
    sourceDate: str
    description: str
    severity: Literal["none", "low", "medium", "high", "unknown"]
    recommendedAction: str
    requiresOwnerReview: bool
    requiresClinicalReview: bool


class AgencyOwnerReview(BaseModel):
    overallStatus: str
    completedCareHighlights: list[str]
    documentationGaps: list[str]
    itemsToMonitor: list[str]
    recommendedFollowUp: list[str]


class DocumentationMetadata(BaseModel):
    sourceNoteCount: int = Field(ge=0)
    clientReadStatus: str
    reportGeneratedDate: str
    dataLimitations: list[str]


class WeeklyNotesSummary(BaseModel):
    clientName: str
    reportPeriod: ReportPeriod
    summary: Summary
    visitSummary: list[VisitSummary]
    careActivities: CareActivities
    healthAndSafetyObservations: HealthAndSafetyObservations
    careConcernsAndFollowUp: list[CareConcernAndFollowUp]
    agencyOwnerReview: AgencyOwnerReview
    documentationMetadata: DocumentationMetadata


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


def build_weekly_notes_prompt(request: WeeklyNotesRequest) -> str:
    source_notes = (
        request.visitNotes
        if isinstance(request.visitNotes, str)
        else json.dumps([note.model_dump() for note in request.visitNotes], indent=2)
    )
    return f"""You are an experienced senior care administrator and quality assurance reviewer for a home care agency.
Your task is to analyze caregiver visit notes for a single client and generate a clear, accurate, and professional summary that can be shared with the agency owner.

Input
You will receive multiple caregiver visit notes containing some or all of the following: visit date and time, caregiver name, client name, caregiver observations, activities completed, personal care provided, meal preparation and nutrition, household tasks, mobility and community activities, incidents, concerns, changes in condition, shift changes, or early departures.

Instructions:
1. Identify the reporting period from the supplied notes.
2. Summarize the client's care activities across all visits.
3. Organize each visit by date, caregiver, and documented activities.
4. Identify documented observations about the client's condition, mood, mobility, appetite, hygiene, and participation.
5. Identify explicitly documented care concerns, safety issues, incidents, missed tasks, or changes in condition.
6. Identify follow-up actions supported by the notes.
7. Highlight incomplete documentation, such as missing visit end times or unclear shift changes.
8. Do not invent facts, diagnoses, medications, symptoms, care needs, or incidents.
9. Do not infer a medical condition or conclude that the client is safe or healthy solely because no concern was documented.
10. Distinguish documented facts, potential follow-up, and not documented information.
11. Preserve unclear notes as closely as possible and flag them for review.
12. If a shift ended early, identify the early departure and documented reason or replacement caregiver. Do not assume care was fully covered.
13. Use reporting dates in YYYY-MM-DD format.
14. Return valid JSON only. Do not include Markdown, explanations, or text outside the JSON object.
15. Keep the JSON concise enough to finish generation: use short phrases, no more than 3 items in each list when possible, and no more than 2 sentences per narrative field. Do not repeat the source notes verbatim.

Rules for care concerns and severity:
- Use "none" when no concern is documented and no follow-up is indicated.
- Use "low" for minor documentation gaps or routine administrative clarification.
- Use "medium" for a documented issue that warrants timely agency-owner review.
- Use "high" only for a documented urgent safety or care concern requiring immediate escalation.
- Use "unknown" when the seriousness of a documented issue cannot be determined.
- Never assign severity based solely on a caregiver's name, task, or absence of information.
- Do not diagnose or recommend medical treatment.
- An empty array means no such issue was documented; it does not mean the issue was definitively absent.

Return an object with exactly these top-level keys and matching value types:
clientName, reportPeriod, summary, visitSummary, careActivities, healthAndSafetyObservations, careConcernsAndFollowUp, agencyOwnerReview, documentationMetadata.

Caregiver visit notes:
Client name: {request.clientName or "Not provided"}
Client ID: {request.clientId or "Not provided"}
{source_notes}

Report generation date:
{request.reportDate or "Not provided"}"""


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


@app.post("/api/weekly-notes-summary", response_model=WeeklyNotesSummary)
async def weekly_notes_summary(request: WeeklyNotesRequest, client: Client) -> WeeklyNotesSummary:
    request_body = {
        "model": settings.ollama_model,
        "prompt": build_weekly_notes_prompt(request),
        "stream": False,
        "think": False,
        "format": WeeklyNotesSummary.model_json_schema(),
        "options": {"num_predict": settings.ollama_num_predict},
    }
    try:
        response = await client.post(f"{settings.ollama_base_url}/api/generate", json=request_body)
        response.raise_for_status()
        ollama_result: dict[str, Any] = response.json()
        raw_summary = ollama_result.get("response")
        if not isinstance(raw_summary, str):
            raise ValueError("Ollama response is not a JSON string")
        return WeeklyNotesSummary.model_validate_json(raw_summary)
    except httpx.TimeoutException as error:
        print(f"Weekly notes summary timed out: {error.__class__.__name__}: {error!r}")
        raise HTTPException(
            status_code=504,
            detail="Ollama timed out while generating the weekly summary",
        ) from error
    except (httpx.HTTPError, ValueError) as error:
        print(f"Weekly notes summary validation failed: {error.__class__.__name__}: {error!r}")
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an invalid weekly summary: {error!r}",
        ) from error
    finally:
        await client.aclose()
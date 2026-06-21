from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class GenderEnum(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"


class GenderPrefEnum(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    ANY = "ANY"


class SmokingStatusEnum(str, Enum):
    NON_SMOKER = "NON_SMOKER"
    SOMETIMES = "SOMETIMES"
    SMOKER = "SMOKER"


class SmokingPrefEnum(str, Enum):
    NOT_ALLOWED = "NOT_ALLOWED"
    ALLOWED = "ALLOWED"


class PetStatusEnum(str, Enum):
    NO_PETS = "NO_PETS"
    HAS_PETS = "HAS_PETS"


class SleepScheduleEnum(str, Enum):
    EARLY_BIRD = "EARLY_BIRD"
    FLEXIBLE = "FLEXIBLE"
    NIGHT_OWL = "NIGHT_OWL"


class ProfileData(BaseModel):
    user_id: int
    age: int = Field(ge=18, le=65)
    gender: GenderEnum
    occupation: Optional[str] = None
    smoking_status: SmokingStatusEnum = SmokingStatusEnum.NON_SMOKER
    pet_status: PetStatusEnum = PetStatusEnum.NO_PETS
    sleep_schedule: SleepScheduleEnum = SleepScheduleEnum.FLEXIBLE
    cleanliness: int = Field(ge=1, le=5, default=3)
    budget_min: int = Field(ge=0, default=500)
    budget_max: int = Field(ge=0, default=5000)
    preferred_locations: list[str] = Field(default_factory=list)
    roommate_gender_pref: GenderPrefEnum = GenderPrefEnum.ANY
    pref_smoking: SmokingPrefEnum = SmokingPrefEnum.NOT_ALLOWED
    personality_traits: list[str] = Field(default_factory=list)


class PriceEstimateRequest(BaseModel):
    age: int = Field(ge=18, le=65)
    gender: GenderEnum
    occupation: Optional[str] = None
    smoking_status: SmokingStatusEnum = SmokingStatusEnum.NON_SMOKER
    pet_status: PetStatusEnum = PetStatusEnum.NO_PETS
    sleep_schedule: SleepScheduleEnum = SleepScheduleEnum.FLEXIBLE
    cleanliness: int = Field(ge=1, le=5, default=3)
    budget_min: int = Field(ge=0, default=500)
    roommate_gender_pref: GenderPrefEnum = GenderPrefEnum.ANY
    pref_smoking: SmokingPrefEnum = SmokingPrefEnum.NOT_ALLOWED


class PriceEstimateResponse(BaseModel):
    estimated_max_budget: int
    confidence: str
    factors: dict[str, float]


class MatchScoreRequest(BaseModel):
    seeker: ProfileData
    candidate: ProfileData


class MatchScoreResult(BaseModel):
    user_id: int
    score: float = Field(ge=0, le=100)
    breakdown: dict[str, float]
    strengths: list[str]
    conflicts: list[str]


class MatchScoreResponse(BaseModel):
    user_id: int
    score: float
    breakdown: dict[str, float]
    strengths: list[str]
    conflicts: list[str]


class MatchRequest(BaseModel):
    seeker: ProfileData
    candidates: list[ProfileData]
    top_n: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=30.0, ge=0, le=100)


class MatchResponse(BaseModel):
    matches: list[MatchScoreResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

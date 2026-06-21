import numpy as np
from schemas import PriceEstimateRequest, PriceEstimateResponse

GENDER_MAP = {"MALE": 1, "FEMALE": 0}
SMOKING_MAP = {"NON_SMOKER": 0, "SOMETIMES": 0, "SMOKER": 1}
PET_MAP = {"HAS_PETS": 0, "NO_PETS": 1}
SLEEP_MAP = {"EARLY_BIRD": 0, "FLEXIBLE": 1, "NIGHT_OWL": 1}
ROOMMATE_GENDER_MAP = {"ANY": 0, "FEMALE": 1, "MALE": 2}
PREF_SMOKING_MAP = {"ALLOWED": 0, "NOT_ALLOWED": 1}

OCCUPATION_MAP = {
    "designer": 0,
    "doctor": 1,
    "engineer": 2,
    "freelancer": 3,
    "medical student": 4,
    "software engineer": 5,
    "student": 6,
    "teacher": 7,
}

FEATURE_ORDER = [
    "age", "gender", "occupation", "smoking_status", "pet_status",
    "sleep_schedule", "cleanliness", "budget_min",
    "roommate_gender_pref", "pref_smoking",
]


def predict_price(model, request: PriceEstimateRequest) -> PriceEstimateResponse:
    occupation_code = OCCUPATION_MAP.get(
        (request.occupation or "student").lower().strip(), 6
    )

    features = np.array([[
        request.age,
        GENDER_MAP[request.gender.value],
        occupation_code,
        SMOKING_MAP[request.smoking_status.value],
        PET_MAP[request.pet_status.value],
        SLEEP_MAP[request.sleep_schedule.value],
        request.cleanliness,
        request.budget_min,
        ROOMMATE_GENDER_MAP[request.roommate_gender_pref.value],
        PREF_SMOKING_MAP[request.pref_smoking.value],
    ]])

    prediction = model.predict(features)[0]
    estimated_max = int(round(prediction))

    budget_ratio = estimated_max / max(request.budget_min, 1)
    if 0.8 <= budget_ratio <= 2.5:
        confidence = "high"
    elif 0.5 <= budget_ratio <= 4.0:
        confidence = "medium"
    else:
        confidence = "low"

    factors = {
        "age_influence": round(float(request.age) / 35 * 10, 1),
        "budget_base": float(request.budget_min),
        "occupation_tier": float(occupation_code),
        "cleanliness_factor": float(request.cleanliness) / 5 * 10,
    }

    return PriceEstimateResponse(
        estimated_max_budget=estimated_max,
        confidence=confidence,
        factors=factors,
    )

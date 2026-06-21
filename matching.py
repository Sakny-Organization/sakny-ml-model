from schemas import ProfileData, MatchScoreResult

WEIGHTS = {
    "gender_compat": 0.20,
    "budget_overlap": 0.20,
    "smoking_compat": 0.15,
    "location_overlap": 0.15,
    "sleep_compat": 0.10,
    "cleanliness": 0.10,
    "pet_compat": 0.05,
    "personality": 0.05,
}

SLEEP_ORDER = {"EARLY_BIRD": 0, "FLEXIBLE": 1, "NIGHT_OWL": 2}

FACTOR_LABELS = {
    "gender_compat": "Gender Compatibility",
    "budget_overlap": "Budget Overlap",
    "smoking_compat": "Smoking Compatibility",
    "location_overlap": "Location Overlap",
    "sleep_compat": "Sleep Schedule",
    "cleanliness": "Cleanliness Level",
    "pet_compat": "Pet Compatibility",
    "personality": "Personality Match",
}


def _gender_compat(a: ProfileData, b: ProfileData) -> float:
    a_ok = (a.roommate_gender_pref.value == "ANY") or (a.roommate_gender_pref.value == b.gender.value)
    b_ok = (b.roommate_gender_pref.value == "ANY") or (b.roommate_gender_pref.value == a.gender.value)
    return 1.0 if (a_ok and b_ok) else 0.0


def _budget_overlap(a: ProfileData, b: ProfileData) -> float:
    lo = max(a.budget_min, b.budget_min)
    hi = min(a.budget_max, b.budget_max)
    if hi < lo:
        return 0.0
    intersection = hi - lo
    union = max(a.budget_max, b.budget_max) - min(a.budget_min, b.budget_min)
    return intersection / union if union > 0 else 0.0


def _smoking_compat(a: ProfileData, b: ProfileData) -> float:
    a_smoker = a.smoking_status.value == "SMOKER"
    b_smoker = b.smoking_status.value == "SMOKER"
    a_ok_with_smoke = a.pref_smoking.value == "ALLOWED"
    b_ok_with_smoke = b.pref_smoking.value == "ALLOWED"

    if a_smoker and not b_ok_with_smoke:
        return 0.0
    if b_smoker and not a_ok_with_smoke:
        return 0.0
    if a.smoking_status == b.smoking_status:
        return 1.0
    if a.smoking_status.value == "SOMETIMES" or b.smoking_status.value == "SOMETIMES":
        return 0.6
    return 1.0


def _location_overlap(a: ProfileData, b: ProfileData) -> float:
    sa = set(loc.lower().strip() for loc in a.preferred_locations)
    sb = set(loc.lower().strip() for loc in b.preferred_locations)
    if not sa or not sb:
        return 0.5
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union > 0 else 0.0


def _sleep_compat(a: ProfileData, b: ProfileData) -> float:
    if a.sleep_schedule.value == "FLEXIBLE" or b.sleep_schedule.value == "FLEXIBLE":
        return 0.8
    diff = abs(SLEEP_ORDER[a.sleep_schedule.value] - SLEEP_ORDER[b.sleep_schedule.value])
    return [1.0, 0.6, 0.2][diff]


def _cleanliness_compat(a: ProfileData, b: ProfileData) -> float:
    diff = abs(a.cleanliness - b.cleanliness)
    return 1.0 - (diff / 4.0)


def _pet_compat(a: ProfileData, b: ProfileData) -> float:
    if a.pet_status == b.pet_status:
        return 1.0 if a.pet_status.value == "NO_PETS" else 0.9
    return 0.4


def _personality_overlap(a: ProfileData, b: ProfileData) -> float:
    sa = set(t.lower().strip() for t in a.personality_traits)
    sb = set(t.lower().strip() for t in b.personality_traits)
    if not sa or not sb:
        return 0.5
    union = len(sa | sb)
    return len(sa & sb) / union if union > 0 else 0.0


def _generate_insights(breakdown: dict[str, float]) -> tuple[list[str], list[str]]:
    strengths = []
    conflicts = []
    for factor, score in breakdown.items():
        label = FACTOR_LABELS.get(factor, factor)
        if score >= 80:
            strengths.append(f"Great {label.lower()}")
        elif score <= 30:
            conflicts.append(f"Low {label.lower()}")
    return strengths, conflicts


def compute_match_score(seeker: ProfileData, candidate: ProfileData) -> MatchScoreResult:
    raw_breakdown = {
        "gender_compat": _gender_compat(seeker, candidate),
        "budget_overlap": _budget_overlap(seeker, candidate),
        "smoking_compat": _smoking_compat(seeker, candidate),
        "location_overlap": _location_overlap(seeker, candidate),
        "sleep_compat": _sleep_compat(seeker, candidate),
        "cleanliness": _cleanliness_compat(seeker, candidate),
        "pet_compat": _pet_compat(seeker, candidate),
        "personality": _personality_overlap(seeker, candidate),
    }

    if raw_breakdown["gender_compat"] == 0.0:
        total = 0.0
    else:
        total = sum(raw_breakdown[k] * WEIGHTS[k] for k in WEIGHTS)

    score = round(total * 100, 1)
    breakdown = {k: round(v * 100, 1) for k, v in raw_breakdown.items()}
    strengths, conflicts = _generate_insights(breakdown)

    return MatchScoreResult(
        user_id=candidate.user_id,
        score=score,
        breakdown=breakdown,
        strengths=strengths,
        conflicts=conflicts,
    )


def recommend_matches(
    seeker: ProfileData,
    candidates: list[ProfileData],
    top_n: int = 10,
    min_score: float = 30.0,
) -> list[MatchScoreResult]:
    results = []
    for candidate in candidates:
        if candidate.user_id == seeker.user_id:
            continue
        result = compute_match_score(seeker, candidate)
        if result.score >= min_score:
            results.append(result)
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────
#  1. DATA MODELS  (matches onboarding form)
# ─────────────────────────────────────────────

Gender          = Literal["MALE", "FEMALE", "PREFER_NOT_TO_SAY"]
GenderPref      = Literal["MALE", "FEMALE", "ANY"]
SmokingStatus   = Literal["NON_SMOKER", "SOMETIMES", "SMOKER"]
SmokingPref     = Literal["NOT_ALLOWED", "ALLOWED"]
PetStatus       = Literal["NO_PETS", "HAS_PETS"]
SleepSchedule   = Literal["EARLY_BIRD", "FLEXIBLE", "NIGHT_OWL"]
Occupation      = Literal["Student", "Working professional", "Both / Other"]
Personality     = Literal["Calm","Social","Introvert","Extrovert","Organized",
                           "Spontaneous","Homebody","Often out","Talkative","Quiet"]
Location        = Literal["Nasr city","New cairo","Maadi","Zamalek","Dokki",
                           "Mohandessin","Heliopolis","Sheikh zayed","6th october","Alexandria"]
AdditionalPref  = Literal["Non-smoker","Quiet","Clean","Student","Works from home"]


@dataclass
class RoommateProfile:
    # ── Step 1: Basics ──────────────────────────────────────
    user_id:        int
    age:            int                         # numeric input
    gender:         Gender                      # radio: Male / Female / Prefer not to say
    occupation:     str                         # free text (e.g. "Software Engineer")
    occupation_type: Occupation                 # pill: Student / Working professional / Both

    # ── Step 2: Personality ──────────────────────────────────
    personality_traits: List[Personality]       # multi-select pills (Calm, Social, …)

    # ── Step 3: Lifestyle ───────────────────────────────────
    smoking_status: SmokingStatus               # radio: Non-smoker / Sometimes / Smokes often
    pet_status:     PetStatus                   # radio: No pets / Cats / Dogs → map to NO_PETS / HAS_PETS
    sleep_schedule: SleepSchedule               # radio: Early sleeper / Flexible / Night owl
    cleanliness:    int                         # slider 1-5  (1=Messy, 5=Very clean)

    # ── Step 4: Budget ──────────────────────────────────────
    budget_min:     int                         # EGP  (left handle of range slider)
    budget_max:     int                         # EGP  (right handle of range slider)

    # ── Step 5: Preferred locations ─────────────────────────
    preferred_locations: List[Location]         # multi-select pills

    # ── Step 6: Roommate preferences ────────────────────────
    roommate_gender_pref: GenderPref            # pill: Same gender only / Open to any gender
    pref_smoking:   SmokingPref                 # derived from additional_prefs "Non-smoker"
    additional_prefs: List[AdditionalPref]      # multi-select pills

    # ── Optional / enrichment ───────────────────────────────
    current_city:   str = ""                    # free text from Step 1


@dataclass
class MatchResult:
    user_id:        int
    match_score:    float           # 0.0 – 100.0
    score_breakdown: dict           # detailed sub-scores for debugging / UI tooltip
    profile:        RoommateProfile


# ─────────────────────────────────────────────
#  2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Weights — total must equal 1.0
WEIGHTS = {
    "gender_compat":    0.20,   # hard filter + soft match
    "budget_overlap":   0.20,   # range intersection ratio
    "smoking_compat":   0.15,   # preference cross-check
    "location_overlap": 0.15,   # shared preferred areas
    "sleep_compat":     0.10,   # schedule similarity
    "cleanliness":      0.10,   # absolute difference
    "pet_compat":       0.05,   # pet tolerance
    "personality":      0.05,   # shared trait count
}

SLEEP_ORDER = {"EARLY_BIRD": 0, "FLEXIBLE": 1, "NIGHT_OWL": 2}


def _gender_compat(a: RoommateProfile, b: RoommateProfile) -> float:
    """
    Returns 0.0 if hard gender filter fails (both sides want same-gender only
    and genders don't match), else 1.0.
    """
    def wants_same(p: RoommateProfile) -> bool:
        return p.roommate_gender_pref == "MALE" or p.roommate_gender_pref == "FEMALE"

    a_ok = (a.roommate_gender_pref == "ANY") or (a.roommate_gender_pref == b.gender)
    b_ok = (b.roommate_gender_pref == "ANY") or (b.roommate_gender_pref == a.gender)
    return 1.0 if (a_ok and b_ok) else 0.0


def _budget_overlap(a: RoommateProfile, b: RoommateProfile) -> float:
    """Intersection-over-union of budget ranges, clamped [0, 1]."""
    lo = max(a.budget_min, b.budget_min)
    hi = min(a.budget_max, b.budget_max)
    if hi < lo:
        return 0.0
    intersection = hi - lo
    union = max(a.budget_max, b.budget_max) - min(a.budget_min, b.budget_min)
    return intersection / union if union > 0 else 0.0


def _smoking_compat(a: RoommateProfile, b: RoommateProfile) -> float:
    """
    Full score if both preferences are compatible; partial if one is flexible.
    """
    a_smoker = a.smoking_status == "SMOKER"
    b_smoker = b.smoking_status == "SMOKER"

    a_ok_with_smoke = a.pref_smoking == "ALLOWED"
    b_ok_with_smoke = b.pref_smoking == "ALLOWED"

    if a_smoker and not b_ok_with_smoke:
        return 0.0
    if b_smoker and not a_ok_with_smoke:
        return 0.0
    if a.smoking_status == b.smoking_status:
        return 1.0
    if a.smoking_status == "SOMETIMES" or b.smoking_status == "SOMETIMES":
        return 0.6
    return 1.0


def _location_overlap(a: RoommateProfile, b: RoommateProfile) -> float:
     
    sa, sb = set(a.preferred_locations), set(b.preferred_locations)
    if not sa or not sb:
        return 0.5    # unknown → neutral
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union


def _sleep_compat(a: RoommateProfile, b: RoommateProfile) -> float:
     
    if a.sleep_schedule == "FLEXIBLE" or b.sleep_schedule == "FLEXIBLE":
        return 0.8
    diff = abs(SLEEP_ORDER[a.sleep_schedule] - SLEEP_ORDER[b.sleep_schedule])
    return [1.0, 0.6, 0.2][diff]


def _cleanliness_compat(a: RoommateProfile, b: RoommateProfile) -> float:
     
    diff = abs(a.cleanliness - b.cleanliness)
    return 1.0 - (diff / 4.0)


def _pet_compat(a: RoommateProfile, b: RoommateProfile) -> float:
   
    if a.pet_status == b.pet_status:
        return 1.0 if a.pet_status == "NO_PETS" else 0.9
    return 0.4


def _personality_overlap(a: RoommateProfile, b: RoommateProfile) -> float:
    
    sa, sb = set(a.personality_traits), set(b.personality_traits)
    if not sa or not sb:
        return 0.5
    return len(sa & sb) / len(sa | sb)


# ─────────────────────────────────────────────
#  3. CORE SCORING FUNCTION
# ─────────────────────────────────────────────

def compute_match_score(
    seeker: RoommateProfile,
    candidate: RoommateProfile,
) -> MatchResult:
    
    breakdown = {
        "gender_compat":    _gender_compat(seeker, candidate),
        "budget_overlap":   _budget_overlap(seeker, candidate),
        "smoking_compat":   _smoking_compat(seeker, candidate),
        "location_overlap": _location_overlap(seeker, candidate),
        "sleep_compat":     _sleep_compat(seeker, candidate),
        "cleanliness":      _cleanliness_compat(seeker, candidate),
        "pet_compat":       _pet_compat(seeker, candidate),
        "personality":      _personality_overlap(seeker, candidate),
    }
 
    if breakdown["gender_compat"] == 0.0:
        total = 0.0
    else:
        total = sum(breakdown[k] * WEIGHTS[k] for k in WEIGHTS)

    return MatchResult(
        user_id=candidate.user_id,
        match_score=round(total * 100, 1),
        score_breakdown={k: round(v * 100, 1) for k, v in breakdown.items()},
        profile=candidate,
    )


# ─────────────────────────────────────────────
#  4. RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def recommend(
    seeker: RoommateProfile,
    all_profiles: List[RoommateProfile],
    top_n: int = 10,
    min_score: float = 40.0,
) -> List[MatchResult]:
     
    results = []
    for candidate in all_profiles:
        if candidate.user_id == seeker.user_id:
            continue  # skip self
        result = compute_match_score(seeker, candidate)
        if result.match_score >= min_score:
            results.append(result)

    results.sort(key=lambda r: r.match_score, reverse=True)
    return results[:top_n]


# ─────────────────────────────────────────────
#  5. CSV LOADER  (for the provided dataset)
# ─────────────────────────────────────────────

def load_profiles_from_csv(csv_path: str) -> List[RoommateProfile]:
     
    df = pd.read_csv(csv_path)
    profiles = []
    for _, row in df.iterrows():
        profile = RoommateProfile(
            user_id=int(row["user_id"]),
            age=int(row["age"]),
            gender=row["gender"],
            occupation=row["occupation"],
            occupation_type="Both / Other",   # not in CSV → default
            personality_traits=[],             # not in CSV → default
            smoking_status=row["smoking_status"],
            pet_status=row["pet_status"],
            sleep_schedule=row["sleep_schedule"],
            cleanliness=int(row["cleanliness"]),
            budget_min=int(row["budget_min"]),
            budget_max=int(row["budget_max"]),
            preferred_locations=[],            # not in CSV → default
            roommate_gender_pref=row["roommate_gender_pref"],
            pref_smoking=row["pref_smoking"],
            additional_prefs=[],               # not in CSV → default
        )
        profiles.append(profile)
    return profiles


# ─────────────────────────────────────────────
#  6. API CONTRACT  (what backend exposes)
# ─────────────────────────────────────────────

def api_recommend(seeker_json: dict, db_profiles: List[RoommateProfile]) -> dict:
     
    seeker = RoommateProfile(**seeker_json)
    matches = recommend(seeker, db_profiles)
    return {
        "matches": [
            {
                "user_id":        m.user_id,
                "match_score":    m.match_score,
                "score_breakdown": m.score_breakdown,
                "profile":        asdict(m.profile),
            }
            for m in matches
        ]
    }


# ─────────────────────────────────────────────
#  7. QUICK DEMO / SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load dataset
    all_profiles = load_profiles_from_csv("user_big2.csv")
    print(f"Loaded {len(all_profiles)} profiles from CSV\n")

    # Simulate a seeker (frontend sends this after completing all 6 steps)
    seeker = RoommateProfile(
        user_id=9999,
        age=24,
        gender="MALE",
        occupation="Software Engineer",
        occupation_type="Working professional",
        personality_traits=["Calm", "Organized", "Quiet"],
        smoking_status="NON_SMOKER",
        pet_status="NO_PETS",
        sleep_schedule="NIGHT_OWL",
        cleanliness=4,
        budget_min=2500,
        budget_max=6000,
        preferred_locations=["Nasr city", "New cairo"],
        roommate_gender_pref="MALE",
        pref_smoking="NOT_ALLOWED",
        additional_prefs=["Non-smoker", "Quiet", "Clean"],
        current_city="Cairo",
    )

    matches = recommend(seeker, all_profiles, top_n=5)

    print("Top 5 Roommate Matches")
    print("=" * 50)
    for rank, m in enumerate(matches, 1):
        p = m.profile
        print(f"\n#{rank}  User {p.user_id} — {m.match_score}% match")
        print(f"     {p.age}y/o {p.gender} | {p.occupation}")
        print(f"     Budget: {p.budget_min}–{p.budget_max} EGP")
        print(f"     Sleep: {p.sleep_schedule} | Smoking: {p.smoking_status}")
        print(f"     Cleanliness: {p.cleanliness}/5 | Pets: {p.pet_status}")
        print(f"     Score breakdown: {m.score_breakdown}")

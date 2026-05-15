import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rent Price Estimator", layout="centered")

st.title("Rent Price Estimator")
st.markdown("Fill in your profile and budget range — we'll predict the fair rent **for every occupation** and show all results.")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("price_estimator_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Make sure `price_estimator_model.pkl` is in the same folder.")
    st.stop()

# ── Encoding maps ─────────────────────────────────────────────────────────────
GENDER_MAP          = {"FEMALE": 0, "MALE": 1}
OCCUPATION_MAP      = {"Designer": 0, "Doctor": 1, "Engineer": 2, "Freelancer": 3,
                       "Medical Student": 4, "Software Engineer": 5, "Student": 6, "Teacher": 7}
SMOKING_MAP         = {"NON_SMOKER": 0, "SMOKER": 1}
PET_MAP             = {"HAS_PETS": 0, "NO_PETS": 1}
SLEEP_MAP           = {"EARLY_BIRD": 0, "NIGHT_OWL": 1}
ROOMMATE_GENDER_MAP = {"ANY": 0, "FEMALE": 1, "MALE": 2}
PREF_SMOKING_MAP    = {"ALLOWED": 0, "NOT_ALLOWED": 1}

# ── Form ──────────────────────────────────────────────────────────────────────
st.subheader("User Profile")
col1, col2 = st.columns(2)

with col1:
    age        = st.slider("Age", 18, 35, 25)
    gender     = st.selectbox("Gender", list(GENDER_MAP.keys()))
    smoking    = st.selectbox("Smoking Status", list(SMOKING_MAP.keys()))
    pets       = st.selectbox("Pet Status", list(PET_MAP.keys()))

with col2:
    sleep           = st.selectbox("Sleep Schedule", list(SLEEP_MAP.keys()))
    cleanliness     = st.slider("Cleanliness Level (1 = messy · 5 = very clean)", 1, 5, 3)
    roommate_gender = st.selectbox("Preferred Roommate Gender", list(ROOMMATE_GENDER_MAP.keys()))
    pref_smoking    = st.selectbox("Smoking Preference for Roommate", list(PREF_SMOKING_MAP.keys()))

st.subheader("Your Budget Range")
bcol1, bcol2 = st.columns(2)
with bcol1:
    budget_min = st.number_input("Minimum Budget (EGP) — min 500",
                                  min_value=500, max_value=10000, value=2000, step=100)
with bcol2:
    budget_max_input = st.number_input("Maximum Budget (EGP)",
                                        min_value=500, max_value=15000, value=4000, step=100)

if budget_max_input <= budget_min:
    st.warning("Maximum budget must be greater than minimum budget.")
    st.stop()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict for All Occupations", use_container_width=True):

    base = {
        "age"                  : age,
        "gender"               : GENDER_MAP[gender],
        "smoking_status"       : SMOKING_MAP[smoking],
        "pet_status"           : PET_MAP[pets],
        "sleep_schedule"       : SLEEP_MAP[sleep],
        "cleanliness"          : cleanliness,
        "budget_min"           : budget_min,
        "roommate_gender_pref" : ROOMMATE_GENDER_MAP[roommate_gender],
        "pref_smoking"         : PREF_SMOKING_MAP[pref_smoking],
    }

    # Exact column order used during model training
    FEATURE_ORDER = ["age", "gender", "occupation", "smoking_status", "pet_status",
                     "sleep_schedule", "cleanliness", "budget_min",
                     "roommate_gender_pref", "pref_smoking"]

    # Build one row per occupation
    rows = []
    for occ_name, occ_code in OCCUPATION_MAP.items():
        row = {**base, "occupation": occ_code}
        rows.append(row)

    df_input = pd.DataFrame(rows)[FEATURE_ORDER]
    predictions = model.predict(df_input)

    # Build results dataframe
    results = pd.DataFrame({
        "Occupation"       : list(OCCUPATION_MAP.keys()),
        "Predicted Max (EGP)" : predictions.round(0).astype(int),
    }).sort_values("Predicted Max (EGP)", ascending=False).reset_index(drop=True)

    results.index += 1  # rank starting from 1

    # Verdict per row
    def verdict(p):
        if p < budget_min:
            return "Below your min"
        elif p <= budget_max_input:
            return "Within your range"
        else:
            return "Above your max"

    results["Verdict"] = results["Predicted Max (EGP)"].apply(verdict)

    # ── Metrics ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Results — All Occupations")

    m1, m2, m3 = st.columns(3)
    m1.metric("Lowest Prediction",  f"{results['Predicted Max (EGP)'].min():,} EGP",
              results.loc[results['Predicted Max (EGP)'].idxmin(), 'Occupation'])
    m2.metric("Highest Prediction", f"{results['Predicted Max (EGP)'].max():,} EGP",
              results.loc[results['Predicted Max (EGP)'].idxmax(), 'Occupation'])
    m3.metric("Average Prediction", f"{results['Predicted Max (EGP)'].mean():,.0f} EGP")

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown("All Predictions")

    def color_verdict(val):
        if "Above" in val:
            return "background-color: #fde8e8; color: #c0392b"
        return "background-color: #e8f8e8; color: #1e8449"

    styled = results.style.applymap(color_verdict, subset=["Verdict"])
    st.dataframe(styled, use_container_width=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.markdown("Predicted Budget Max by Occupation")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [
        "#e74c3c" if p > budget_max_input else
        "#2ecc71" if p >= budget_min else
        "#3498db"
        for p in results["Predicted Max (EGP)"]
    ]
    bars = ax.barh(results["Occupation"], results["Predicted Max (EGP)"],
                   color=colors, edgecolor="white", height=0.55)

    # Budget range shading
    ax.axvspan(budget_min, budget_max_input, alpha=0.08, color="green", label="Your budget range")
    ax.axvline(budget_min,       color="green", linestyle="--", linewidth=1.2, label=f"Min {budget_min:,}")
    ax.axvline(budget_max_input, color="orange", linestyle="--", linewidth=1.2, label=f"Max {budget_max_input:,}")

    for bar, val in zip(bars, results["Predicted Max (EGP)"]):
        ax.text(val + 30, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    ax.set_xlabel("Predicted Budget Max (EGP)")
    ax.set_title("Predicted Fair Rent by Occupation")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

    # ── Summary ───────────────────────────────────────────────────────────────
    within = results[results["Verdict"] == "Within your range"]
    st.markdown("---")
    if not within.empty:
        occ_list = ", ".join(within["Occupation"].tolist())
        st.success(f"**{len(within)} occupation(s)** fit within your budget ({budget_min:,}–{budget_max_input:,} EGP): **{occ_list}**")
    else:
        st.warning("No occupation prediction falls within your stated budget range. Consider adjusting your range.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Roommate Project · Price Estimator · Random Forest Model")

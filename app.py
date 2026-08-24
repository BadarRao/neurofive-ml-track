"""
Titanic Survival Predictor
NeuroFive ML Track, Task 10

A Streamlit web app that loads the trained pipeline from Task 7
and predicts survival for passenger details entered by the user.
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

MODEL_PATH = "titanic_pipeline.joblib"

# The exact column order the pipeline was fitted on.
# This must match the training notebook or the prediction will fail.
FEATURE_COLUMNS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
    "FamilySize", "IsAlone", "Title", "FarePerPerson"
]


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the saved pipeline once and keep it in memory across reruns."""
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_PATH}`. Make sure the model file sits in the "
        "same folder as this app and has been committed to the repository."
    )
    st.stop()
except Exception as error:
    st.error(
        "The model file could not be loaded. This is usually a library version "
        "mismatch between the environment that trained the model and this one. "
        "Check that requirements.txt pins the same scikit-learn version used "
        "during training."
    )
    st.exception(error)
    st.stop()


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------

def engineer_features(data):
    """
    Recreate the engineered features from Task 7.

    These were built outside the pipeline, so they have to be rebuilt here
    before calling predict. All operations are row-wise.
    """
    data = data.copy()
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    data["FarePerPerson"] = data["Fare"] / data["FamilySize"]
    return data


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger details below and the model will estimate their chance of "
    "surviving the Titanic disaster. The model is a Logistic Regression pipeline "
    "trained on the Kaggle Titanic dataset."
)

st.divider()


# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------

st.subheader("Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Ticket Class",
        options=[1, 2, 3],
        index=2,
        format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x],
        help="Passenger class. First class cabins were closer to the lifeboats."
    )

    sex = st.radio("Sex", options=["male", "female"], horizontal=True)

    age = st.slider("Age", min_value=0, max_value=80, value=30)

    title = st.selectbox(
        "Title",
        options=["Mr", "Mrs", "Miss", "Master", "Rare"],
        help="Taken from the passenger's name. 'Master' indicates a young boy. "
             "'Rare' covers titles such as Dr, Rev, Col and Countess."
    )

with col2:
    sibsp = st.number_input(
        "Siblings / Spouses aboard",
        min_value=0, max_value=10, value=0, step=1
    )

    parch = st.number_input(
        "Parents / Children aboard",
        min_value=0, max_value=10, value=0, step=1
    )

    fare = st.number_input(
        "Ticket Fare (£)",
        min_value=0.0, max_value=550.0, value=32.0, step=1.0,
        help="Total fare paid for the ticket, not per person."
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        format_func=lambda x: {
            "S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"
        }[x]
    )

# Show the derived values so the user can see what the model actually receives
family_size = sibsp + parch + 1
st.caption(
    f"Derived automatically: family size {family_size}, "
    f"{'travelling alone' if family_size == 1 else 'travelling with family'}, "
    f"fare per person £{fare / family_size:.2f}"
)

st.divider()


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------

if st.button("Predict Survival", type="primary", width='stretch'):

    passenger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": float(age),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Fare": float(fare),
        "Embarked": embarked,
        "Title": title
    }])

    passenger = engineer_features(passenger)

    prediction = model.predict(passenger[FEATURE_COLUMNS])[0]
    probability = model.predict_proba(passenger[FEATURE_COLUMNS])[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.success(f"**Likely to survive** with {probability:.1%} estimated probability")
    else:
        st.error(f"**Unlikely to survive** with {probability:.1%} estimated survival probability")

    st.progress(float(probability))

    result_col1, result_col2 = st.columns(2)
    result_col1.metric("Survival probability", f"{probability:.1%}")
    result_col2.metric("Model verdict", "Survived" if prediction == 1 else "Did not survive")

    with st.expander("What the model received"):
        st.dataframe(passenger[FEATURE_COLUMNS], width='stretch')


# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------

st.divider()

with st.expander("About this model"):
    st.markdown(
        """
        **Model:** Logistic Regression inside a scikit-learn Pipeline

        **Preprocessing handled automatically by the pipeline:**
        - Median imputation for missing numerical values
        - Most frequent imputation for missing categorical values
        - StandardScaler on numerical features
        - OneHotEncoder on categorical features

        **Engineered features:** FamilySize, IsAlone, Title, FarePerPerson

        **Performance:** roughly 0.83 cross validated accuracy on the training data,
        with about 0.84 accuracy on a held out test set.

        This is a learning project built for the NeuroFive ML Track. The predictions
        describe patterns in the 1912 passenger manifest and are not meaningful for
        anything beyond that dataset.
        """
    )

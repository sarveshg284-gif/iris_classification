
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Title
st.title("🌸 Iris Flower Classification")
st.write("Machine Learning Based Flower Prediction System")

st.markdown("---")

# Load Dataset
@st.cache_resource
def train_model():
    df = pd.read_csv("iris_dataset.csv")

    X = df.drop("species", axis=1)
    y = df["species"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)
    return model

model = train_model()

# Sidebar
st.sidebar.header("Project Information")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Model Accuracy: 93.33%")
st.sidebar.write("Dataset: Iris Flower Dataset")
st.sidebar.write("Total Records: 150")

# User Input
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.1,
        max_value=10.0,
        value=5.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.1,
        max_value=10.0,
        value=1.4
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.1,
        max_value=10.0,
        value=3.5
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=10.0,
        value=0.2
    )

# Prediction Button
if st.button("Predict Iris Species", type="primary"):

    input_data = pd.DataFrame([{
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    species_images = {
        "setosa": "🌼",
        "versicolor": "🌺",
        "virginica": "🌷"
    }

    st.success(
        f"Predicted Species: {prediction.capitalize()}"
    )

    st.markdown(
        f"# {species_images.get(prediction, '🌸')} {prediction.capitalize()}"
    )

    st.subheader("Prediction Probabilities")

    probability_df = pd.DataFrame({
        "Species": model.classes_,
        "Probability": probabilities
    })

    st.bar_chart(
        probability_df.set_index("Species")
    )

    st.write("Prediction completed successfully!")

# Footer
st.markdown("---")
st.caption("Developed by Sarvesh Sachin Ghadi | Machine Learning Project")

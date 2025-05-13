import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Animated gradient background using CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: white;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .main {
        font-family: 'Segoe UI', sans-serif;
    }
    .alert {
        padding: 20px;
        margin-top: 20px;
        border-radius: 5px;
        font-size: 18px;
        text-align: center;
        font-weight: bold;
    }
    .green {background-color: #4CAF50; color: white;}
    .red {background-color: #f44336; color: white;}
    </style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart_disease_final_filled.csv")

df = load_data()
target_column = "Output"

st.title("💓 Heart Disease Prediction App")

if target_column not in df.columns:
    st.error(f"Target column '{target_column}' not found.")
else:
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Train/test split for accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @st.cache_resource
    def train_model():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        return model, accuracy

    model, accuracy = train_model()

    st.markdown(f"### 🎯 Model Accuracy: `{accuracy*100:.2f}%`")

    st.markdown("### 📋 Input Patient Information:")
    user_input = {}
    for col in X.columns:
        if df[col].dtype in ['float64', 'int64']:
            user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        else:
            user_input[col] = st.selectbox(f"{col}", df[col].unique())

    input_df = pd.DataFrame([user_input])

    if st.button("🔍 Predict"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.subheader("📊 Prediction Confidence")
        fig1, ax1 = plt.subplots()
        ax1.bar(["No Heart Disease", "Heart Disease"], proba, color=["green", "red"])
        ax1.set_ylabel("Probability")
        ax1.set_ylim(0, 1)
        st.pyplot(fig1)

        if prediction == 1:
            st.markdown(
                f'<div class="alert red">🔴 Alert: The model predicts <b>Heart Disease</b><br>Confidence: {proba[1]:.2f}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="alert green">🟢 Good News: The model predicts <b>No Heart Disease</b><br>Confidence: {proba[0]:.2f}</div>',
                unsafe_allow_html=True)

    st.subheader("📌 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

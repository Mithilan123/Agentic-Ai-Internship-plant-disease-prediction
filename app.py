import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and data
model = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv("dataset.csv")

st.set_page_config(page_title="Fake Product Detection", layout="wide")

st.title("🛒 Fake Product Detection System")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Prediction", "Visualization", "Dataset"])

# -------------------------------
# HOME
# -------------------------------
if option == "Home":
    st.subheader("Project Overview")
    st.write("""
    This project detects whether a product is **Fake or Genuine**
    based on review characteristics and linguistic features.
    
    Features used:
    - Rating
    - Vocabulary Richness
    - Average Word Length
    - Sentence Count
    - Text Length
    """)

# -------------------------------
# PREDICTION
# -------------------------------
elif option == "Prediction":
    st.subheader("🔍 Predict Product Authenticity")

    rating = st.slider("Rating", 1.0, 5.0)
    vocab = st.number_input("Vocabulary Richness", min_value=0.0)
    avg_len = st.number_input("Average Word Length", min_value=0.0)
    sent_count = st.number_input("Sentence Count", min_value=1)
    text_len = st.number_input("Text Length", min_value=1)

    if st.button("Predict"):
        input_data = [[rating, vocab, avg_len, sent_count, text_len]]
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ This product is likely FAKE")
        else:
            st.success("✅ This product appears GENUINE")

# -------------------------------
# VISUALIZATION
# -------------------------------
elif option == "Visualization":
    st.subheader("📊 Data Visualization")

    col1, col2 = st.columns(2)

    # Rating Distribution
    with col1:
        st.write("### Rating Distribution")
        fig, ax = plt.subplots()
        df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # Fake vs Genuine
    with col2:
        st.write("### Fake vs Genuine Count")
        fig, ax = plt.subplots()
        df['label_binary'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # Text Length Distribution
    st.write("### Text Length Distribution")
    fig, ax = plt.subplots()
    df['text_length'].plot(kind='hist', bins=30, ax=ax)
    st.pyplot(fig)

# -------------------------------
# DATASET VIEW
# -------------------------------
elif option == "Dataset":
    st.subheader("📄 Dataset Overview")

    st.write("### First 10 Rows")
    st.dataframe(df.head(10))

    st.write("### Random Reviews")
    st.write(df['text_'].sample(5))

    st.write("### Dataset Info")
    st.write(df.describe())
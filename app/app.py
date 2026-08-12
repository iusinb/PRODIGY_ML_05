"""
Streamlit app: upload a food photo -> predict category -> estimate calories.

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---- Config ----
IMG_SIZE = (128, 128)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "food_classifier.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "..", "model", "class_names.txt")
CALORIE_CSV_PATH = os.path.join(BASE_DIR, "..", "data", "calorie_data.csv")

PORTION_MULTIPLIERS = {
    "Small (~100g)": 1.0,
    "Medium (~200g)": 2.0,
    "Large (~350g)": 3.5,
}


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH) as f:
        return [line.strip() for line in f if line.strip()]


@st.cache_data
def load_calorie_table():
    return pd.read_csv(CALORIE_CSV_PATH).set_index("class_name")


def predict(image: Image.Image, model, class_names):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    return class_names[top_idx], float(preds[top_idx]), preds


def main():
    st.set_page_config(page_title="Food Calorie Estimator", page_icon="🍽️")
    st.title("🍽️ Food Recognition & Calorie Estimator")
    st.write(
        "Upload a photo of your food. The model will guess the food category "
        "and estimate its calorie content based on the portion size you select."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "No trained model found. Run `python model/train_model.py` first "
            "(see README) to generate `model/food_classifier.keras`."
        )
        return

    model = load_model()
    class_names = load_class_names()
    calorie_table = load_calorie_table()

    uploaded_file = st.file_uploader(
        "Upload a food image", type=["jpg", "jpeg", "png"]
    )

    portion = st.selectbox("Portion size", list(PORTION_MULTIPLIERS.keys()))

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing..."):
            label, confidence, all_preds = predict(image, model, class_names)

        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence: {confidence * 100:.1f}%")

        if label in calorie_table.index:
            cal_per_100g = calorie_table.loc[label, "calories_per_100g"]
            multiplier = PORTION_MULTIPLIERS[portion]
            estimated_calories = cal_per_100g * multiplier
            st.metric("Estimated calories", f"{estimated_calories:.0f} kcal")
            st.caption(
                f"Based on ~{cal_per_100g} kcal/100g average for '{label}', "
                f"scaled to a {portion.lower()} portion. This is an estimate, "
                "not precise nutritional data."
            )
        else:
            st.warning("No calorie data found for this class.")

        with st.expander("See all class probabilities"):
            probs_df = pd.DataFrame(
                {"class": class_names, "probability": all_preds}
            ).sort_values("probability", ascending=False)
            st.bar_chart(probs_df.set_index("class"))


if __name__ == "__main__":
    main()

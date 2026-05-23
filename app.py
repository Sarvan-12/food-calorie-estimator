import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---- Load Model ----
model = tf.keras.models.load_model("food_model.h5")

# ---- Load Class Labels ----
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# ---- Calorie Mapping ----
CALORIE_DATA = {
    "ice_cream": 207,
    "masala_dosa": 168,
    "pizza": 266
}

# ---- Prediction Function ----
def predict_food(image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    food_name = class_labels[idx]
    confidence = preds[0][idx] * 100

    return food_name, confidence

# ---- Streamlit UI ----
st.title("🍕 Food Calorie Estimator")
st.write("Upload a food image to get prediction + calories")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    food, confidence = predict_food(image)

    st.success(f"**Prediction:** {food.capitalize()}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    st.write("---")
    cal = CALORIE_DATA.get(food)

    if cal:
        st.write(f"🔥 **Calories (per 100g)**: {cal} kcal")
    else:
        st.error("⚠️ Calorie data not available for this food")
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# -------------------------------------------------------------
# Page Setup
# -------------------------------------------------------------
st.set_page_config(page_title="🧠 Brain Disease Classifier", layout="wide")
st.title("🧬 Brain MRI Disease Detection & Stage Classification")
st.write("Upload a brain MRI to detect **Alzheimer, Tumor, Parkinson, or Healthy** condition. "
         "If applicable, it will also classify the disease subtype.")

# -------------------------------------------------------------
# Model Loader (Downloads from Google Drive if not found)
# -------------------------------------------------------------
@st.cache_resource
def load_model(file_id, filename):
    if not os.path.exists(filename):
        st.info(f"⬇️ Downloading {filename} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    try:
        model = tf.keras.models.load_model(filename)
        st.success(f"✅ {filename} loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading {filename}: {e}")
        st.stop()

# -------------------------------------------------------------
# Load all models
# -------------------------------------------------------------
# Replace these IDs with your actual Google Drive file IDs
MODEL_MAIN_ID = "1qFQwjgFRMox2qlHvUIpLlVvi_dll2ePi"
MODEL_ALZ_ID = "1pwklSWRBo20jQAMD3DVTv8EgFngxaskh"
MODEL_TUMOR_ID = "1Z0Pj-gQrxCWqqaKwIsjtKSlb9ZzEtWoe"

model_main = load_model(MODEL_MAIN_ID, "model_level1_main_disease.h5")
model_alz = load_model(MODEL_ALZ_ID, "model_level2_alzheimer_stage.h5")
model_tumor = load_model(MODEL_TUMOR_ID, "model_level2_tumor_type.h5")

# -------------------------------------------------------------
# Class Labels
# -------------------------------------------------------------
classes_main = ['alzheimer', 'healthy', 'parkinson', 'tumor']
classes_alz = ['Mild Demented', 'Moderate Demented', 'Non-Demented', 'Very Mild Demented']
classes_tumor = ['Glioma', 'Meningioma', 'Pituitary']

# -------------------------------------------------------------
# Disease Recommendations
# -------------------------------------------------------------
recommendations = {
    "alzheimer": {
        "Non-Demented": {
            "Precautions": [
                "Maintain a healthy diet rich in omega-3 fatty acids.",
                "Engage in regular physical and mental exercises.",
                "Get adequate sleep and avoid stress."
            ],
            "Medications": ["No medication needed; maintain brain health."]
        },
        "Very Mild Demented": {
            "Precautions": [
                "Increase mental stimulation (puzzles, memory games).",
                "Maintain social connections.",
                "Monitor behavior regularly."
            ],
            "Medications": ["Donepezil (Aricept)", "Rivastigmine (Exelon)"]
        },
        "Mild Demented": {
            "Precautions": [
                "Keep a structured daily routine.",
                "Avoid stressful environments.",
                "Ensure calm surroundings."
            ],
            "Medications": ["Donepezil", "Galantamine", "Memantine"]
        },
        "Moderate Demented": {
            "Precautions": [
                "24-hour supervision may be required.",
                "Use reminders or visual labels at home.",
                "Consult neurologist regularly."
            ],
            "Medications": ["Memantine", "Combination therapy (Donepezil + Memantine)"]
        }
    },
    "tumor": {
        "Glioma": {
            "Precautions": [
                "Regular MRI scans to monitor tumor size.",
                "Avoid radiation exposure when unnecessary.",
                "Maintain healthy diet and stay hydrated."
            ],
            "Medications": ["Temozolomide", "Bevacizumab"]
        },
        "Meningioma": {
            "Precautions": [
                "Regular neurological check-ups.",
                "Report new headaches or vision changes immediately."
            ],
            "Medications": ["Steroids to reduce swelling", "Anti-seizure drugs"]
        },
        "Pituitary": {
            "Precautions": [
                "Monitor hormonal changes regularly.",
                "Avoid stress and maintain sleep hygiene."
            ],
            "Medications": ["Cabergoline", "Bromocriptine"]
        }
    },
    "parkinson": {
        "Precautions": [
            "Exercise regularly to improve motor control.",
            "Eat high-fiber food to avoid constipation.",
            "Consult a neurologist for tremor management."
        ],
        "Medications": ["Levodopa", "Carbidopa", "MAO-B inhibitors"]
    },
    "healthy": {
        "Precautions": [
            "Maintain balanced nutrition and mental stimulation.",
            "Avoid smoking and alcohol.",
            "Continue brain exercises regularly."
        ],
        "Medications": ["No medications required."]
    }
}

# -------------------------------------------------------------
# File Upload
# -------------------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload MRI Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

# -------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------
def predict_hierarchical(image):
    img = image.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Level 1: Main disease prediction
    main_pred = model_main.predict(img_array)
    main_idx = np.argmax(main_pred)
    main_label = classes_main[main_idx]
    main_conf = np.max(main_pred) * 100

    sub_label, sub_conf = None, None

    if main_label == 'alzheimer':
        sub_pred = model_alz.predict(img_array)
        sub_label = classes_alz[np.argmax(sub_pred)]
        sub_conf = np.max(sub_pred) * 100

    elif main_label == 'tumor':
        sub_pred = model_tumor.predict(img_array)
        sub_label = classes_tumor[np.argmax(sub_pred)]
        sub_conf = np.max(sub_pred) * 100

    return main_label, main_conf, sub_label, sub_conf

# -------------------------------------------------------------
# Prediction and Display
# -------------------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🧩 Uploaded Brain MRI", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing MRI..."):
            main_label, main_conf, sub_label, sub_conf = predict_hierarchical(image)

        st.success(f"🧠 **Detected Disease:** {main_label.capitalize()} ({main_conf:.2f}% confidence)")

        if sub_label:
            st.info(f"**Subtype:** {sub_label} ({sub_conf:.2f}% confidence)")

        st.divider()

        # Precautions & Medications
        if main_label in ["alzheimer", "tumor"]:
            recs = recommendations[main_label].get(sub_label, {})
        else:
            recs = recommendations.get(main_label, {})

        st.subheader("🩺 Precautions")
        for p in recs.get("Precautions", []):
            st.write(f"- {p}")

        st.subheader("💊 Medications")
        for m in recs.get("Medications", []):
            st.write(f"- {m}")

        st.divider()
        st.info("⚕️ Early detection and continuous medical monitoring are crucial for effective treatment and better outcomes.")
else:
    st.warning("⚠️ Please upload an MRI image to begin analysis.") 
import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="Ophthalmic AI System", page_icon="👁️")

# --- 1. LOAD THE BRAIN (BACKEND) ---
@st.cache_resource
def load_model():
    # Load the ONNX model exported from MATLAB
    session = ort.InferenceSession("Eye_Disease_Brain.onnx")
    return session

try:
    session = load_model()
    model_status = "✅ AI Model Online"
except Exception as e:
    model_status = f"❌ Model Offline: {e}"

# --- 2. PREPROCESSING FUNCTION ---
def process_image(image):
    # Standard resize for medical scans
    image = image.resize((224, 224))
    img_array = np.array(image).astype('float32')
    
    # Transpose to Channel-First (3x224x224) - Required by MATLAB-to-ONNX
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add Batch Dimension (1x3x224x224)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- 3. THE FRONTEND (UI) ---
st.title("🏥 Ophthalmic AI Robot Console")
st.sidebar.header("System Status")
st.sidebar.write(model_status)

# File Uploader
uploaded_file = st.file_uploader("Upload Patient Eye Scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # A. Display the Image
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Patient Scan", use_container_width=True)
    
    # B. Run Diagnosis
    if st.button("Analyze Scan"):
        with st.spinner("AI Robot is thinking..."):
            # Prepare data
            input_name = session.get_inputs()[0].name
            input_data = process_image(image)
            
            # Run Inference
            outputs = session.run(None, {input_name: input_data})
            logits = outputs[0][0]  # Raw brain signals
            
            # THE SOFTMAX GEM: Manually scale the raw output to get true percentages
            # This is what boosts your score back to 90%+
            exp_logits = np.exp(logits - np.max(logits)) # Subtract max for stability
            probs = exp_logits / exp_logits.sum()
            
            # CORRECTED CLASS LIST (Must match MATLAB folders exactly!)
            classes = [
                "Central Serous Chorioretinopathy", 
                "Diabetic Retinopathy", 
                "Disc Edema", 
                "Glaucoma"
            ]
            
            # Get Top Prediction
            top_idx = np.argmax(probs)
            confidence = probs[top_idx] * 100
            diagnosis = classes[top_idx]

            # Drug Logic
            treatment_map = {
                "Central Serous Chorioretinopathy": ("Eplerenone", "Tablet"),
                "Diabetic Retinopathy": ("Anti-VEGF", "Injection"),
                "Disc Edema": ("Corticosteroids", "Oral"),
                "Glaucoma": ("Latanoprost", "1 Drop")
            }
            drug, dosage = treatment_map.get(diagnosis, ("Observation", "N/A"))

            # C. Show Results
            with col2:
                st.success(f"**Diagnosis:** {diagnosis}")
                st.info(f"**Confidence:** {confidence:.1f}%")
                st.warning(f"**Rx:** {drug} ({dosage})")
                
                # Visual Bar Chart for Probability Distribution
                chart_data = dict(zip(classes, probs))
                st.bar_chart(chart_data)

# --- FOOTER ---
st.divider()
st.caption("Developed for Advanced Ophthalmic Engineering")
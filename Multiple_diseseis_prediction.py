import streamlit as st
import numpy as np
import pickle

# Sidebar Configuration
st.sidebar.title("Multiple Disease Prediction System")

disease_options = ["Kidney Prediction", "Liver Prediction", "Parkinson's Prediction"]
selected_option = st.sidebar.radio("", disease_options)

# -------------------- Parkinson's Prediction --------------------
if selected_option == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    # Load Parkinson's model
    parkinsons_model_path = "parkinsons_model.pkl"
    with open(parkinsons_model_path, "rb") as file:
        parkinsons_model = pickle.load(file)

    # Create three columns for input fields
    col1, col2, col3 = st.columns(3)

    def get_float_input(label):
        """Returns a float input from text field, or None if invalid."""
        value = st.text_input(label)
        try:
            return float(value)
        except ValueError:
            return None

    with col1:
        mdvp_fo = get_float_input("MDVP:Fo(Hz)")
        mdvp_flo = get_float_input("MDVP:Flo(Hz)")
        jitter_percent = get_float_input("MDVP:Jitter(%)")
        shimmer_apq3 = get_float_input("Shimmer:APQ3")
        shimmer_dda = get_float_input("Shimmer:DDA")
        jitter_ddp = get_float_input("Jitter:DDP")  # 🔹 Missing Feature Added

    with col2:
        mdvp_fhi = get_float_input("MDVP:Fhi(Hz)")
        mdvp_rap = get_float_input("MDVP:RAP")
        jitter_abs = get_float_input("MDVP:Jitter(Abs)")
        shimmer_apq5 = get_float_input("Shimmer:APQ5")
        hnr = get_float_input("HNR")
        nhr = get_float_input("NHR")  # 🔹 Missing Feature Added

    with col3:
        mdvp_ppq = get_float_input("MDVP:PPQ")
        mdvp_shimmer = get_float_input("MDVP:Shimmer")
        shimmer_apq = get_float_input("MDVP:APQ")
        rpde = get_float_input("RPDE")
        dfa = get_float_input("DFA")
        mdvp_shimmer_db = get_float_input("MDVP:Shimmer(dB)")  # 🔹 Missing Feature Added

    # Additional inputs
    spread1 = get_float_input("spread1")
    spread2 = get_float_input("spread2")
    d2 = get_float_input("D2")
    ppe = get_float_input("PPE")

    # Ensure all inputs are valid
    input_values = [
        mdvp_fo, mdvp_fhi, mdvp_flo, jitter_percent, jitter_abs, mdvp_rap, 
        mdvp_ppq, mdvp_shimmer, shimmer_apq3, shimmer_apq5, shimmer_apq, shimmer_dda, 
        hnr, rpde, dfa, spread1, spread2, d2, ppe, jitter_ddp, mdvp_shimmer_db, nhr
    ]

    if st.button("Predict Parkinson's Disease"):
        if None in input_values:
            st.error("Please enter valid numerical values for all fields.")
        else:
            # Convert each input to a native Python float
            input_values = [float(x) for x in input_values]
            input_array = np.array(input_values).reshape(1, -1)
            prediction = parkinsons_model.predict(input_array)[0]
            probability = parkinsons_model.predict_proba(input_array)[0][1]  # probability for class '1'
            
            st.subheader("Parkinson's Test Result")
            if prediction == 1:
                st.error(f"🔴 High Risk of Parkinson's Disease (Confidence: {probability:.2%})")
            else:
                st.success(f"🟢 Low Risk of Parkinson's Disease (Confidence: {(1-probability):.2%})")

# -------------------- Kidney Disease Prediction --------------------
elif selected_option == "Kidney Prediction":
    st.title("Kidney Disease Prediction using ML")

    # Load Kidney model
    kidney_model_path = "kidney_disease_model.pkl"
    with open(kidney_model_path, "rb") as file:
        kidney_model = pickle.load(file)

    st.write("Enter the following details for Kidney Disease Prediction:")

    # Create input fields for all relevant features
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    bp = st.number_input("Blood Pressure", min_value=0.0, value=80.0)
    sg = st.selectbox("Specific Gravity", ["1.005", "1.010", "1.015", "1.020", "1.025", "1.030"])
    al = st.selectbox("Albumin", ["0", "1", "2", "3", "4", "5"])
    su = st.selectbox("Sugar", ["0", "1", "2", "3", "4", "5"])
    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
    pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
    ba = st.selectbox("Bacteria", ["notpresent", "present"])
    bgr = st.number_input("Blood Glucose Random", min_value=0.0, value=120.0)
    bu = st.number_input("Blood Urea", min_value=0.0, value=30.0)
    sc = st.number_input("Serum Creatinine", min_value=0.0, value=1.0)
    sod = st.number_input("Sodium", min_value=0.0, value=140.0)
    pot = st.number_input("Potassium", min_value=0.0, value=4.5)
    hemo = st.number_input("Hemoglobin", min_value=0.0, value=13.0)
    pcv = st.number_input("Packed Cell Volume", min_value=0.0, value=40.0)
    wc = st.number_input("White Blood Cell Count", min_value=0, value=6000)
    rc = st.number_input("Red Blood Cell Count", min_value=0.0, value=5.0)
    htn = st.selectbox("Hypertension", ["yes", "no"])
    dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
    cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
    appet = st.selectbox("Appetite", ["good", "poor"])
    pe = st.selectbox("Pedal Edema", ["yes", "no"])
    ane = st.selectbox("Anemia", ["yes", "no"])

    # Convert categorical values to numeric values (you may need to adjust based on your model's encoding)
    sg = float(sg)
    al = int(al)
    su = int(su)
    rbc = 1 if rbc == "normal" else 0
    pc = 1 if pc == "normal" else 0
    pcc = 1 if pcc == "present" else 0
    ba = 1 if ba == "present" else 0
    htn = 1 if htn == "yes" else 0
    dm = 1 if dm == "yes" else 0
    cad = 1 if cad == "yes" else 0
    appet = 1 if appet == "good" else 0
    pe = 1 if pe == "yes" else 0
    ane = 1 if ane == "yes" else 0

    if st.button("Predict Kidney Disease"):
        # Convert each input to a native Python float
        kidney_input = [
            float(age), float(bp), float(sg), float(al), float(su), float(rbc), 
            float(pc), float(pcc), float(ba), float(bgr), float(bu), float(sc), 
            float(sod), float(pot), float(hemo), float(pcv), float(wc), float(rc), 
            float(htn), float(dm), float(cad), float(appet), float(pe), float(ane)
        ]
        
        # Check if the number of features matches the expected number (25 features)
        # If the model expects 25 features, we can add a default value for the missing feature (e.g., id)
        if len(kidney_input) == 24:
            # Add a dummy value for the missing 25th feature (id or other feature)
            kidney_input.insert(0, 0)  # Insert a dummy value (0) for the 'id' or missing feature
            
        # Ensure the input has the correct shape (25 features)
        if len(kidney_input) == 25:
            input_array = np.array(kidney_input).reshape(1, -1)
            prediction = kidney_model.predict(input_array)[0]
            probability = kidney_model.predict_proba(input_array)[0][1]

            # Output the classification result
            st.subheader("Kidney Disease Test Result")
            if prediction == 1:
                st.error(f"🔴 High Risk of Kidney Disease (Confidence: {probability:.2%})")
                st.write("Predicted Classification: **Kidney Disease**")
            else:
                st.success(f"🟢 Low Risk of Kidney Disease (Confidence: {(1-probability):.2%})")
                st.write("Predicted Classification: **No Kidney Disease**")
        else:
            st.error("Input data has incorrect shape. Please check your inputs.")

# -------------------- Liver Disease Prediction --------------------
elif selected_option == "Liver Prediction":
    st.title("Liver Disease Prediction using ML")

    # Load Liver model
    liver_model_path = "Liver_disease_model.pkl"
    with open(liver_model_path, "rb") as file:
        liver_model = pickle.load(file)

    st.write("Enter the following details for Liver Disease Prediction:")

    # Example input fields (adjust these based on your model features)
    age = st.number_input("Age", min_value=1, max_value=100, value=40)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=1.0, step=0.1)
    direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.3, step=0.1)
    alkaline_phosphatase = st.number_input("Alkaline Phosphatase", min_value=0, value=250)
    alanine_aminotransferase = st.number_input("Alanine Aminotransferase", min_value=0, value=30)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0, value=30)
    total_proteins = st.number_input("Total Proteins", min_value=0.0, value=6.0, step=0.1)
    albumin = st.number_input("Albumin", min_value=0.0, value=3.5, step=0.1)
    ag_ratio = st.number_input("A/G Ratio", min_value=0.0, value=1.0, step=0.1)
    
    # Add input for Albumin and Globulin Ratio
    albumin_and_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, value=1.0, step=0.1)

    # Convert gender to numeric value (example: Male=1, Female=0)
    gender_value = 1 if gender == "Male" else 0

    if st.button("Predict Liver Disease"):
        # Convert each input to a native Python float
        liver_input = [
            float(age), float(gender_value), float(total_bilirubin),
            float(direct_bilirubin), float(alkaline_phosphatase),
            float(alanine_aminotransferase), float(aspartate_aminotransferase),
            float(total_proteins), float(albumin), float(ag_ratio),
            float(albumin_and_globulin_ratio)  # Add this line
        ]
        input_array = np.array(liver_input).reshape(1, -1)
        prediction = liver_model.predict(input_array)[0]
        probability = liver_model.predict_proba(input_array)[0][1]

        st.subheader("Liver Disease Test Result")
        if prediction == 1:
            st.error(f"🔴 High Risk of Liver Disease (Confidence: {probability:.2%})")
        else:
            st.success(f"🟢 Low Risk of Liver Disease (Confidence: {(1-probability):.2%})")

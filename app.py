import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open("svc.pkl",'rb'))

# Load datasets
@st.cache_data
def load_data():
    precautions = pd.read_csv("datasets/precautions_df.csv")
    work_out = pd.read_csv("datasets/workout_df.csv")
    description = pd.read_csv("datasets/description.csv")
    medications = pd.read_csv("datasets/medications.csv")
    diets = pd.read_csv("datasets/diets.csv")
    return precautions, work_out, description, medications, diets

# Helper function to get disease information
def get_disease_info(dis, precautions, work_out, description, medications, diets):
    # Description
    desc = description[description['Disease'] == dis]['Description'].values[0]
    
    # Precautions
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0]
    
    # Medications
    med = medications[medications['Disease'] == dis]['Medication'].values[0].split(',')
    
    # Diets
    die = diets[diets['Disease'] == dis]['Diet'].values[0].split(',')
    
    # Workouts
    workout = work_out[work_out['disease'] == dis]['workout'].values[0].split(',')
    
    return desc, pre, med, die, workout

# Symptoms dictionary and diseases list
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 
                 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 
                 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 
                 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 
                 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 
                 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 
                 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 
                 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 
                 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 
                 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 
                 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 
                 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 
                 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 
                 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 
                 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 
                 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 
                 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 
                 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 
                 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 
                 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 
                 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 
                 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 
                 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 
                 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 
                 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 
                 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 
                 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 
                 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 
                 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 
                 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 
                 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 
                 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 
                 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def main():
    st.set_page_config(page_title="Medicine Recommendation System", page_icon="💊", layout="wide")
    
    # Load model and data
    svc = load_model()
    precautions, work_out, description, medications, diets = load_data()
    
    # App title and description
    st.title("💊 Medicine Recommendation System")
    st.markdown("""
    This system helps predict potential diseases based on symptoms and provides recommendations for:
    - Medications 💊
    - Precautions 🛡️
    - Diets 🥗
    - Workouts 🏋️‍♂️
    """)
    
    # Sidebar with symptom selection
    st.sidebar.header("Select Symptoms")
    selected_symptoms = st.sidebar.multiselect(
        "Choose your symptoms:",
        options=list(symptoms_dict.keys()),
        help="Select all symptoms you're experiencing"
    )
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Predict Disease"):
            if len(selected_symptoms) == 0:
                st.warning("Please select at least one symptom")
            else:
                # Create input vector
                input_vector = np.zeros(len(symptoms_dict))
                for symptom in selected_symptoms:
                    input_vector[symptoms_dict[symptom]] = 1
                
                # Predict disease
                predicted_disease = diseases_list[svc.predict([input_vector])[0]]
                
                # Get disease information
                desc, pre, med, die, workout = get_disease_info(
                    predicted_disease, precautions, work_out, description, medications, diets
                )
                
                # Display results
                st.success(f"Predicted Disease: **{predicted_disease}**")
                
                # Tabs for different information sections
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Description", "Precautions", "Medications", "Diet", "Workout"
                ])
                
                with tab1:
                    st.subheader("Description")
                    st.write(desc)
                
                with tab2:
                    st.subheader("Precautions")
                    for i, precaution in enumerate(pre, 1):
                        st.write(f"{i}. {precaution}")
                
                with tab3:
                    st.subheader("Medications")
                    for i, medication in enumerate(med, 1):
                        st.write(f"{i}. {medication.strip()}")
                
                with tab4:
                    st.subheader("Recommended Diet")
                    for i, diet in enumerate(die, 1):
                        st.write(f"{i}. {diet.strip()}")
                
                with tab5:
                    st.subheader("Recommended Workout")
                    for i, exercise in enumerate(workout, 1):
                        st.write(f"{i}. {exercise.strip()}")
    
    with col2:
        st.image(Image.open("health_care.jpg"), caption="Health is Wealth", use_column_width=True)
        st.markdown("""
        **Note:**  
        This system provides recommendations based on machine learning.  
        Always consult with a healthcare professional for proper diagnosis and treatment.
        """)

if __name__ == "__main__":
    main()
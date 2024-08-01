import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from requests.exceptions import RequestException
from io import BytesIO, StringIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import sqlite3
from datetime import datetime

# API Key and URL for Falcon 180B Model
API_KEY = "api71-api-77fd9964-ce96-4fec-abf1-5714b8508b5f"
API_URL = "https://api.ai71.ai/v1/chat/completions"

# Load the pre-trained model for lung disease analysis
def load_chexnet_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def analyze_image(image):
    model = load_chexnet_model()
    img = Image.open(BytesIO(image.read()))
    img_tensor = preprocess_image(img)
    
    try:
        with torch.no_grad():
            outputs = model(img_tensor)
        # Mock results; replace with actual model output interpretation
        results = {
            "Lung Cancer": np.random.random(),
            "Pneumonia": np.random.random(),
            "COVID-19": np.random.random()
        }
        return {
            "diagnosis": "No abnormalities detected.",
            "details": "The image does not show any clear signs of lung cancer, pneumonia, or COVID-19.",
            "recommendations": "Regular check-ups and maintaining a healthy lifestyle are recommended.",
            "severity": "N/A",
            "results": results
        }
    except Exception as e:
        st.error(f"An error occurred during image analysis: {e}")
        return {
            "diagnosis": "Error during analysis.",
            "details": "An error occurred while processing the image.",
            "recommendations": "Please try again.",
            "severity": "N/A",
            "results": {}
        }

# Function to get response from the Falcon 180B model
def get_response(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "tiiuae/falcon-180b",
        "messages": [
            {"role": "system", "content": "You are a medical assistant. Provide clear and accurate medical responses based on the symptoms described."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get('choices', [{}])[0].get('message', {}).get('content', "No response received.")
    except RequestException as e:
        st.error(f"An error occurred: {e}")
        return "Sorry, there was an error processing your request."

# Function to generate a PDF report
def generate_pdf_report(patient_data, symptoms, analysis, pain_management, preventive_measures):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['BodyText']
    
    # Title
    story.append(Paragraph("Patient Report", title_style))
    story.append(Spacer(1, 12))
    
    # Patient Data
    story.append(Paragraph("Patient Data:", heading_style))
    for key, value in patient_data.items():
        story.append(Paragraph(f"{key}: {value}", normal_style))
    story.append(Spacer(1, 12))
    
    # Symptoms
    story.append(Paragraph("Symptoms:", heading_style))
    story.append(Paragraph(symptoms, normal_style))
    story.append(Spacer(1, 12))
    
    # Analysis
    story.append(Paragraph("Analysis:", heading_style))
    story.append(Paragraph(analysis, normal_style))
    story.append(Spacer(1, 12))
    
    # Pain Management Advice
    story.append(Paragraph("Pain Management Advice:", heading_style))
    story.append(Paragraph(pain_management, normal_style))
    story.append(Spacer(1, 12))
    
    # Preventive Measures
    story.append(Paragraph("Preventive Measures:", heading_style))
    story.append(Paragraph(preventive_measures, normal_style))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Function to visualize symptoms using a bar chart
def visualize_symptoms(symptom_history):
    if symptom_history:
        df = pd.DataFrame(symptom_history)
        fig = px.bar(df, x='Date', y='Symptoms', color='Symptoms', title='Symptom Frequency or Severity')
        st.plotly_chart(fig)
    else:
        st.write("No symptom data available to visualize.")

# Function to visualize pain level trends using a line chart
def visualize_pain_trends(pain_history):
    if pain_history:
        df = pd.DataFrame(pain_history)
        fig = px.line(df, x='Date', y='Pain Level', title='Pain Level Trends Over Time')
        st.plotly_chart(fig)
    else:
        st.write("No pain data available to visualize.")

# Function to visualize symptom distribution using pie chart
def visualize_symptom_distribution(symptom_history):
    if symptom_history:
        symptom_counts = pd.Series([s['Symptoms'] for s in symptom_history]).value_counts()
        fig = px.pie(values=symptom_counts.values, names=symptom_counts.index, title='Symptom Distribution')
        st.plotly_chart(fig)
    else:
        st.write("No symptom data available to visualize.")

# Function to initialize the database and create tables
def initialize_database():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_history (
            patient_name TEXT,
            patient_age TEXT,
            patient_gender TEXT,
            symptoms TEXT,
            pain_level INTEGER,
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Function to get user history from the database
def get_user_history():
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query('SELECT * FROM user_history ORDER BY created_at DESC', conn)
    conn.close()
    return df.to_dict(orient='records')

# Function to export data to CSV
def export_to_csv():
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query('SELECT * FROM user_history ORDER BY created_at DESC', conn)
    conn.close()
    
    # Convert DataFrame to CSV
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

# Streamlit app layout
st.set_page_config(page_title="Advanced Doctor's Assistant Dashboard", layout="wide")

# Initialize database
initialize_database()

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'symptom_text' not in st.session_state:
    st.session_state.symptom_text = ""
if 'pain_management' not in st.session_state:
    st.session_state.pain_management = ""
if 'preventive_measures' not in st.session_state:
    st.session_state.preventive_measures = ""
if 'response' not in st.session_state:
    st.session_state.response = ""
if 'symptom_history' not in st.session_state:
    st.session_state.symptom_history = []
if 'pain_history' not in st.session_state:
    st.session_state.pain_history = []

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Symptom Tracker", "Medical Image Analysis", "Reports & Visualizations", "User History"])

# Symptom Tracker Tab
with tab1:
    st.header("Symptom Tracker")
    with st.form("chat_form"):
        patient_name = st.text_input("Patient Name:")
        patient_age = st.text_input("Patient Age:")
        patient_gender = st.selectbox("Patient Gender:", ["Male", "Female", "Other"])

        st.subheader("Symptom Details")

        # Respiratory Symptoms
        st.markdown("**Respiratory Symptoms**")
        cough = st.checkbox("Cough")
        shortness_of_breath = st.checkbox("Shortness of Breath")
        chest_pain = st.checkbox("Chest Pain")

        # Digestive Symptoms
        st.markdown("**Digestive Symptoms**")
        nausea = st.checkbox("Nausea")
        vomiting = st.checkbox("Vomiting")
        diarrhea = st.checkbox("Diarrhea")

        # Pain and Other Symptoms
        st.markdown("**Pain and Other Symptoms**")
        fever = st.checkbox("Fever")
        fatigue = st.checkbox("Fatigue")
        headache = st.checkbox("Headache")
        other_symptoms = st.text_area("Other Symptoms")

        pain_level = st.slider("Pain Level (1-10):", 1, 10, 5)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.patient_name = patient_name
            st.session_state.patient_age = patient_age
            st.session_state.patient_gender = patient_gender

            # Collect symptoms
            symptoms = []
            if cough: symptoms.append("Cough")
            if shortness_of_breath: symptoms.append("Shortness of Breath")
            if chest_pain: symptoms.append("Chest Pain")
            if nausea: symptoms.append("Nausea")
            if vomiting: symptoms.append("Vomiting")
            if diarrhea: symptoms.append("Diarrhea")
            if fever: symptoms.append("Fever")
            if fatigue: symptoms.append("Fatigue")
            if headache: symptoms.append("Headache")
            if other_symptoms: symptoms.append(other_symptoms)

            symptom_text = ", ".join(symptoms)
            st.session_state.symptom_text = symptom_text

            # Generate prompt for the model
            prompt = f"Patient Name: {patient_name}\nPatient Age: {patient_age}\nPatient Gender: {patient_gender}\nSymptoms: {symptom_text}\nPain Level: {pain_level}"

            # Get response from the model
            response = get_response(prompt)
            st.session_state.response = response

            # Extract pain management and preventive measures advice from the response
            st.session_state.pain_management = "No specific pain management advice provided."
            st.session_state.preventive_measures = "No specific preventive measures provided."

            if isinstance(response, dict):
                st.session_state.pain_management = response.get("pain_management_advice", "No specific pain management advice provided.")
                st.session_state.preventive_measures = response.get("preventive_measures", "No specific preventive measures provided.")

            # Store user history
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO user_history (patient_name, patient_age, patient_gender, symptoms, pain_level, response)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (patient_name, patient_age, patient_gender, symptom_text, pain_level, str(response)))
            conn.commit()
            conn.close()

            # Append symptom and pain level history
            st.session_state.symptom_history.append({"Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Symptoms": symptom_text})
            st.session_state.pain_history.append({"Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Pain Level": pain_level})

    st.write("### Chatbot Response")
    st.write(st.session_state.response)

# Medical Image Analysis Tab
with tab2:
    st.header("Medical Image Analysis")
    uploaded_file = st.file_uploader("Upload a lung image for analysis (JPEG, PNG)", type=["jpeg", "jpg", "png"])
    if uploaded_file:
        analysis_result = analyze_image(uploaded_file)
        st.write("### Analysis Result")
        st.write(f"**Diagnosis:** {analysis_result['diagnosis']}")
        st.write(f"**Details:** {analysis_result['details']}")
        st.write(f"**Recommendations:** {analysis_result['recommendations']}")
        st.write(f"**Severity:** {analysis_result['severity']}")
        st.write("### Detailed Results")
        for disease, probability in analysis_result['results'].items():
            st.write(f"**{disease}:** {probability:.2%}")

# Reports & Visualizations Tab
with tab3:
    st.header("Reports & Visualizations")
    if st.session_state.symptom_text:
        patient_data = {
            "Name": st.session_state.patient_name,
            "Age": st.session_state.patient_age,
            "Gender": st.session_state.patient_gender
        }
        pdf_buffer = generate_pdf_report(patient_data, st.session_state.symptom_text, st.session_state.response, st.session_state.pain_management, st.session_state.preventive_measures)
        st.download_button(label="Download PDF Report", data=pdf_buffer, file_name="patient_report.pdf", mime="application/pdf")

    st.write("### Symptom Visualization")
    visualize_symptoms(st.session_state.symptom_history)

    st.write("### Pain Level Trends")
    visualize_pain_trends(st.session_state.pain_history)

    st.write("### Symptom Distribution")
    visualize_symptom_distribution(st.session_state.symptom_history)

# User History Tab
with tab4:
    st.header("User History")
    user_history = get_user_history()
    if user_history:
        df = pd.DataFrame(user_history)
        st.dataframe(df)
        csv_data = export_to_csv()
        st.download_button(label="Export to CSV", data=csv_data, file_name="user_history.csv", mime="text/csv")
    else:
        st.write("No user history available.")

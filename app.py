import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("diabetes_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    input_df = pd.DataFrame([
        [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], 
        columns=["Pregnancies", "Glucose", "BloodPressure","SkinThickness", "Insulin", "BMI","DiabetesPedigreeFunction", "Age"
    ])

    prediction = model.predict(input_df)[0]

    return "Diabetic" if prediction == 1 else "Not Diabetic"

inputs = [
    gr.Number(label="Pregnancies", value=0),
    gr.Number(label="Glucose", value=120),
    gr.Number(label="Blood Pressure", value=70),
    gr.Number(label="Skin Thickness", value=20),
    gr.Number(label="Insulin", value=80),
    gr.Number(label="BMI", value=25.0),
    gr.Number(label="Diabetes Pedigree Function", value=0.5),
    gr.Number(label="Age", value=30),
]

app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Prediction System",
)

app.launch(share=True)

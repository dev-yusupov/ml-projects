from django.shortcuts import render
from django.http.request import HttpRequest
import pickle
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")

# Create your views here.
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def predict(request: HttpRequest):
    if request.method == "GET":
        return render(request, 'index.html')
    
    if request.method == "POST":
        age = int(request.POST["age"])
        anaemia = int(request.POST["anaemia"])
        creatinine_phosphokinase = int(request.POST["creatinine_phosphokinase"])
        diabetes = int(request.POST["diabetes"])
        ejection_fraction = int(request.POST["ejection_fraction"])
        high_blood_pressure = int(request.POST["high_blood_pressure"])
        serum_creatinine = float(request.POST["serum_creatinine"])
        sex = int(request.POST["sex"])
        smoking = int(request.POST["smoking"])

        X = pd.DataFrame({
            "age": [age],
            "anaemia": [anaemia],
            "creatinine_phosphokinase": [creatinine_phosphokinase],
            "diabetes": [diabetes],
            "ejection_fraction": [ejection_fraction],
            "high_blood_pressure": [high_blood_pressure],
            "serum_creatinine": [serum_creatinine],
            "sex": [sex],
            "smoking": [smoking]
        })

        prediction = model.predict(X)

        print(prediction)
        survived = bool(prediction[0])

        return render(request, 'index.html', {'prediction': survived})

from django.shortcuts import render
from django.http.request import HttpRequest
import pickle
import pandas as pd

# Create your views here.
with open("log_reg_model.pkl", 'rb') as f:
    model = pickle.load(f)

def predict(request: HttpRequest):
    if request.method == "GET":
        return render(request, 'index.html')
    
    if request.method == "POST":
        pclass = int(request.POST["pclass"])
        sex = request.POST["sex"]
        age = int(request.POST["age"])

        X = pd.DataFrame({"pclass": [pclass], "sex": [sex], "age": [age]})
        X["sex"] = X["sex"].map({"male": 0, "female": 1})

        prediction = model.predict(X)

        return render
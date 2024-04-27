from django.shortcuts import render
from django.http.request import HttpRequest
import pickle
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "log_reg_model.pkl")

# Create your views here.
with open(model_path, 'rb') as f:
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
        
        # Convert prediction to boolean value (True if survived, False otherwise)
        survived = bool(prediction[0])

        # Pass the predicted value to the template using a context dictionary
        return render(request, 'index.html', {'survived': survived})

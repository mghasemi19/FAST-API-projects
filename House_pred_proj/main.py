from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()

# Load model once when app starts
model = joblib.load("model.pkl")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>House Price Predictor</title>
        </head>
        <body>
            <h2>House Price Prediction</h2>
            <form action="/predict" method="post">
                <label>Enter Area (sq ft):</label>
                <input type="number" name="area" required>
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
def predict(area: float = Form(...)):
    prediction = model.predict(np.array([[area]]))[0]

    return f"""
    <html>
        <body>
            <h2>Predicted Price: ₹{prediction:,.2f}</h2>
            <a href="/">Go Back</a>
        </body>
    </html>
    """
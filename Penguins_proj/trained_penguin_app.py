from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before the server starts → run initialization code
    # Set up the ML model here
    ml_models["clf"] = joblib.load("model.joblib")
    ml_models["le"] = joblib.load("label_encoder.joblib")


    yield
    # After the server shuts down → run cleanup code
    # Clean up the models and release resources

    ml_models.clear()

app = FastAPI(lifespan=lifespan)


#app = FastAPI()

# ---------- Mount static files and templates ----------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------- Root page ----------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


# ---------- Predict route (handles form data) ----------
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  bill_length_mm: float = Form(...),
                  flipper_length_mm: float = Form(...)):

    param = {"bill_length_mm": bill_length_mm, "flipper_length_mm": flipper_length_mm}

    if bill_length_mm <= 0.0 or flipper_length_mm <= 0.0:
        result = "❌ Invalid input values"
    else:
        result_class = ml_models["clf"].predict([[bill_length_mm, flipper_length_mm]])
        result = ml_models["le"].inverse_transform(result_class)[0]

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "param": param}
    )


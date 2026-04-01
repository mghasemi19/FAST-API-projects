from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks
from fastapi.responses import RedirectResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

import numpy as np
from tensorflow import keras
from PIL import Image
from io import BytesIO
import base64


from zipfile import ZipFile

# MNIST model meta parameters
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 50

ml_models = {}

training_status = {
    "running": False,
    "progress": 0,
    "message": "Idle"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before the server starts → run initialization code
    # Set up the ML model here
    ml_models["cnn"] = keras.models.load_model("mnist_model.keras")


    yield
    # After the server shuts down → run cleanup code
    # Clean up the models and release resources

    ml_models.clear()

# Build the FAST API application
app = FastAPI(lifespan=lifespan)

# static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
#templates = Jinja2Templates(directory="templates")
templates = Jinja2Templates(directory="test_templates")

# hello endpoint to test 
@app.get("/hello")
async def root():
    return {
        "Name": "Number Prediction",
        "description": "This is a number prediction model based on the image the user uploads.",
    }

# GET endpoint to upload the image coming from the user
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None, "image": None},
    )

# Predict the number and shows the result
@app.post("/predict-image/", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):

    img_bytes = await file.read()

    # convert uploaded image to base64 so HTML can display it
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # preprocess image for MNIST model
    img = Image.open(BytesIO(img_bytes)).convert("L").resize((28, 28))
    img_array = np.array(img).astype("float32") / 255
    img_array = np.expand_dims(img_array, (0, -1))

    prediction = ml_models["cnn"].predict(img_array)
    result = int(np.argmax(prediction, axis=-1)[0])
    prob = np.max(prediction) * 100
    prob = "{:.2f}".format(prob)  # Format to two decimal places

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "prob": prob,
            "image": img_base64
        },
    )
# curl.exe -X POST "http://127.0.0.1:8003/predict-image/" -F "file=@img/test_img0.PNG"

def retrain_model(train_img, labels_data):

    training_status["running"] = True
    training_status["progress"] = 0
    training_status["message"] = "Training started"

    class ProgressCallback(keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):
            training_status["progress"] = int((epoch + 1) / epochs * 100)
            training_status["message"] = f"Epoch {epoch+1}/{epochs}"

    ml_models["cnn"].fit(
        train_img,
        labels_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[ProgressCallback()]
    )

    training_status["running"] = False
    training_status["message"] = "Training finished"


@app.post("/retrain_upload_file")
async def retrain_upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):

    img_files = []
    labels_file = None
    train_img = None

    with ZipFile(BytesIO(await file.read()), 'r') as zfile:
        for fname in zfile.namelist():
            if fname[-4:] == '.txt' and fname[:2] != '__':
                labels_file = fname
            elif fname[-4:] == '.png':
                img_files.append(fname)

        if len(img_files) == 0:
            return {"error": "No training images (png files) found."}
        else:
            for fname in sorted(img_files):
                with zfile.open(fname) as img_file:
                    img = img_file.read()

                    # process image
                    img = Image.open(BytesIO(img)).convert('L')
                    img = np.array(img).astype("float32") / 255
                    img = np.expand_dims(img, (0, -1))

                    if train_img is None:
                        train_img = img
                    else:
                        train_img = np.vstack((train_img, img))     # np.vstack() stacks images vertically to build the batch.

        if labels_file is None:
            return {"error": "No training labels file (txt file) found."}
        else:
            with zfile.open(labels_file) as labels:
                labels_data = labels.read()
                labels_data = labels_data.decode("utf-8").split()
                labels_data = np.array(labels_data).astype("int")
                labels_data = keras.utils.to_categorical(labels_data, num_classes)

    # retrain model
    background_tasks.add_task(
        retrain_model,
        train_img,
        labels_data
    )


    return {"message": "Training started"}

    #return RedirectResponse(url="/", status_code=303)

@app.get("/training-status")
def get_training_status():
    return training_status




from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Annotated
from pydantic import BaseModel
import base64
import tensorflow
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import os
import torch
from collections import Counter
from google.cloud import bigquery, storage
from google.oauth2 import service_account
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import psycopg2
from fastapi.responses import RedirectResponse
import torchvision
from torchvision import transforms

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from glob import glob


import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import medmnist
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import f1_score
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP
import seaborn as sns


conn = psycopg2.connect(
    dbname="sampledb",
    user="app",
    password="pOud4unh16k5Xp9b1HE754U2",
    host="absolutely-verified-stag.a1.pgedge.io",
    port="5432"
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


UPLOAD_FOLDER='static'



class LNN (nn.Module):
    def __init__(self, ncp_input_size, hidden_size, num_classes, sequence_length):
        super(LNN, self).__init__()

        self.hidden_size = hidden_size
        self.ncp_input_size = ncp_input_size
        self.sequence_length = sequence_length

        ### CNN HEAD
        self.conv1 =  nn.Conv2d(1,16,3) # in channels, output channels, kernel size
        self.conv2 =  nn.Conv2d(16,32,3, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 =  nn.Conv2d(32,64,5, padding=2, stride=2)
        self.conv4 =  nn.Conv2d(64,128,5, padding=2, stride = 2)
        self.bn4 = nn.BatchNorm2d(128)

        ### DESIGNED NCP architecture
        wiring = AutoNCP(hidden_size, num_classes)    # 234,034 parameters

        self.rnn = CfC(ncp_input_size, wiring)

    def forward(self, x, device):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2,2))

        ## RNN MODE
        x = x.view(-1, self.sequence_length, self.ncp_input_size)
        h0 = torch.zeros(x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]   # we have 28 outputs since each part of sequence generates an output. for classification, we only want the last one
        return out




def make_wiring_diagram(wiring, layout):
    sns.set_style("white")
    plt.figure(figsize=(12, 12))
    legend_handles = wiring.draw_graph(layout=layout,neuron_colors={"command": "tab:cyan"})
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()






class Item(BaseModel):
    image_Path : str | None = None

@app.get("/")
async def dynamic_file(request: Request):
    path = "No Image Uploaded Yet"
    prediction = [[0]]
    return templates.TemplateResponse("index.html", {"request": request, "img_Path": path ,"probability": prediction})


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("open.html", {"request": request})

@app.get('/index')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/login')
def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get('/sign')
def sign(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get('/about')
def sign(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})




@app.post("/sign")
async def signup(
    request: Request, username: str = Form(...), email: str = Form(...),password1: str = Form(...),password2:str = Form(...) 
):
   
    cur = conn.cursor()
    cur.execute("INSERT INTO cancertb (uname,email,password1,password2) VALUES (%s, %s,%s, %s)", (username,email,password1,password2))
    conn.commit()
    cur.close() 
 
    return RedirectResponse("/login", status_code=303)


@app.post("/login",response_class=HTMLResponse)
async def do_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    cur = conn.cursor()
    cur.execute("SELECT * FROM cancertb WHERE uname=%s and password1=%s", (username,password))
    existing_user = cur.fetchone()
    cur.close()
    
    print(username)
    print(password)
    if existing_user:
        print(existing_user)
        return templates.TemplateResponse("index.html",{"request": request, "username": username, "password": password,"existing_user": existing_user})
    
    else:
        return HTMLResponse(status_code=401, content="Wrong credentials")




@app.post("/upload_image")
async def upload_image(request: Request, image_file: UploadFile = File(...)):
    # Save the uploaded image to the specified folder
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    with open(image_path, "wb") as f:
        content = await image_file.read()
        f.write(content)

    # Load the PyTorch model
    try:
        model_file = "cancer2.pt"
        bucket_name = "sandeep_personal"
        key_path = "ck-eams-9260619158c0.json"
        client = storage.Client.from_service_account_json(key_path)
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_file)
        blob.download_to_filename(model_file)
        model_state_dict = torch.load(model_file, map_location=torch.device('cpu'))  # Load model on CPU

    except Exception as e:
        return {"error": str(e)}
    

    HIDDEN_NEURONS = 19 # how many hidden neurons within LNN
    NUM_EPOCHS = 40
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 4
    NUM_OUTPUT_CLASSES = 2
    # Constants based off CNN architecture
    NCP_INPUT_SIZE = 16
    SEQUENCE_LENGTH = 32
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to match the input size of the model
        transforms.ToTensor(),         # Convert to tensor format
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # 2. Load the Pretrained Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Ensure device compatibility
    
    # Now you can instantiate your model and load the state_dict
    model = LNN(NCP_INPUT_SIZE, HIDDEN_NEURONS, NUM_OUTPUT_CLASSES, SEQUENCE_LENGTH)
    model.load_state_dict(model_state_dict)  # Load pretrained model weights
    model.eval()

    # 3. Run Inference on the Single Image
    with torch.no_grad():
        image_tensor = image_tensor.to(device)  # Move tensor to device
        output = model(image_tensor, device)  # Pass device to the forward method
        prediction = torch.argmax(output, dim=1).item()

    # 4. Interpret the Prediction
    predicted_class=""
    if prediction == 0:
        predicted_class=predicted_class+"The image depicts the cancer."
    else:
        predicted_class=predicted_class+"The image does not depicts cancer."


    # Prepare response data
    context = {
        "request": request,
        "predicted_class": predicted_class,
        "path":image_path
    }

    # Render an HTML template with the prediction result
    return templates.TemplateResponse("result.html", context)

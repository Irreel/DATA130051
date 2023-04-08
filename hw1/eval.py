import json
import pickle
import random
import numpy as np
from modules import NNClassifier, NNClassifierLoss, DIM_HID
from utils import MNISTDataLoader

random.seed(52)

# Read hyperparameters
with open('data.json', 'r') as f:
    params = json.load(f)
    
lr = params["learning_rate"]
dim_hid = params["hidden_dim"]
lamda = params["reg_strength"]

# Initialize
dataloader = MNISTDataLoader()
model = NNClassifier(n_in=28*28, n_hid=dim_hid)

# Load the model after grid search
with open(f"./saved/model_gridsearch.pkl", "wb") as f:
    model = pickle.load(f)


    



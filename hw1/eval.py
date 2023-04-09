import json
import pickle
import random
import numpy as np
from train import BATCH_SIZE
from modules import NNClassifier, NNClassifierLoss
from utils import MNISTDataLoader, vis_model, vis_acc, vis_loss

random.seed(52)

# Read hyperparameters from the grid search result
with open('./saved/hparams_gridsearch.json', 'r') as f:
    params = json.load(f)
lr = params["learning_rate"]
dim_hid = params["hidden_dim"]
lamda = params["reg_strength"]

# Initialize
dataloader = MNISTDataLoader()
best_model = NNClassifier(n_in=28*28, n_hid=dim_hid)

# Load the model after grid search
with open(f"./saved/model_gridsearch.pkl", "wb") as f:
    best_model = pickle.load(f)

# testing
best_model.training_mode(False)
lossFunc = NNClassifierLoss(best_model, lamda, lr)
test_loss_ls = []
test_acc_ls = []
        
for x, y_label in dataloader.load(BATCH_SIZE, "test"):
        y = best_model.forward(x)
        loss = lossFunc(y, y_label)
        test_loss_ls.append(loss)
        pred_label = np.argmax(y, axis = 1)
        true_label = np.argmax(y_label, axis = 1)
        test_acc_ls.append(np.sum(pred_label == true_label) / len(true_label))
    
    # return np.mean(train_loss_ls), np.mean(valid_loss_ls), np.mean(valid_acc_ls)
    
# Visualize the model params
vis_model(best_model)
print(f"Loss is {np.mean(test_loss_ls)}")
print(f"Accuracy is {np.mean(test_acc_ls)}")


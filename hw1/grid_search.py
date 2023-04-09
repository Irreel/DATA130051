import numpy as np
from train import main
from modules import NNClassifier
from utils import save_checkpoint

# Grid Search
# only design grid search for the following hyperparams
LR_GRID = [1e-2, 1e-3, 1e-4, 1e-5]
HID_GRID = [196, 98, 49]
LAMBDA_GRID = [1, 0.5, 0.25] # regularization term

# keep the result
best_acc = 0.
best_lr = None
best_n_hid = None
best_lamda = None
best_model = None

for lr in LR_GRID:
    for dim_hid in HID_GRID:
        for lamda in LAMBDA_GRID:
            model = NNClassifier(n_in=28*28, n_hid=dim_hid)
            _, acc_avg, acc_ls, valid_loss_ls = main(model, lr_start=lr, reg_term=lamda)
            if acc_avg > best_acc:
                best_acc = acc_avg
                best_lr = lr
                best_n_hid = dim_hid
                best_lamda = lamda
                best_model = model
                
save_checkpoint(best_model, suffix="gridsearch", params={"learning_rate": best_lr, "hidden_dim": best_n_hid, 
                                        "reg_strength": best_lamda, "best_acc": best_acc})
            

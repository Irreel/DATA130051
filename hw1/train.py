import os
import time
import pathlib
import random
import numpy as np
from modules import *
from utils import MNISTDataLoader, save_checkpoint, vis_acc, vis_loss

# Hyperparameters
EPOCH = 500
BATCH_SIZE = 128
# for Decay strategy
DECAY_FACTOR = 0.25
DECAY_PER_EPOCH = 150
# Grid Search
L_RATE = 1e-4
DIM_HID = 98 
LAMBDA = 0.5 # regularization term

random.seed(52)

# Load data
dataloader = MNISTDataLoader()
# Initialize model
class_model = NNClassifier(n_in=28*28, n_hid=DIM_HID)


def training(epoch, learning_rate, reg_term, model):
    # Training
    model.training_mode(True)
    lossFunc = NNClassifierLoss(model, reg_term, learning_rate)
    train_loss_ls = []
    
    # SGD
    for x, y_label in dataloader.load(BATCH_SIZE, "train"):
        # Clean the middle computation result in the last epoch
        model.zero_grad()
        y = model.forward(x)
        
        loss = lossFunc(y, y_label)
        lossFunc.backward()
        
        train_loss_ls.append(loss)
        
    # Valid
    model.training_mode(False)
    valid_loss_ls = []
    valid_acc_ls = []
        
    for x, y_label in dataloader.load(BATCH_SIZE, "valid"):
        y = model.forward(x)
        loss = lossFunc(y, y_label)
        valid_loss_ls.append(loss)
        pred_label = np.argmax(y, axis = 1)
        true_label = np.argmax(y_label, axis = 1)
        valid_acc_ls.append(np.sum(pred_label == true_label) / len(true_label))
    
    return np.mean(train_loss_ls), np.mean(valid_loss_ls), np.mean(valid_acc_ls)


def main(model, epoch=EPOCH, lr_start=L_RATE, reg_term=LAMBDA):
    # initialize
    lr = lr_start
    val_loss_best = np.inf
    epoch_best = 0
    acc_best = 0
    acc_ls = []
    val_loss_ls = []
    train_loss_ls = []
    
    for i in range(epoch):
        # Decay strategy
        if i // DECAY_PER_EPOCH == 0: 
            lr *= DECAY_FACTOR
        train_loss, valid_loss, valid_acc = training(i, lr, reg_term, model)
        acc_ls.append(valid_acc)
        val_loss_ls.append(valid_loss)
        train_loss_ls.append(train_loss)
        if valid_loss <= val_loss_best and valid_acc >= acc_best:
            epoch_best = i
            acc_best = valid_acc
            val_loss_best = valid_loss
            model_best = model
    
    # Save model and parameters
    save_checkpoint(model_best, params={"learning_rate": lr, "hidden_dim": model.n_hid, 
                                        "reg_strength": reg_term, "best_epoch": epoch_best, 
                                        "best_acc": acc_best})
    
    print(f"Best epoch is {epoch_best} with valid loss {val_loss_best} and acc {acc_best}")
    
    return epoch_best, acc_best, acc_ls, train_loss_ls, val_loss_ls


if __name__ == "__main__":
    _, acc, acc_ls, train_loss_ls, val_loss_ls = main(class_model)
    vis_loss(train_loss_ls, val_loss_ls, EPOCH, save_flg=True)
    vis_acc(acc_ls, save_flg=True)
    
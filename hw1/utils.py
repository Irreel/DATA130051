import time
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
Save models
"""
def save_checkpoint(model, params = None, suffix = None, path="./saved"):
    # Get the current time
    curr_time = time.localtime()
    curr_time = time.strftime("%d %H:%M:%S", curr_time)
    if suffix:
        with open(path+f"/model_{suffix}.pkl", "wb") as f:
            pickle.dump(model, f)
        if params:
            with open(path+f"hparams_{suffix}.json", 'w') as f:
                json.dump(params, f)
    else:
        with open(path+f"/{curr_time}/model.pkl", "wb") as f:
            pickle.dump(model, f)
        if params:
            with open(path+f"/{curr_time}/hparams_{curr_time}.json", 'w') as f:
                json.dump(params, f)
    

"""
Visualization Utils
"""
def vis_image(data, batch_first=False, save_flg=False):
    if batch_first:
        # sample the first one
        data = data[0]
    plt.imshow(data, cmap=plt.get_cmap('gray'))
    plt.show()
    
def vis_loss(train_loss, valid_loss, epoch: int, save_flg=False):
    # # Create a figure and axis object
    # fig, ax = plt.subplots()

    # # Plot the curves
    # ax.plot(x, y1, label='')
    # ax.plot(x, y2, label='')

    plt.plot(train_loss, range(epoch), label='Training')
    plt.plot(valid_loss, range(epoch), label='Valid')   
    plt.xlabel('Epoch') 
    if save_flg: plt.savefig('loss.png')
    else: plt.show()
    
    
def vis_acc(acc_list, epoch: int, save_flg=False):
    plt.figure()
    plt.plot(range(epoch), acc_list)  
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if save_flg: plt.savefig('acc.png')
    else: plt.show()

"""
Data preprocess
"""
def one_hot_encoder(labels, num_classes=10):
    """
    One-hot encodes an array of labels.

    Parameters:
    labels (numpy.ndarray): Array of labels to be encoded.
    num_classes (int): Number of classes in the label set.

    Returns:
    numpy.ndarray: Array of one-hot encoded labels.
    """
    # Create an array of zeros with the same number of rows as the labels array, and
    # num_classes columns
    encoded = np.zeros((len(labels), num_classes))
    
    # Set the value of the corresponding column in each row to 1
    for i, label in enumerate(labels):
        encoded[i, label] = 1
        
    return encoded


"""
Data Loader Utils
"""
def read_images(filepath):
    with open(filepath, 'rb') as f:
        magic_number = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
        num_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
        rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
        cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
        # Read image data and reshape into 3D array (num_images x rows x cols)
        data = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return data
    
def read_labels(filename):
    with open(filename, 'rb') as f:
        magic_number = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
        num_labels = np.frombuffer(f.read(4), dtype='>i4')[0]
        # Read label data
        data = np.frombuffer(f.read(num_labels), dtype=np.uint8)
    return data

class MNISTDataLoader():
    def __init__(self):
        """_summary_
        Read data from local file, then encode labels with one-hot encoder
        
        NOTE: Valid dataset is splited from the training file!
        """
        # Read from file
        self.label = read_labels('../data/train-labels-idx1-ubyte')
        self.image = read_images('../data/train-images-idx3-ubyte')
        # Split into train data and valid data
        self.train_label, self.valid_label = one_hot_encoder(self.label[:-10000]), one_hot_encoder(self.label[-10000:])
        self.train_image, self.valid_image = self.image[:-10000], self.image[-10000:]
        print(f"Loaded training images {self.train_image.shape} and labels {self.train_label.shape}")
        print(f"Loaded valid images {self.valid_image.shape} and labels {self.valid_label.shape}")
        
        # Read test data from file
        self.test_label = one_hot_encoder(read_labels('../data/t10k-labels-idx1-ubyte'))
        self.test_image = read_images('../data/t10k-images-idx3-ubyte')
        print(f"Loaded testing images {self.test_image.shape} and labels {self.test_label.shape}")
        
    def load(self, batch_size, flg="train"):
        if flg == "train":
            n_batch = (self.train_image.shape[0] // batch_size) + 1
            for i in range(n_batch):
                idx_start = i * batch_size
                idx_end = (i+1) * batch_size - 1
                if idx_end < self.train_image.shape[0]:
                    yield self.train_image[idx_start:idx_end], self.train_label[idx_start:idx_end]
                else:
                    yield self.train_image[idx_start:], self.train_label[idx_start:]
            
        elif flg == "valid":
            n_batch = (self.valid_image.shape[0] // batch_size) + 1
            for i in range(n_batch):
                idx_start = i * batch_size
                idx_end = (i+1) * batch_size - 1
                if idx_end < self.valid_image.shape[0]:
                    yield self.valid_image[idx_start:idx_end], self.valid_label[idx_start:idx_end]
                else:
                    yield self.valid_image[idx_start:], self.valid_label[idx_start:]
            
        elif flg == "test":
            n_batch = (self.test_image.shape[0] // batch_size) + 1
            for i in range(n_batch):
                idx_start = i * batch_size
                idx_end = (i+1) * batch_size - 1
                if idx_end < self.test_image.shape[0]:
                    yield self.test_image[idx_start:idx_end], self.test_label[idx_start:idx_end]
                else:
                    yield self.test_image[idx_start:], self.test_label[idx_start:]
        else:
            print("Invalid flg!")
            raise Exception    
    
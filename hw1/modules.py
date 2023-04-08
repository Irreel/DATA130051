import pickle
import numpy as np

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NNFC():
    def __init__(self, n_in, n_out, bias=True):
        self.n_in = n_in
        self.n_out = n_out
        self.w = np.ones((n_in, n_out))
        self.bias = bias
        if self.bias:
            self.b = np.zeros((n_out))
        # Flag for training or testing
        self._training = False
        self.initialization()
        # Clean computation results
        self.zero_grad()
            
    def initialization(self):
        """Xavier Initialization"""
        scale_factor = np.sqrt(1 / self.n_in)
        self.w = np.random.randn(self.n_in, self.n_out) * scale_factor
        return
        
    def zero_grad(self):
        self.X = None
        self.Y = None
        
    def forward(self, x):
        b_s = x.shape[0] # batch size
        y = x @ self.w
        y += np.array(b_s * [self.b])
        if self._training:
            # Keep the result
            self.X = x
            self.Y = y
        return y
        
    def backward(self, grad_y, lr, L2_scale = None):
        """_summary_
        Backward and SGD Optimizer
        Args:
            grad_y (_type_): the gradient of loss with respect to layer's output y
            lr (_type_): learning rate
            L2_scale (int, optional): parameter of L2 regularization. Defaults to None.

        Returns:
            grad_x: the gradient of loss with respect to layer's input x
        """
        assert self.X is not None
        
        # Calculate grad_x
        grad_x = grad_y @ self.w.T
        
        if L2_scale: 
            grad_w = self.X.T @ grad_y + 2 * L2_scale * self.w
        else:
            grad_w = self.X.T @ grad_y
        self.w -= lr * grad_w
        if self.bias:
            # grad_b = np.ones_like(self.b).T @ grad_y
            grad_b = np.mean(grad_y, axis=0) 
            self.b -= lr * grad_b
        # return grad_w, grad_b if self.bias else grad_w, None
        return grad_x
    

class NNClassifier():
    def __init__(self, n_in, n_hid, n_out=10):
        self.n_hid = n_hid
        self.FC1 = NNFC(n_in, n_hid)
        self.FC2 = NNFC(n_hid, n_out)
        # Activation function
        self.Sigmoid = Sigmoid
        
        self._training = False
        self.zero_grad()
        
    def training_mode(self, flg):
        """Whether the middle computation results should be saved
        """
        self._training = flg
        self.FC1._training = flg
        self.FC2._training = flg
            
    def zero_grad(self):
        """
        X_0 saves the input of the 1st block (FC + Ac)
        X_1 saves the input of the 2nd block (FC + Ac)  zAa
        """
        self.X_0 = None
        self.X_1 = None
        self.FC1.zero_grad()
        self.FC2.zero_grad()
    
    def forward(self, x):
        # Flatten 2D image
        b_s = x.shape[0]
        x_0 = x.reshape(b_s, -1)
        x_1 = self.Sigmoid(self.FC1.forward(x_0))
        y = self.FC2.forward(x_1)
        if self._training:
            self.X_0 = x_0
            self.X_1 = x_1
        return y
           
    def backward(self, grad_y, lr, lamda=None):
        assert self.FC1.Y is not None
        grad_x1 = self.FC2.backward(grad_y, lr, L2_scale = lamda)
        # Because grad_Sigmoid = lambda x: (1 - Sigmoid(x)) * Sigmoid(x)
        _ = self.FC1.backward( (self.X_1 * (1-self.X_1)) * grad_x1, lr, L2_scale = lamda)
        return
    
    # def save(self, path="./model.npy"):
    #     # obj = pickle.dumps(self)
    #     # with open(path, "wb") as f:
    #     #     f.write(obj)
    #     np.save("./model.npy", self)

    # def load(path):
    #     obj = None
    #     with open(path, "rb") as f:
    #         try:
    #             obj = pickle.load(f)
    #         except:
    #             print("IOError")
    #     return obj
   

# MSELoss = lambda y, label_y: np.mean(np.linalg.norm(y - label_y, axis=-1)**2)
MSELoss = lambda y, label_y: np.mean(np.sum(np.square(y - label_y), axis=-1))

class NNClassifierLoss():
    def __init__(self, model, lamda, lr):
        """_summary_
        Args:
            model (NNClassifier): the training model
            lamda (_type_): L2 regularization params
            lr (_type_): learning rates
        """
        self.model = model
        self.lamda = lamda
        self.lr = lr
        self.Y = None
        self.target_Y = None
        self.batch_size = None
        
    def __call__(self, y, target_y):
        """Calculate MSE loss and L2 regularization term

        Args:
            y (_type_): _description_
            target_y (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.Y = y
        self.target_Y = target_y
        self.batch_size = y.shape[0]
        assert y.shape[0] == target_y.shape[0]
        
        # MSE loss with size average
        loss_mse = MSELoss(y, target_y)
        # print(f"loss_mse is {loss_mse}")
        
        # L2 Regularization
        reg_term = 0
        weights = [self.model.FC1.w, self.model.FC2.w]
        for weight in weights:
            # reg_term += np.linalg.norm(weight)**2
            reg_term += np.sum(np.square(weight))
            
        return 0.5 * loss_mse + self.lamda * reg_term 
    
    def backward(self):
        """Stochastic Gradient Descending
        """
        grad_y = (self.Y - self.target_Y) / self.batch_size
        self.model.backward(grad_y, self.lr, self.lamda)
        return
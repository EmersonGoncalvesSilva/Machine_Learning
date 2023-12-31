""" 
Neural Nerwork
==============

Neural Network Algorithm.
Features:
---------
    - Activation Functions Options: 
        - `linear`: linear regression funtion
        - `sigmoid`: Logistic Regression function
        - `ReLU`: Rectified linear unit
    - Optimization Algorithm: `Gradient Descent`
    - Cost Function: `Log Loss` if output layer is `sigmoid`, `Mean Squared Error` otherwise.
    - About the parementers initialization(`weights` and `bias`):
        - Weights are initialized with value equal to 1. Bias are initalized with values equal to 0. 
        
Use cases:
------------
This Neural Network module can be used to train models for `regression` and `binary classification`.    
        
How to use:
-----------
    - Use class `NN_layer()` to build hidden layers for NN
    - Wire the layers in the model using the class `NN()`. e.g. `nn_model = NN(layer_01, layer_02, layer_03)`
    - train the model using `NN()` class `.train()` method. e.g. `nn_model.train(features, labels, alpha, )`

"""



import numpy as np

# Class for a layer:

class NN_Layer():
    """ 
    - Summary: Class used to create a hidden layer for a Neural Net.

    - Args:
        - layer (str): Name of the layer, e.g `'layer_01'`
        - units (int): Number of perceptrons in the hidden layer.
        - activation_func (str, optional): Activation function. Defaults to 'linear'.
    """
    def __init__(self, layer: str ,units: int, activation_func: str= 'linear') -> None:

        self.units = units
        self.act_func = activation_func
        self.layer = layer
        # Initialize the Attributes that will be computed in the NN Model:
        self.w = None
        self.b = None
        self.a = None
        self.dj_dw = None
        self.dj_db = None
        
        match activation_func:
            case 'sigmoid':
                self.g = self.Sigmoid.g
                self.dg_dz = self.Sigmoid.dg_dz
                self.cost = self.Sigmoid.cost
            case 'relu':
                self.g = self.Relu.g
                self.dg_dz = self.Relu.dg_dz
                self.cost = self.Relu.cost
            case 'linear':
                self.g = self.Linear.g
                self.dg_dz = self.Linear.dg_dz
                self.cost = self.Linear.cost
    

    class Sigmoid:
        @staticmethod
        def g(z):
            return 1/(1+np.exp(-z))
        @staticmethod
        def dg_dz(z):
            return (1/(1+np.exp(-z)))*(1- (1/(1+np.exp(-z))))
        @staticmethod
        def cost(y:np.array, y_hat:np.array, m):
            return sum(1/m)*(-y*np.log(y_hat) - (1-y)*np.log(1-y_hat))
    class Relu:
        @staticmethod
        def g(z: np.array):
            return np.maximum(0, z)
        @staticmethod
        def dg_dz(z: np.array):
            return np.where(z>0, 1, 0)
        @staticmethod
        def cost(y:np.array, y_hat:np.array, m):
            return sum((1/m)*(y - y_hat)**2)
    class Linear:
        @staticmethod
        def g(z):
            return z
        @staticmethod
        def dg_dz(z):
            return 1
        @staticmethod
        def cost(y:np.array, y_hat:np.array, m):
            return sum((1/m)*(y - y_hat)**2)
    def __str__(self) -> str:
        return f'Layer: {self.layer}'


class NN():
    """
    Neural Network Model
    --------------------
    - Use this class to create a neural net model. 
    - eg. `nn_model = NN( [NN_Layer('layer_01', 8), NN_Layer('layer_02', 8), ] ) `

    """
    def __init__(self, hidden_layers: list) -> None:
        # NN Architecture: 
        self.hidden_layers = hidden_layers
    ##

    def train(self, features: np.array, labels: np.array, alpha: float= .001, n_iterations:int=10000, e:float=1e-4, print_cost=True):
        """
        Training:
        --------

        - Args:
            - features (np.array): X training data. `2-D array - (dataset_size x Nr features)`
            - labels (np.array): Y training data `2-D array - (dataset_size x Nr labels)`
            - alpha (float, optional): Learning rate used to update W,B. Defaults to .001.
            - n_iterations (int, optional): Number of iterations. Defaults to 10000.
            - e (float, optional): Model convergency treshold . Defaults to 1e-4.
        """

        #01- Setting Paremeters:
        #=======================
        self.features = features
        self.labels = labels
        self.m = self.features.shape[0]
        self.alpha = alpha
        self.n_iterations = n_iterations

        #01.01- Initialize W and B Matrices:
        a_in_shape1 = self.features.shape[1]                                # Nr of features in the a_in matrix.
        for i, hidden_layer in enumerate(self.hidden_layers):
            # Parameters w, b:
            # b.shape(1, Nr_neurons)
            hidden_layer.b = np.zeros([1, hidden_layer.units])
            # Initialization using Xavier initialization techinique  
            hidden_layer.w = np.random.normal(loc=0.0, scale=a_in_shape1, size=(a_in_shape1, hidden_layer.units))  # w.shape(a_in.shape[1], Nr_neurons)
            a_in_shape1 = hidden_layer.units
            
            

        #02- LOOP FOR OPTIMIZATION - GRADIENT DESCENT:
        #=============================================
        cost_prev = 0 # to store the cost 
        for _ in range(0, self.n_iterations):


            #02.01 FORWARD PROPAGATION:
            #===========================
            a_in = self.features
            for hidden_layer in self.hidden_layers:
                hidden_layer.z = np.matmul(a_in, hidden_layer.w) + hidden_layer.b
                hidden_layer.a = hidden_layer.g(hidden_layer.z)                 
                a_in = hidden_layer.a           


            #02.02 BACKWARD PROPAGATION:
            #===========================

            #02.02.01 At the Last Hidden Layer:
            cost = self.hidden_layers[-1].cost(y=self.labels, y_hat=self.hidden_layers[-1].a, m= self.m)
            if _%1000 == 0:
                print(cost)
            if abs(cost - cost_prev) <= e:
                print('Optimization has converged!')
                print(f'Iteration: \n\t{_}')
                print(f'Cost: \n\t{cost}')
                break
            cost_prev = cost

            
            da_dz = self.hidden_layers[-1].dg_dz(self.hidden_layers[-1].z)          # Activation function incorporated,
            dj = (1/self.m)*(self.hidden_layers[-1].a - self.labels)
            dj = dj*da_dz
            self.hidden_layers[-1].dj_db = np.sum(dj, axis=0).reshape(self.hidden_layers[-1].b.shape)  
            self.hidden_layers[-1].dj_dw = np.matmul(self.hidden_layers[-2].a.T, dj) 
            w_pass = self.hidden_layers[-1].w

            #02.02.02 Loop through the inner hidden layers:
            for hidden_layer in reversed(self.hidden_layers[1:-1]):  
                da_dz = hidden_layer.dg_dz(hidden_layer.z)                          # Activation function incorporated;
                dj = np.matmul(dj, w_pass.T)
                dj= dj*da_dz

                w_pass = hidden_layer.w 

                hidden_layer.dj_db = np.sum(dj, axis=0).reshape(hidden_layer.b.shape)  #  
                # Get index of the previous layer:
                i_prev_ly = self.hidden_layers.index(hidden_layer) - 1
                hidden_layer.dj_dw = np.matmul(self.hidden_layers[i_prev_ly].a.T, dj) 

            ##02.02.03 At the First Hidden Layer:
            da_dz = self.hidden_layers[0].dg_dz(self.hidden_layers[0].z)
            dj = np.matmul(dj, w_pass.T)
            dj = dj*da_dz
            self.hidden_layers[0].dj_db = np.sum(dj, axis=0).reshape(self.hidden_layers[0].b.shape)
            self.hidden_layers[0].dj_dw = np.matmul(self.features.T, dj)


            #02.03 UPDATE PARAMETERS:
            #=========================

            for hidden_layer in self.hidden_layers:
                hidden_layer.b = hidden_layer.b - self.alpha*hidden_layer.dj_db
                hidden_layer.w = hidden_layer.w - self.alpha*hidden_layer.dj_dw
    ##
        
    def predict(self, x: np.array):
        """_summary_

        Args:
            x (np.array): _description_

        Returns:
            _type_: _description_
        """
        a_input = x
        for hidden_layer in self.hidden_layers:
            z = np.matmul(a_input, hidden_layer.w) + hidden_layer.b
            # pass z in the init function:
            a = hidden_layer.g(z)
            #print(f'Shape of the array a: \n\t{hidden_layer.a.shape}') 
            a_input = a
        return a_input
    

##
##

# Testing:
if __name__ == '__main__':
    pass





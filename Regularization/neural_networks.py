""" 
Neural Network
==============

Neural Network Algorithm
Features:
---------
    - Activation Functions Options: 
        - `linear`: linear regression funtion
        - `sigmoid`: Logistic Regression function
        - `ReLU`: Rectified linear unit
    - Optimization Algorithm: 
        `Gradient Descent`
    - Regularization:
        `L2 Regularization`
    - Cost Function:
        `Log Loss` if output layer is `sigmoid`, `Mean Squared Error` otherwise.
    - Paremeters Initialization:
        `Xavier` Initialization of the W parameters. Bias(b) are initalized with values equal to 0. 
        
Use case:
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

    def __init__(self, layer: str, units: int, activation_func: str = 'linear') -> None:

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
                self.dj_dy_hat = self.Sigmoid.dj_dy_hat
            case 'relu':
                self.g = self.Relu.g
                self.dg_dz = self.Relu.dg_dz
                self.cost = self.Relu.cost
                self.dj_dy_hat = self.Relu.dj_dy_hat
            case 'linear':
                self.g = self.Linear.g
                self.dg_dz = self.Linear.dg_dz
                self.cost = self.Linear.cost
                self.dj_dy_hat = self.Linear.dj_dy_hat

    class Sigmoid:
        @staticmethod
        def g(z):
            return 1/(1+np.exp(-z))

        @staticmethod
        def dg_dz(z):
            g_z = 1/(1+np.exp(-z))
            return (g_z)*(1 - g_z)

        @staticmethod
        def cost(y: np.array, y_hat: np.array, m):
            return (1/(2*m))*np.sum(-y*np.log(y_hat) - (1-y)*np.log(1-y_hat))
        @staticmethod
        def dj_dy_hat(y: np.array, y_hat: np.array, m):
            return -(1/m)*(np.divide(y, y_hat) - np.divide(1-y, 1-y_hat))

    class Relu:
        @staticmethod
        def g(z: np.array):
            return np.maximum(0, z)

        @staticmethod
        def dg_dz(z: np.array):
            return np.where(z > 0, 1, 0)

        @staticmethod
        def cost(y: np.array, y_hat: np.array, m):
            return (1/(2*m))*np.sum((y - y_hat)**2)

        @staticmethod
        def dj_dy_hat(y: np.array, y_hat: np.array, m):
            return (1/m)*(y-y_hat)

    class Linear:
        @staticmethod
        def g(z):
            return z

        @staticmethod
        def dg_dz(z):
            return 1

        @staticmethod
        def cost(y: np.array, y_hat: np.array, m):
            return (1/(2*m))*np.sum((y - y_hat)**2)

        @staticmethod
        def dj_dy_hat(y: np.array, y_hat: np.array, m):
            return (1/m)*(y-y_hat)

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

    def train(self, features: np.array, labels: np.array, alpha: float = .001, n_iterations: int = 10000, e: float = 1e-4, print_cost=True, l2_regularization:bool=True, lambd:float=0.1):
        """
        Training:
        --------

        - Args:
            - features (np.array): X training data. `2-D array - (Nr_features X Nr_samples)`
            - labels (np.array): Y training data `2-D array - (Nr_labels X Nr_samples)`
            - alpha (float, optional): Learning rate used to update W,B. Defaults to .001.
            - n_iterations (int, optional): Number of iterations. Defaults to 10000.
            - e (float, optional): Model convergency treshold . Defaults to 1e-4.
        """

        #01- Setting Paremeters:
        #=======================
        self.features = features
        self.labels = labels
        self.m = self.features.shape[1]
        self.alpha = alpha
        self.n_iterations = n_iterations

        #01.01- Initialize W and b Paremeters:
        # Nr of features in the a_in matrix.
        a_in_shape1 = self.features.shape[0]
        for i, hidden_layer in enumerate(self.hidden_layers):
            # Parameters W, b:
            hidden_layer.b = np.zeros([hidden_layer.units, 1]) # b.shape(Nr_neurons, 1)
            # Initialization using Xavier initialization techinique
            hidden_layer.w = np.random.normal(loc=0.0, size=(hidden_layer.units, a_in_shape1))*np.sqrt(2./(a_in_shape1))  # w.shape(Nr_neurons, a_in.shape[1])
            a_in_shape1 = hidden_layer.units


        #02- LOOP FOR OPTIMIZATION - GRADIENT DESCENT:
        #=============================================
        cost_prev = 0  # to store the cost
        for _ in range(0, self.n_iterations):


            #02.01 FORWARD PROPAGATION:
            #===========================
            a_in = self.features
            for hidden_layer in self.hidden_layers:
                hidden_layer.z = np.matmul(hidden_layer.w, a_in) + hidden_layer.b
                hidden_layer.a = hidden_layer.g(hidden_layer.z)
                a_in = hidden_layer.a


            # COMPUTE COST:
            #==============           
            cost = self.hidden_layers[-1].cost(y=self.labels, y_hat=self.hidden_layers[-1].a, m=self.m)
            if l2_regularization:
                # Calculate regularization cost:
                reg_cost = 0
                for hidden_layer in self.hidden_layers:
                    reg_cost += np.sum(np.square(hidden_layer.w))
                reg_cost = (1/self.m)*(lambd/2)*reg_cost
                cost = cost + reg_cost

            if _ % 100 == 0:
                print(cost)
            if abs(cost - cost_prev) <= e:
                print('Optimization has converged!')
                print(f'Iteration: \n\t{_}')
                print(f'Cost: \n\t{cost}')
                break
            cost_prev = cost
            
            #02.02 BACKWARD PROPAGATION:
            #===========================
            #02.02.01 At the Last Hidden Layer:
            # Cost derivative in regard to y_hat:
            dj_dy_hat = self.hidden_layers[-1].dj_dy_hat(y=self.labels, y_hat=self.hidden_layers[-1].a, m=self.m)
            dj_da_previous= dj_dy_hat
            #
            #02.02.02 Loop through the inner hidden layers:
            for hidden_layer in reversed(self.hidden_layers[1:]):
                
                da_dz = hidden_layer.dg_dz(hidden_layer.z)
                dj_dz = dj_da_previous*da_dz
                
                # a- Calculate db:
                hidden_layer.dj_db = np.sum(dj_dz, axis=1, keepdims=True) # check axis in sum
                
                # b- Calculate dw:
                    # Get index of the previous layer:
                i_prev_ly = self.hidden_layers.index(hidden_layer) - 1
                hidden_layer.dj_dw = np.matmul(dj_dz, self.hidden_layers[i_prev_ly].a.T)
                if l2_regularization:
                    hidden_layer.dj_dw += (lambd/self.m)*hidden_layer.w
                
                # c- Calculate dj_da_previous:
                dj_da_previous = np.matmul(hidden_layer.w.T, dj_dz)


            ##02.02.03 At the First Hidden Layer:
            da_dz = self.hidden_layers[0].dg_dz(self.hidden_layers[0].z)
            dj_dz = dj_da_previous*da_dz
            self.hidden_layers[0].dj_db = np.sum(dj_dz, axis=1, keepdims=True)
            self.hidden_layers[0].dj_dw = np.matmul(dj_dz, self.features.T)
            if l2_regularization:
                self.hidden_layers[0].dj_dw += (lambd/self.m)*self.hidden_layers[0].w

            
            
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
            z = np.matmul(hidden_layer.w, a_input) + hidden_layer.b
            a = hidden_layer.g(z)
            a_input = a
        return a


##
##

# Testing:
if __name__ == '__main__':
    pass

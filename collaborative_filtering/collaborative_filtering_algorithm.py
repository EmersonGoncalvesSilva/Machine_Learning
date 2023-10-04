"""
This Module contains the implementation from scratch of `Collaborative Filtering Algorithm`.

Application:
    - This unsupervised ML Algorithm can be used for recommentaions tasks, e.g. Movies Recomendations.

Main Features:
    - Gradient Descent - Minimization of the loss function
    - Regularization - to prune the paremeters of the model - avoid overfiting.
    - 

"""


import numpy as np



class Col_filt:
    def __init__(self, nr_features_x: int) -> None:
        """
        `Collaborative Filtering Algorithm`

        Args:
            nr_features_x (int): number of features to be discovered in the optimization process.
        """
        self.nr_features_x = nr_features_x  # Number of features X to be discovered in the algorithm.
    ##
        
    def train(self, y: np.array, nr_iterations: int= 100, alpha: float = 1e-2, lbda: float = 1e-1, print_cost:bool=True) -> None:
        """
        Trains the Collaborative Filtering Algorithm

        Args:
        - y (np.array): Contains all the ratings given to the items(e.g ratings given to movie, video, restaurant, etc). Shape[(numbers of items) X(number of ratings given per item)]
        - nr_iterations (int): Number of iterations to optimiza the model using gradient descent. `Defaults to 100`.
        - alpha (float, optional): Learing Rate in the Gradient Descent Optimization. `Defaults to 1e-2`.
        - lbda (float, optional): Regularization coeficient. `Defaults to 1e-1`.
        """
        self.y = y.copy()
        self.nr_iterations = nr_iterations
        self.alpha = alpha
        self.lbda = lbda
        
        # Indexes:
        self.n_i = self.y.shape[0]  # Number of items rated.
        self.n_r = self.y.shape[1]  # Number of users who rated the items.
        
        # Mean Normalization - Calculate the Y Mean:
        self.y_mean = np.nanmean(self.y, axis=1)
        self.y_norm = self.y - self.y_mean.reshape(-1, 1) 


        ## Initiate Matrices:
        #01- X:
        # Matrix to store the features of each item.
        self.x= np.random.normal(loc=0.0, scale=self.n_i, size=(self.n_i, self.nr_features_x)) # SHAPE: Nr_items X Nr_features
        #02- W:
        # Matrix to store the weights of each user.
        self.w= np.random.normal(loc=0.0, scale=self.nr_features_x, size=(self.nr_features_x, self.n_r))# SHAPE: Nr_Users X Nr_features
        #03- b:
        # Matrix to store the Bias of each user
        self.b= np.zeros(shape=(1, self.n_r))  # SHAPE: 1 x Nr_Users

        r = self.y.copy()
        nan_index = np.isnan(r)
        r[nan_index] = 0
        r[~nan_index] = 1


        ### LOOP FOR OPTIMIZATION - Using Gradient Descent:
        for i in range(0, self.nr_iterations):

            ##01- Forward Function:
            #F(x) = W*X + B
            self.y_hat = (np.matmul(self.x, self.w) + self.b)*r

            # Cost Function:
            #  COST= W*X + B - Y + regularization_W + regularization_X
            j = (1/2)*(self.y_hat - self.y_norm)**2 + \
                    (self.lbda/2)*np.sum(self.w**2) + \
                    (self.lbda/2)*np.sum(self.x**2)
            if print_cost and i%100 == 0:
                print(np.nansum(j))

            ##02- Back-Propagation:
            
            dj = (self.y_hat - self.y_norm)
            
            dj_db = np.nansum(dj, axis=0)
            # Set NaN to Zero:
            index_nan = np.isnan(dj)
            dj[index_nan] = 0
            
            dj_dw = np.matmul(self.x.T, dj) + self.lbda*self.w 

            dj_dx = np.matmul(dj, self.w.T) + self.lbda*self.x

            ##03- Update the Paremeters:
            self.w= self.w - self.alpha*dj_dw
            self.b= self.b - self.alpha*dj_db
            self.x= self.x - self.alpha*dj_dx
    ##

    def predict(self, item_index: int, user_index: int) -> np.array:
        """
        Predicts the Rating of the item(item_index) that the user(user_index) would give. 

        Args:
            item_index (int): Item index, (row of the dataset)
            user_index (int): User Index, (column of the dataset)

        Returns:
            np.array: Rating prediction
        """
        y_hat_i = np.matmul(self.x[item_index,], self.w[:, user_index]) + self.b[0, user_index]  + self.y_mean[item_index]
        
        return round(y_hat_i, 1)#np.ndarray.round(y_hat_i) 
###
###



# Testing:
if __name__ == '__main__':
    
    y = np.array([[3, 0, np.NAN, 1, 5],
                [np.NAN, np.NAN, 1, 1, 1],
                [3, 4, 5, 3, 5],
                [3, 0, np.NAN, 1, np.NAN]])

    col_filter = Col_filt(nr_features_x=5)

    col_filter.train(y=y, nr_iterations=100, alpha=.01, lbda=.5)


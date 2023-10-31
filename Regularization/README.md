# Regularization

Deep Learning are capable to be very flexible which makes their capacity of overfitting be a serious problem, specially when the training dataset is not big enough. 

When the model suffers from overfitting, it will do well on the training set, but the learned network won't generalize well on unseen data.

## L2 Regularization

The standard way to avoid overfitting is called **L2 regularization**. It consists of appropriately modifying the cost function as follow.

### Binary Classification problem - Cost Function:
```Math
J = -\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small  y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} \tag{1}
```
To:

$$
J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} \tag{2}
$$

### Regression problem  - Cost Function:

$$
J = \frac{1}{2m} \sum_{i=1}^{m} (a^{[L](i)}- y^{(i)})^2
$$

To:

$$
J_{regularized} = \small \underbrace{\frac{1}{2m} \sum_{i=1}^{m} (a^{[L](i)}- y^{(i)})^2 }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} \tag{2}
$$



### Backward Propagation with Regularization
As the cost function was changed, the backward propagation must be changed as as well. The gradients must take into consideration the new cost function. 


To implement the changes needed for backward propagation to take into account regularization, only dW calculation will be modified. dw/db=0, thus nothing need to be changed in db.

Not much need to be changed in dw, but the addition to the usual dw equation, of the following element:

$
\frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m}  W^2) = \frac{\lambda}{m} W
$
    

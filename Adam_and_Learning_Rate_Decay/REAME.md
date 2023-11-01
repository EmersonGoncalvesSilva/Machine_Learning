# Adam Algorithm and Learning Rate Decay

## Adam Algorithm
Adam stands as one of the most efficient optimization algorithms for training neural networks. It uses concepts from RMSProp, and Momentum. 
Here's a breakdown of how Adam operates:
- 1- It computes an exponentially weighted average of past gradients and preserves this information in the variables 𝑣 (before bias correction) and 𝑣𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑒𝑑 (with bias correction).
- 2- It calculates an exponentially weighted average of the squared magnitudes of past gradients, which it stores in variables 𝑠 (before bias correction) and 𝑠𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑒𝑑 (with bias correction).

Adam then updates the parameters(w,b) by taking a direction that combines information from steps "1" and "2".

### Calculate the Momentum:

$$
v_{dW^{[l]}} = \frac{\beta_1 v_{dW^{[l]}} + (1 - \beta_1) \frac{\partial \mathcal{J} }{ \partial W^{[l]} }}{1 - (\beta_1)^t} \\

v_{db^{[l]}} = \frac{\beta_1 v_{db^{[l]}} + (1 - \beta_1) \frac{\partial \mathcal{J} }{ \partial b^{[l]} }}{1 - (\beta_1)^t} \\
$$



### Calculate the RMSprop:

$$
s_{dW^{[l]}} = \frac{\beta_2 s_{dW^{[l]}} + (1 - \beta_2) (\frac{\partial \mathcal{J} }{\partial W^{[l]} })^2}{1 - (\beta_2)^t} \\

s_{db^{[l]}} = \frac{\beta_2 s_{db^{[l]}} + (1 - \beta_2) (\frac{\partial \mathcal{J} }{\partial b^{[l]} })^2}{1 - (\beta_2)^t} \\
$$



### Update model parameters - Adam:
$$
W^{[l]} = W^{[l]} - \alpha \frac{v_{dW^{[l]}}}{\sqrt{s_{dW^{[l]}}} + \varepsilon}\\

b^{[l]} = b^{[l]} - \alpha \frac{v_{db^{[l]}}}{\sqrt{s_{db^{[l]}}} + \varepsilon}\\
$$

$\varepsilon$ is a very  small number placed in the denominador to avoid division by 0.

## Learning Rate Decay

In the initial stages of training, the model can make bigger updates on w, and b, but relying on a constant learning rate (alpha) can eventually lead the model to become trapped in wide oscillations that prevent it from converging effectively. However, by gradually decreasing the learning rate alpha as training progresses, the optimization process can adopt smaller, more deliberate steps that guide the model closer to the minimum. This concept is the fundamental principle behind learning rate decay.

Adaptative alpha:
$$\alpha = \frac{1}{1 + decayRate \times epochNumber} \alpha_{0}$$
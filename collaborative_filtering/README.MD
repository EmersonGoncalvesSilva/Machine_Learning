# Collaborative Filtering

Collaborative filtering is a popular technique in the field of recommendation systems that helps in making personalized recommendations to users based on their historical interactions and preferences.

This little project aims to show the main concepts and Python code implementation and the application of this model to make movies recommentaions.


## Main Concepts

Collaborative Filtering is a technique that learns about the underlining features of products and the underlining preference of their users.
The idea is that an information $X$ can be learned about the a product, and an information $W$ and $b$ can be learned about its user preference.

The preference of an $user^{(j)}$  towards a given $product^{(i)}$ is represented by the following model:
```math
\hat{y} =\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)}
```
The determination of the best values of $X$, $W$, and $b$ is given by the minimization of the cost function:

$$J({\mathbf{x}^{(0)},...,\mathbf{x}^{(n_m-1)},\mathbf{w}^{(0)},b^{(0)},...,\mathbf{w}^{(n_u-1)},b^{(n_u-1)}})= \left[ \frac{1}{2}\sum_{(i,j):r(i,j)=1}(\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)})^2 \right]
+ \underbrace{\left[
\frac{\lambda}{2}
\sum_{j=0}^{n_u-1}\sum_{k=0}^{n-1}(\mathbf{w}^{(j)}_k)^2
+ \frac{\lambda}{2}\sum_{i=0}^{n_m-1}\sum_{k=0}^{n-1}(\mathbf{x}_k^{(i)})^2
\right]}_{regularization}
\tag{1}$$
The first summation in (1) is "for all $i$, $j$ where $r(i,j)$ equals $1$" and could be written:

$$
= \left[ \frac{1}{2}\sum_{j=0}^{n_u-1} \sum_{i=0}^{n_m-1}r(i,j)*(\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)})^2 \right]
+\text{regularization}
$$
## Implementation of collaborative filtering
In the `collaborative_filtering_algorithm.py` file there is the implementation of collaborative filtering. It uses the cost fucntion described above and the cost is minimized using gradient descent.

## Collobarative filtering for Movies Prediction
Check out the `movies_recomendation.ipynb` notebook to see the application os collaborative filter for movies recommentation.

## Conclusion

Collaborative filtering is a powerful technique for building recommendation systems that can provide personalized recommendations to users. It leverages the collective wisdom of users to make accurate predictions about their preferences. 

---

Happy Recommending! 🎬🍿👍

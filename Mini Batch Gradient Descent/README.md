# Mini-Batch Gradient Descent

Deep Learning takes advantage of the large amount of data available nowdays. 
However, the larger is the training dataset the longer is the traning time. Some actions can help to speed up the training time.

In Mini-batch Gradient descent the training dataset is split into mini-batches of data which are faster to process and allow earlier updates on dw, db parameters.
This allows a training step to run faster, plus decreases the number of iterations necessary to achieve a good model performance. 
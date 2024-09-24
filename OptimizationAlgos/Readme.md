# Gradient Descent Optimization algorithms 

In deep learning, gradient descent (GD) is a key optimization algorithm used to minimize the loss function. There are three main variations in terms how training data is used to compute the gradient: 

* batch gradient descent 
* stochastic gradient descent (SGD)
* mini-batch gradient descent.

Also, following algorithms in terms of different update rules are considered:
* Vanilla gradient descent 
* Momentum based gradient descent
* Nesterov Accelerated Gradient descent
* AdaGrad
* RMSProp
* Adam 

AdaGrad, RMSProp and Adam are adaptive optimization algorithms. In practice, Adam with mini-batch gradient descent is widely used due to many advantages.

Notebooks `0404_GDAlgorithms-1554543997405.ipynb` and `0407_GDAlgorithms-1554564131435.ipynb` use a single sigmoid neuron whereas the notebook `0407_VectorisedGDAlgorithms-1554564131433.ipynb` uses a simple feedforward NN.

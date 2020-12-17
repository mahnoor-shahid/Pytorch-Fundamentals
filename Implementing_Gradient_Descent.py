# for performing mathematical operations
import numpy as np

# for plotting and visualizing data
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    
    # Training Data
    x_data = [1.0, 2.0, 3.0]
    y_data = [12.0, 17.0, 25.0]

    # random initial weights
    w1 = 1.0  
    learning_rate = 0.01

    # linear model function
    def linear_model(x, w1):
        return x * w1

    # loss function
    def loss(y, y_pred):
        return (y_pred - y)**2

    # gradient function
    def gradient(x, y, w1):  
        return 2 * x * (x * w1 - y)

    for epoch in range(10):
        for x, y in zip(x_data,y_data):
            
            # computing the predicted y value
            y_pred = linear_model(x, w1)
            print("The predicted value is y = {}".format(y_pred))

            # computing the loss
            l = round(loss(y, y_pred),2)
            print("The loss is = {}".format(l))

            # updating the weight
            w1 = round(w1 - (learning_rate * gradient(x,y, w1)),2)
            
        print("*******************************")
        print("Iteration {} with w = {} and loss {}".format(epoch+1, w1, l))

# for performing mathematical operations
import numpy as np

if __name__ == '__main__':

    # Training Data
    x_vals = [1.0, 2.0, 3.0]
    y_vals = [2.0, 4.0, 6.0]

    # initial random weights
    w1 = 0.5
    w2 = 0.5

    # learning rate initialization
    learning_rate = 0.01

    def linear_model(x, w1, w2):
        return (w2*x**2 + w1*x)

    def loss(y, y_pred):
        return (y-y_pred)**2

    def derivative_w1(x, y, w1, w2):
        return 2*x*(x*w1+y-w2*x**2)
    
    def derivative_w2(x, y, w1, w2):
        return -2*x**2*(-w2*x**2+y+w1*x)

    for epoch in range(10):
        for x, y in zip(x_vals, y_vals):

            # computing the predicted y value
            y_pred = linear_model(x, w1, w2)
            print("The predicted value is y = {}".format(y_pred))

            # computing the loss
            l = round(loss(y, y_pred),2)
            print("The loss is = {}".format(l))

            # updating the weights
            w1 = round(w1 - (learning_rate * derivative_w1(x, y, w1, w2)),2)
            w2 = round(w2 - (learning_rate * derivative_w2(x, y, w1, w2)),2)
            
        print("*******************************")
        print("Iteration {} with w1 = {} , w2 ={} and loss {}".format(epoch+1, w1, w2, l))



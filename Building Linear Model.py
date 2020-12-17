# for performing mathematical operations
import numpy as np

# for plotting and visualizing data
import matplotlib.pyplot as plt 

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0,8.0]
weights = np.arange(1.0,4.1,0.5)

def linear_model(x):
    ## function for linear model
    return x * w

def loss(y, y_pred):
    ## calculating squared error
    return (y_pred - y)**2

# List of weights and mse for each input
w_list = []
mse_list = []


for i, w in enumerate(weights):
    ## print the weights and initialize the loss sum
    print("\n\nw=", w)
    l_sum = 0

    for x, y in zip(x_data, y_data):
        # for each input and output, calculate y_hat
        # compute the total loss and add to the total error
        y_pred = linear_model(x)
        l = loss(y, y_pred)
        l_sum += l
        print("\t", x, y, y_pred, l)

    # now compute the mean squared error (mse) of each
    # aggregate the weight/mse from this run
    mse = l_sum / len(x_data)
    print("MSE=", mse)
    w_list.append(w)
    mse_list.append(mse)

# estimate the lowest mse by which weight
a = mse_list[0]
for ind, i in enumerate(mse_list):
    if(a>i):
        a = i
        w = w_list[ind]
print("The best weight found is {}".format(w))


# Plot it all
plt.plot(w_list, mse_list, color='red', lw="2", ls="solid", marker="o", markerfacecolor="purple", markersize="6", alpha=0.5)
plt.title('Weights vs Loss')
plt.grid()
plt.ylabel('Loss')
plt.xlabel('Weights')
plt.show()
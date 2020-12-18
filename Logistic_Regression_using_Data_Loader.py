import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Constructing the DataLoader
class diabetesDatasetLoader(Dataset):

    def __init__(self):
        """ loading and initializing dataset """
        data = np.loadtxt('.\diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, 0:-1])
        self.y_data = torch.from_numpy(data[:, [-1]])
    
    def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# instantiate diabetesDatasetLoader
dataset = diabetesDatasetLoader() 

# training data set
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

# Constructing the Model 
class Model(torch.nn.Module):
    def __init__(self):
        """ 
        instantiating three nn.Linear models 
        """
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. 
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

# our model
model = Model()

# Construct our loss function and an Optimizer. 
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

if __name__ == '__main__':
    # Training loop
    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(inputs)

            # Compute loss
            loss = criterion(y_pred, labels)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1} | Batch: {i+1} | Loss: {loss.item():.4f}')





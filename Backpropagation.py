import torch
import pdb

if __name__ == '__main':
    
    # Training Data
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]

    # Initial Weight
    w = torch.tensor([1.0], requires_grad=True)

    # our model forward pass
    def forward_pass(x,w):
        return x * w

    # Loss function
    def loss(y_pred, y_val):
        return (y_pred - y_val) ** 2

    # Training loop
    for epoch in range(10):
        for x_val, y_val in zip(x_data, y_data):
            y_pred = forward_pass(x_val) 
            l = loss(y_pred, y_val) 
            l.backward() 
            print("\tgrad: ", x_val, y_val, w.grad.item())
            w.data = w.data - 0.01 * w.grad.item()

            # Manually zero the gradients after updating weights
            w.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}")
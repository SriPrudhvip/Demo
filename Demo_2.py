'''
In this ICE, we will explore l2 regularization and gradient clipping.
Name your file reg.py when submitting.
'''

import torch
import torch.nn as nn
import torch.optim as optim

#########################################################################################
################################### L2 regularization ###################################
#########################################################################################

# we reuse the network in single perceptron chapter
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1, 1) 

    def forward(self, x):
        out = self.fc(x)
        return out

net = Net()

# Setting weight decay for L2 regularization
l2_optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=0.005)

#########################################################################################
################################### Gradient Clipping ###################################
#########################################################################################

batch_size, dim_in, dim_h, dim_out = 128, 2000, 200, 20 
learning_rate = 1e-4

input_X = torch.randn(batch_size, dim_in)
output_Y = torch.randn(batch_size, dim_out)

model = torch.nn.Sequential(
    torch.nn.Linear(dim_in, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)

loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# we just update one step in our simple example
for values in range(1):
    optimizer.zero_grad()
    pred_y = model(input_X)
    loss = loss_fn(pred_y, output_Y)
    loss.backward()

    # Clipping gradients with max_norm of 5
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2.0)
    optimizer.step()

    # Check if the norm of gradient is indeed clipped
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(total_norm)

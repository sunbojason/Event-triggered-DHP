'''
@File        :system_model.py
@Description :
@Date        :2021/09/09 18:50:58
@Author      :Bo Sun
'''


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from system_dynamics import Sys_dynamics


sys_dynamics = Sys_dynamics() # instantiation


# training data    
num_data = 3000

u_b = 15*np.pi/180

n_state = 3
n_control = 1
dt = 0.001
n_nn = 10


x_b1 = 15*np.pi/180
x_b2 = 90*np.pi/180


uc_rand = np.random.uniform(-u_b,u_b,(num_data,n_control))
x1_rand = np.random.uniform(-x_b1,x_b1,(num_data,1))
x2_rand = np.random.uniform(-x_b2,x_b2,(num_data,1))
x3_rand = np.random.uniform(-u_b,u_b,(num_data,1))


x_rand = np.concatenate((x1_rand,x2_rand,x3_rand),axis = 1)
m_input_rand = np.concatenate((x_rand,uc_rand), axis= 1)


x_next_rand = np.zeros((num_data,n_state))

# print('uc_rand shape', uc_rand.shape)
# print('x_rand shape', x_rand.shape)
# print('m_input_rand shape', m_input_rand.shape)

for i in range(num_data):
    uc = uc_rand[i]
    x_in = x_rand[i]
    x_next_rand[i,:]= sys_dynamics.missile(x_in,uc,dt)


print(m_input_rand.shape, '\n\n', x_next_rand.shape)
print(m_input_rand[0])

# neural network
class Model(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_input, n_hidden, bias=False)
        # self.linear1.weight.data.uniform_(-1, 1)   # initialization
        self.linear2 = torch.nn.Linear(n_hidden, n_hidden, bias=False)
        # self.linear2.weight.data.uniform_(-1, 1)   # initialization
        self.linear3 = torch.nn.Linear(n_hidden, n_output, bias=False)
        # self.linear3.weight.data.uniform_(-0.1, 0.1)   # initialization

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        output = self.linear3(x)
        return output


model = Model(n_input=n_state+n_control, n_hidden=n_nn, n_output=n_state)


# training
learning_rate = 1e-2
num_epochs = 1000

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# data feed one by one
def train_loop(x_input, x_next_label, model, loss_fn, optimizer):
    for i in range(num_data):
        x = torch.tensor(x_input[i], dtype=torch.float32)
        y = torch.tensor(x_next_label[i], dtype=torch.float32)
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('loss:', loss.item())


# batch training (data size as batch size)
def train_loop_batch(x_input, x_next_label, model, loss_fn, optimizer, epoch):
    x = torch.tensor(x_input, dtype=torch.float32)
    y = torch.tensor(x_next_label, dtype=torch.float32)
    # Compute prediction and loss
    pred = model(x)
    loss = loss_fn(pred, y)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print('loss:', loss.item(),'epoch:', epoch)

# training loop
for epoch in range(num_epochs):
    train_loop_batch(m_input_rand, x_next_rand, model, loss_fn, optimizer,epoch)


# Testing
uc_rand = np.random.uniform(-u_b,u_b,(num_data,n_control))
x1_rand = np.random.uniform(-x_b1,x_b1,(num_data,1))
x2_rand = np.random.uniform(-x_b2,x_b2,(num_data,1))
x3_rand = np.random.uniform(-u_b,u_b,(num_data,1))


x_rand = np.concatenate((x1_rand,x2_rand,x3_rand),axis = 1)
m_input_rand = np.concatenate((x_rand,uc_rand), axis= 1)


x_next_rand = np.zeros((num_data,n_state))

# print('uc_rand shape', uc_rand.shape)
# print('x_rand shape', x_rand.shape)
# print('m_input_rand shape', m_input_rand.shape)

for i in range(num_data):
    uc = uc_rand[i]
    x_in = x_rand[i]
    x_next_rand[i,:]= sys_dynamics.missile(x_in,uc,dt)


m_input_tensor = torch.tensor(m_input_rand, dtype=torch.float32)
x_next_pred = model(m_input_tensor).detach().numpy()
print('test x_next_label: \n', x_next_rand[:5])
print('x_next prediction: \n', x_next_pred[:5])

rms = (np.mean((x_next_pred-x_next_rand)**2,axis=1))

print('mse: \n', np.size(rms))

idx_sample = np.linspace(1,num_data,num_data)



torch.save(model.state_dict(), "model.pkl")



# plt.figure()
# plt.plot(idx_sample, rms)
# plt.grid(axis='both')
# plt.show()





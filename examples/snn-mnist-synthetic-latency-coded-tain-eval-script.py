import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikegen
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
from torchvision import datasets, transforms


# Training Parameters
batch_size=128
data_path='/data/mnist'
num_classes = 10  # MNIST has 10 output classes
num_steps = 100
TAU = 5
THRESHOLD = 0.1
beta = 0.5
num_epochs = 5
num_iters = 50
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Torch Variables
dtype = torch.float
# neuron and simulation parameters
spike_grad = surrogate.atan()
fcn_hidden = 1000
net_type = 'fcn'


class SaveOutput:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        self.inputs.append(module_in)
        
    def clear(self):
        self.outputs = []
        self.inputs = []


def forward_pass(net, data):  
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        if net_type == 'cnn':
            spk_out, mem_out = net(data[step])
        elif net_type == 'fcn':
            m_batch_size = data.size(1)
            spk_out, mem_out = net(data[step].view(m_batch_size, -1))
        spk_rec.append(spk_out)
    
    return torch.stack(spk_rec)


def batch_accuracy(data_loader, net):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()
    
    data_loader = iter(data_loader)
    for data, targets in data_loader:
      data = data.to(device)
      targets = targets.to(device)
        
      data = spikegen.latency(data, num_steps=num_steps, tau=TAU, threshold=THRESHOLD, clip=True, normalize=True, linear=True)

      spk_rec = forward_pass(net, data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total


# Define a transform
transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)


THRESH_RANGE = [x /10 for x in range(1, 10)]
# print(THRESH_RANGE)

for THRESHOLD in THRESH_RANGE:

    #  Initialize Network
    # CNN
    if net_type == 'cnn':

        net = nn.Sequential(nn.Conv2d(1, 12, 5), #in=1x[32x32] out=12x[28x28]
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(2), #in=12x[28x28] out=12x[14x14]
                        nn.Conv2d(12, 32, 5),#in=12x[14x14] out=32x[10x10]
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(2), #in=32x[10x10] out=32x[5x5]
                        nn.Flatten(),
                        nn.Linear(32*5*5, 10), 
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)

    # FCN
    if net_type == 'fcn':
        net = nn.Sequential(nn.Linear(1024, fcn_hidden),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Linear(fcn_hidden, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
                        ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss()

    loss_hist = []
    acc_hist = []
    # '''
    net.train()

    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(train_loader)):
            data = data.to(device)
            data = spikegen.latency(data, num_steps=num_steps, tau=5, threshold=THRESHOLD, clip=True, normalize=True, linear=True)

            targets = targets.to(device)

            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
    
            # print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            acc = SF.accuracy_rate(spk_rec, targets) 
            acc_hist.append(acc)
            # print(f"Accuracy: {acc * 100:.2f}%\n")

    #         This will end train÷ing after 50 iterations by default
            if i == num_iters:
                break

    # print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
    # print(f"Accuracy: {acc * 100:.2f}%\n")

    PATH = f"{net_type}_snn_mnist_latency_tau_{TAU}_thresh_{str(THRESHOLD).replace('.','_')}_beta_{str(beta).replace('.','_')}_num_steps_{num_steps}.pt"
    # print(f"{acc * 100:.2f}, {loss_val.item():.2f}")
    # print(f'model saving to {PATH}')
    torch.save(net.state_dict(), PATH)

    # '''

    PATH = f"{net_type}_snn_mnist_latency_tau_{TAU}_thresh_{str(THRESHOLD).replace('.','_')}_beta_{str(beta).replace('.','_')}_num_steps_{num_steps}.pt"

    net.load_state_dict(torch.load(PATH))
    net.eval()

    test_acc = batch_accuracy(test_loader, net)
    # print(f"The total accuracy on the test set is: {test_acc * 100:.2f}%")

    save_output = SaveOutput()

    hook_handles = []

    for layer in net.modules():
        if isinstance(layer, snn.Leaky):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)

    save_output.clear()


    test_loader_batch_size_1 = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    for data, targets in test_loader_batch_size_1:
        data_for_counting_spkikes = spikegen.latency(data, num_steps=num_steps, tau=TAU, threshold=THRESHOLD, clip=True, normalize=True, linear=True)
        break

    data_for_counting_spkikes = data_for_counting_spkikes.to(device)
    spk_rec = forward_pass(net, data_for_counting_spkikes)

    # i = 0
    leaky1_none_zero_outputs = 0
    leaky2_none_zero_outputs = 0
    leaky3_none_zero_outputs = 0

    num_of_layers = 3 if net_type == 'cnn' else 2

    for i in range(0, len(save_output.outputs), num_of_layers):
        l1 = save_output.outputs[i]
        leaky1_none_zero_outputs += torch.count_nonzero(l1)
        if net_type == 'fcn':
            l2 = save_output.outputs[i+1][0]
            leaky2_none_zero_outputs += torch.count_nonzero(l2)
        if net_type == 'cnn':
            l2 = save_output.outputs[i+1]
            leaky2_none_zero_outputs += torch.count_nonzero(l2)

            l3 = save_output.outputs[i+2][0]
            leaky3_none_zero_outputs += torch.count_nonzero(l3)
        


    input_data_spikes = torch.count_nonzero(data_for_counting_spkikes)
    # print(f'input_data_spikes = {input_data_spikes}')
    # print(f'leaky1_none_zero_outputs {leaky1_none_zero_outputs}')
    # print(f'leaky2_none_zero_outputs {leaky2_none_zero_outputs}')
    # if net_type == 'cnn':

        # print(f'leaky3_none_zero_outputs {leaky3_none_zero_outputs}')
    print(f"{THRESHOLD}, {input_data_spikes}, {leaky1_none_zero_outputs}, {leaky2_none_zero_outputs}, {leaky3_none_zero_outputs}, {test_acc * 100:.2f}, {acc * 100:.2f}, {loss_val.item():.2f}")
    # print(f"{THRESHOLD}, {input_data_spikes}, {leaky1_none_zero_outputs}, {leaky2_none_zero_outputs}, {leaky3_none_zero_outputs}, {test_acc * 100:.2f}")

        



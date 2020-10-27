import torch.nn as nn
import numpy as np
import torch

class Net(nn.Module):
    """
    Input is normalized (x, y), output is RGB color at that point.
    """
    def __init__(self, num_hidden_layers=4, num_neurons=8):
        super(Net, self).__init__()
        
        # Input layer
        layers = [nn.Linear(2, num_neurons, bias=False), nn.Tanh()]
        
        # Hidden layers
        layers.extend(num_hidden_layers * [nn.Linear(num_neurons, num_neurons, bias=False), nn.Tanh(),])
            
        # Output layer
        layers.extend([nn.Linear(num_neurons, 3, bias=False), nn.Sigmoid()])
        
        self.layers = nn.Sequential(*layers) 
        
    def forward(self, x):
        x = self.layers(x)
        return x
    
def init_weights(m):
    """
    Ref: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=1)
        
def create_input(img_width, img_height):
    """
    Creates the input for the generative net. 
    Outputs numpy array w/ shape (img_width * img_height, 2)
    """
    # Create vectors of xs and ys   
    xs = np.linspace(start=-1, stop=1, num=img_width)
    ys = np.linspace(start=-1, stop=1, num=img_height)
    
    # Use np.meshgrid to create a mesh grid
    xv, yv = np.meshgrid(xs, ys)
    input_arr = np.stack((xv, yv), axis=2)
    input_arr = input_arr.reshape(img_width * img_height, 2)
    return input_arr

def create_net(num_hidden_layers=4, num_neurons=8):
    net = Net(num_hidden_layers=num_hidden_layers, num_neurons=num_neurons).double()
    net.apply(init_weights)
    return net

def generate_abstract_art(img_width, img_height, net):
    # Create input to net
    net_input = torch.tensor(create_input(img_width, img_height)).double()
    
    # Run input through net
    net_output = net(net_input).detach().numpy()
    
    # Reshape into (x, y, 3) for plotting
    net_output = net_output.reshape(320, 320, 3)
    
    # Re-format to color output
    # Scale to range 0 to 255, and set type to int
    net_output = (net_output * 255).astype(np.uint8)
    return net_output
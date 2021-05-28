import torch.nn as nn
import numpy as np
import torch
from typing import List, Dict, Any, NoReturn, Optional, Union
from collections import OrderedDict

# from mindist impor find_min_dist_mat
# Imports are a bit weird just to keep the required libraries
# to run this (as opposed to developing this) to a minimum


class Net(nn.Module):
    """
    Describes a generative CPPN that takes `x`, `y`, and optionally distance to origin as input,
    and outputs 3-channel / 1-channel pixel intensity.
    """

    def __init__(
        self,
        num_hidden_layers: int = 4,
        num_neurons: int = 8,
        latent_len: int = 3,
        include_bias: bool = True,
        include_dist_to_origin: bool = True,
        include_R: bool = False,
        rgb: bool = True,
    ) -> NoReturn:
        """
        Initializes the CPPN.

        Inputs
        ------
        num_hidden_layers: Number of hidden layers in the network.
        num_neurons: Number of neurons in each hidden layer.
        latent_len: Length of latent vector
        include_bias: If True, includes bias term in input layer.
        include_dist_to_origin: If True, includes distance to origin as one of the inputs.
        rgb: If True, produces 3-channel output. Else, produces 1-channel output.

        Output
        ------
        None
        """
        super(Net, self).__init__()

        # Run arch init
        self.layers: nn.Sequential = self.init_arch(
            num_hidden_layers=num_hidden_layers,
            num_neurons=num_neurons,
            latent_len=latent_len,
            include_bias=include_bias,
            include_dist_to_origin=include_dist_to_origin,
            include_R=include_R,
            rgb=rgb,
        )

        # Run weight init
        self.init_weights()

    def forward(self, loc_vec: torch.Tensor, latent_vec: torch.Tensor) -> torch.Tensor:
        """
        `forward` function for the generative network.

        Input
        -----
        loc_vec, latent_vec:
            Location vector and latent vector.
            Location vector should have shape (N, 2) or shape (N, 3).
            Latent vector should have shape (N, `latent_len`)

        Output
        ------
        x: torch.Tensor
        """
        x = torch.cat([loc_vec, latent_vec], dim=1)
        x = self.layers(x)
        return x

    def _init_weights(self, m) -> NoReturn:
        """
        Function to apply to the generative network (literally with `Net.apply()`) to initialize
        network weights properly. Required as the default initialization is for deep learning
        training, while we're only interested in starting all layers with a normal distribution.

        Ref: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

        Input
        -----
        m: various subclases in torch.nn.modules.* (I think?), called recursively
        using the `children.()` method.

        Output
        ------
        None
        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=1)

    def init_weights(self) -> NoReturn:
        """
        Initializes the weights of the network.
        Can also be used to re-init weights
        """
        self.apply(self._init_weights)

    def init_arch(
        self,
        num_hidden_layers,
        num_neurons,
        latent_len,
        include_bias,
        include_dist_to_origin,
        include_R,
        rgb,
    ) -> nn.Sequential:
        """
        Initializes the architecture of the network.
        """
        layers = OrderedDict()

        # Input dims
        # 2 base inputs per pixel: (x, y) on top of the latent vector
        input_dims = 2
        if include_dist_to_origin:
            input_dims += 1
        if include_R:
            input_dims += 1

        layers.update(
            {
                "fc0": nn.Linear(
                    input_dims + latent_len, num_neurons, bias=include_bias
                ),
                "tanh0": nn.Tanh(),
            }
        )

        # Hidden layers
        for i in range(num_hidden_layers):
            layers.update(
                {
                    "fc" + str(i + 1): nn.Linear(num_neurons, num_neurons, bias=False),
                    "tanh" + str(i + 1): nn.Tanh(),
                }
            )

        # Output layer
        if rgb:
            layers.update(
                {
                    "fc" + str(i + 1 + 1): nn.Linear(num_neurons, 3, bias=False),
                    "sigmoid": nn.Sigmoid(),
                }
            )
        else:
            layers.update(
                {
                    "fc" + str(i + 1 + 1): nn.Linear(num_neurons, 1, bias=False),
                    "sigmoid": nn.Sigmoid(),
                }
            )

        # Assign layers to self.layers
        return nn.Sequential(layers)


def create_input(
    img_width: int,
    img_height: int,
    include_dist_to_origin: bool = True,
    R: Optional[np.ndarray] = None,
    xs_start: float = -1,
    xs_stop: float = 1,
    ys_start: float = -1,
    ys_stop: float = 1,
) -> np.ndarray:
    """
    Creates the input for the generative net.

    Input
    -----
    img_width, img_height: int
    include_dist_to_origin: bool
    R is a reference array that is an extension of include_dist_to_origin

    Output
    ------
    input_arr: np.ndarray
        Should have shape (img_width * img_height, 2)
    """
    if R is not None:
        assert isinstance(R, np.ndarray)
        assert R.shape == (
            img_height,
            img_width,
        ), f"R.shape is {R.shape} which is not {(img_height, img_width)}"

    # Create vectors of xs and ys
    xs = np.linspace(start=xs_start, stop=xs_stop, num=img_width)
    ys = np.linspace(start=ys_start, stop=ys_stop, num=img_height)

    # Use np.meshgrid to create a mesh grid
    xv, yv = np.meshgrid(xs, ys)
    input_arr_grid = np.stack((xv, yv), axis=2)

    # Reshape input to NN as one row per pixel
    input_arr = input_arr_grid.reshape(img_width * img_height, 2)

    if include_dist_to_origin:
        dist_to_origin = np.sum(np.square(input_arr_grid), axis=2, keepdims=True)
        dist_to_origin = dist_to_origin.reshape(img_width * img_height, 1)
        input_arr = np.concatenate([input_arr, dist_to_origin], axis=1)

    # if R is not None:
    # Find min_dist_matrix from R, and reformat into correct input shape
    # min_dist_mat = mindist.find_min_dist_mat(R)
    # min_dist_mat = min_dist_mat.reshape(img_width * img_height, 1)
    # input_arr = np.concatenate([input_arr, min_dist_mat], axis=1)

    return input_arr


def generate_one_art(
    net: Net,
    latent_vec: torch.Tensor,
    net_input: Optional[np.ndarray] = None,
    input_config: Optional[Dict[str, Any]] = {"img_width": 320, "img_height": 320},
) -> np.ndarray:
    """
    Wrapper function to generate a single image output from the given network.

    Input
    -----
    net: Net
    latent_vec: torch.Tensor
    input_arr: output of `create_input`
    input_config: dict
        Dict of parameters to be passed to `create_input` as kwargs.

    TODO: Think about removing img_width and img_height from input_config

    Output
    ------
    net_output: np.ndarray
        Should have shape (y, x, 3) or (y, x, 1)
    """
    if net_input is None:
        # Create input to net, and convert from ndarray to torch.FloatTensor
        net_input = torch.tensor(create_input(**input_config)).float()
    else:
        net_input = torch.tensor(net_input).float()

    # Create input array from latent_vec, and convert from ndarray to torch.FloatTensor
    latent_vec = np.expand_dims(latent_vec, axis=0)
    latent_vec = np.repeat(latent_vec, repeats=net_input.shape[0], axis=0)
    latent_vec = torch.tensor(latent_vec).float()

    # This logic is wrong!
    # assert net_input.shape == latent_vec.shape, (
    #     "Shape of net_input is "
    #     f"{net_input.shape} while shape of latent_vec is {latent_vec.shape}"
    # )

    # Run input through net
    net_output = net(net_input, latent_vec).detach().numpy()

    # Reshape into (y, x, 3) for plotting in PIL
    net_output = net_output.reshape(
        input_config["img_height"], input_config["img_width"], -1
    )

    # Re-format to color output
    # Scale to range 0 to 255, and set type to int
    net_output = (net_output * 255).astype(np.uint8)
    return net_output


def generate_one_gallery(
    nets: Optional[Union[Net, List[Net]]],
    input_config: Dict[str, Any] = {"img_width": 320, "img_height": 320},
) -> NoReturn:
    """
    Plots grid of 40 images, accepting net_config as dict containing parameters to Net constructor.
    From these images, the ones in a column are the same network but have their latent vector perturbed.

    Input
    -----
    net_config: dict
        Dict of parameters to be passed to `Net()` as kwargs.

    input_config: dict
        Dict of parameters to be passed to `create_input` through `generate_one_art` as kwargs.

    Output
    ------
    None
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    # Wrap into list if not already
    if isinstance(nets, Net):
        nets = [nets]

    _, ax = plt.subplots(ncols=10, nrows=4, figsize=(16, 6))

    for net in nets:
        for i in range(10):
            net.init_weights()
            latent_vec = np.random.normal(size=(3,))

            for j in range(4):
                out = generate_one_art(
                    net, latent_vec=latent_vec, input_config=input_config
                )
                latent_vec += 0.2

                # if (n, n, 1) output i.e. if grayscale
                if out.shape[2] == 1:
                    img = Image.fromarray(
                        np.squeeze(out), mode="L"
                    )  # Mode inference is automatic but better be explicit
                    ax[j, i].imshow(img, cmap="gray")
                else:
                    img = Image.fromarray(out, mode="RGB")
                    ax[j, i].imshow(img, cmap="gray")

                ax[j, i].xaxis.set_visible(False)
                ax[j, i].yaxis.set_visible(False)

        plt.tight_layout()
        plt.show()

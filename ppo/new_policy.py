from stable_baselines3.common.torch_layers import MlpExtractor
from typing import Dict, List, Tuple, Type, Union
import torch as th
from torch import nn
from stable_baselines3.common.utils import get_device
from itertools import zip_longest

# Original code is from stable-baselines3
# Link to source: https://github.com/DLR-RM/stable-baselines3/blob/723b341c61d168e1460399592d5cebd4c6ef3cc8/stable_baselines3/common/torch_layers.py#L131

class CustomMlpExtractor(MlpExtractor):
    def __init__(
            self,
            feature_dim: int,
            net_arch: List[Union[int, Dict[str, List[int]]]],
            activation_fn: Type[nn.Module],
            device: Union[th.device, str] = "auto",
            mode:str = "normal"
    ):
        super(MlpExtractor, self).__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Added mode to switch between input representation
        if mode is "oen":
            shared_net = []
        elif mode is "attention":
            shared_net = []
        else:
            assert mode is "normal", "Error: mode can only be \"oen\", \"attention\" or \"normal\""

        # Iterate through the shared layers and build the shared parts of the network
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
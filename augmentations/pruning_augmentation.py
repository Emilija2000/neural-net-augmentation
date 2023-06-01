import torch
import numpy as np
import json

import haiku as hk
import jax
import functools
from jax import nn, random, image, jit, grad
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from haiku import initializers
from haiku._src.basic import Linear
from haiku._src.conv import Conv2D
from haiku._src.batch_norm import BatchNorm
from haiku._src.module import Module
from typing import Callable, Tuple
from haiku.initializers import Initializer, Constant, RandomNormal, TruncatedNormal, VarianceScaling

from optax import adam, rmsprop, sgd, softmax_cross_entropy, apply_updates

from scipy.stats import skew

import datasets


optimizer_dict = {
    "Adam": adam,
    "RMSProp": rmsprop,
    "MomentumSGD": functools.partial(sgd, momentum=0.9),
}

activation_dict = {
    "ReLU": nn.relu,
    "ELU": nn.elu,
    "Sigmoid": nn.sigmoid,
    "Tanh": nn.tanh,
}

initializer_dict = {
    "Constant": Constant(0.1),
    "RandomNormal": RandomNormal(),
    "GlorotUniform": VarianceScaling(1.0, "fan_avg", "uniform"),
    "GlorotNormal": VarianceScaling(1.0, "fan_avg", "truncated_normal"),
}

"""
dataset_dict = {
    "MNIST": datasets.load_dataset("mnist", split="train").with_format("jax"),
    "CIFAR-10": datasets.load_dataset("cifar10", split="train").with_format("jax").rename_column('img', 'image'),
    "SVHN": datasets.load_dataset("svhn", "cropped_digits", split="train").with_format("jax"),
    "Fashion-MNIST": datasets.load_dataset("fashion_mnist", split="train").with_format("jax"),  
}
"""



class CTCNet(hk.Module):
    def __init__(self,
                 n_classes: int,
                 activation: Callable = nn.relu,
                 w_init: Initializer = TruncatedNormal(),
                 kernel_size: Tuple[int, int] = (5, 5),
                 n_conv_layers: int = 3,
                 n_filters: int = 32,
                 n_fc_layers: int = 3,
                 fc_width: int = 128,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.n_classes = n_classes
        self.activation = activation
        self.w_init = w_init
        self.kernel_size = kernel_size
        self.n_conv_layers = n_conv_layers
        self.n_filters = n_filters
        self.n_fc_layers = n_fc_layers
        self.fc_width = fc_width
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        for _ in range(self.n_conv_layers):
            x = hk.Conv2D(output_channels=self.n_filters, kernel_shape=self.kernel_size,
                          padding="SAME", w_init=self.w_init)(x)
            x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training)
            x = self.activation(x)

        x = hk.Flatten()(x)

        for _ in range(self.n_fc_layers - 1):
            x = hk.Linear(self.fc_width, w_init=self.w_init)(x)
            x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training)
            x = self.activation(x)
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if is_training else x

        x = hk.Linear(self.n_classes, w_init=self.w_init)(x)
        return x


# Load the hyperparameters from the JSON file
# Here I am using the original path to weight zoo, to be changed depending on directory
with open('./transfer/0/run_data.json', 'r') as f:
    data = json.load(f)
hyperparameters = data['hyperparameters'] 


def net_fn(x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    return CTCNet(n_classes=10,
                  activation=activation_dict[hyperparameters['activation']],
                  w_init=initializer_dict[hyperparameters['initialization']],
                  n_conv_layers=3,
                  n_filters=32,
                  n_fc_layers=3,
                  fc_width=128,
                  dropout_rate=0.5)(x, is_training)

# Transform it into a haiku model
net = hk.transform(net_fn)

params = jnp.load('./transfer/0/epoch_20.npy', allow_pickle=True).item()
#print(params)

print("NETWORK ARCHITECTURE")
for layer, param_dict in params.items():
    for param_name, array in param_dict.items():
        print(f"{layer} {param_name} shape: {array.shape}")


        
def prune_weights_random_layer(params, prune_ratio):
    pruned_params = {}
    for layer_name, layer_params in params.items():
        pruned_layer_params = {}
        for param_name, param_values in layer_params.items():
            if param_name == 'w':  # only prune weights ('w')
                # Generate a mask of values to keep
                mask = jax.random.uniform(jax.random.PRNGKey(0), param_values.shape) > prune_ratio
                pruned_param_values = param_values * mask
                pruned_layer_params[param_name] = pruned_param_values
            else:
                pruned_layer_params[param_name] = param_values
        pruned_params[layer_name] = pruned_layer_params

    return pruned_params


def prune_weights_smallest_layer(params, prune_ratio):
    pruned_params = {}
    for layer_name, layer_params in params.items():
        pruned_layer_params = {}
        for param_name, param_values in layer_params.items():
            if param_name == 'w':  # only prune weights ('w')
                # Flatten the weights to 1D for easier processing
                flat_values = param_values.flatten()
                # Compute the threshold value below which weights will be pruned
                threshold = jnp.percentile(jnp.abs(flat_values), prune_ratio * 100)
                # Create a mask that is True for weights we want to keep
                mask = jnp.abs(param_values) > threshold
                pruned_param_values = param_values * mask
                pruned_layer_params[param_name] = pruned_param_values
            else:
                pruned_layer_params[param_name] = param_values
        pruned_params[layer_name] = pruned_layer_params

    return pruned_params


def prune_weights_smallest_global(params, prune_ratio):
    # Concatenate all weights into a single 1D array
    all_weights = jnp.concatenate([jnp.abs(param_dict['w']).flatten() for layer, param_dict in params.items() if 'w' in param_dict])

    # Compute the pruning threshold
    threshold = jnp.percentile(all_weights, prune_ratio * 100)

    # Prune the weights
    pruned_params = {}
    for layer, layer_param in params.items():
        pruned_layer = {}
        for k, v in layer_param.items():
            if 'w' in k:
                pruned_layer[k] = jnp.where(jnp.abs(v) < threshold, 0, v)
            else:
                pruned_layer[k] = v
        pruned_params[layer] = pruned_layer

    return pruned_params

#TESTED
def snr_pruning_global(params, prune_ratio):
    # Compute the SNR for all weights
    snr_values = {k: jnp.abs(param_dict['w'].mean()) / param_dict['w'].std() for k, param_dict in params.items() if 'w' in param_dict}

    # Flatten all SNR values into a single 1D array
    all_snr = jnp.concatenate([v.flatten() for v in snr_values.values()])

    # Compute the pruning threshold
    threshold = jnp.percentile(all_snr, prune_ratio * 100)

    # Prune the weights
    pruned_params = {}
    for layer, layer_param in params.items():
        pruned_layer = {}
        for k, v in layer_param.items():
            if 'w' in k:
                snr_layer = snr_values[layer]
                pruned_layer[k] = jnp.where(snr_layer < threshold, 0, v)
            else:
                pruned_layer[k] = v
        pruned_params[layer] = pruned_layer

    return pruned_params


#TESTED
def snr_pruning_layer(params, prune_ratio):
    pruned_params = {}
    for layer, param_dict in params.items():
        layer_param = {}
        for k, v in param_dict.items():
            if 'w' in k:
                # Compute the SNR for the weights of this layer
                snr_values = jnp.abs(v) / v.std()

                # Compute the pruning threshold for this layer
                threshold = jnp.percentile(snr_values, prune_ratio * 100)

                # Prune the weights of this layer
                layer_param[k] = jnp.where(snr_values < threshold, 0, v)
            else:
                layer_param[k] = v
        pruned_params[layer] = layer_param
    return pruned_params


# TESTED
def get_statistics(params):
    """ Instead of using the weights, we return 8 statistics for each layer (Classifying the classifier paper - feature-based meta-classification

        The features are specified from 8 different statistical measures of a weight vector: mean, variance, skewness (third standardized moment), 
        and five-number summary (1, 25, 50, 75 and 99 percentiles)."""

    stats_params = {}
    for layer_name, layer_params in params.items():
        stats_layer_params = {}
        for param_name, param_values in layer_params.items():
            if 'w' in param_name:  # only prune weights ('w')
                stats = [np.mean(param_values), np.std(param_values)**2, skew(param_values), np.percentile(param_values, 1), 
                np.percentile(param_values, 25), np.percentile(param_values, 50), np.percentile(param_values, 75), np.percentile(param_values, 99)]
                stats_layer_params[param_name] = stats
            else:
                stats_layer_params[param_name] = param_values
        stats_params[layer_name] = stats_layer_params


    return stats_params


"""
Maskig approach to pruning. This should still be considered as pruning but applied with a mask.
Essentially, you create a mask of the same shape as your weights. The mask contains 1s for weights you want to keep and 0s for weights you want to prune. 
Then, you simply multiply the weights with the mask. This has the effect of "zeroing out" the pruned weights.

In this example, we're creating a binary mask for each layer of weights, where values in the mask are set to 1 
if the corresponding weight is greater than the pruning threshold and 0 otherwise. 
"""

# TESTED
def create_mask(params, prune_ratio):
    mask = {}
    for layer, param_dict in params.items():
        sub_mask = {}
        for k, v in param_dict.items():
            if 'w' in k:
                # Compute the threshold
                threshold = jnp.percentile(jnp.abs(v), prune_ratio * 100)
                # Create a mask that is 1 where weights are greater than the threshold and 0 otherwise
                sub_mask[k] = jnp.where(jnp.abs(v) > threshold, 1, 0)
            else:
                # For biases just create a mask of ones
                sub_mask[k] = jnp.ones_like(v)
        mask[layer] = sub_mask
    return mask



def main():
    for i in range(1):
        params = jnp.load('./transfer/'+ str(i) + '/epoch_20.npy', allow_pickle=True).item()
        prune_ratio = 0.5  # 50% of the weights will be pruned
        pruned_params = prune_weights_random_layer(params, prune_ratio)

        # Using masking as a form of pruning
        mask = create_mask(params, prune_ratio) # Create a mask
        pruned_params = {k: v * mask[k] for k, v in params.items()} # Apply the mask

        jnp.save('./transfer/'+ str(i) + '/epoch_20_pruned.npy', pruned_params)



#main()
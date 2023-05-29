import chex
import jax
import numpy as np
import haiku as hk

from augmentations import prune_weights_random_layer, prune_weights_smallest_layer, prune_weights_smallest_global, get_statistics, create_mask

"""
STILL WORK IN PROGRESS
"""

def test_prune_weights_random_layer(filename):
    params = jnp.load(filename, allow_pickle=True).item()

    prune_ratio = 0.5  # 50% of the weights will be pruned
    pruned_params = prune_weights_random_layer(params, prune_ratio)

    # Check if shapes of all parameters are preserved
    chex.assert_trees_all_equal_shapes(params, pruned_params)
    
    # Check if around 50% of weights have been pruned
    total_weights = np.sum([np.prod(v.shape) for k, v in params.items() if 'w' in k])
    pruned_weights = np.sum([np.prod(v.shape) for k, v in pruned_params.items() if np.sum(v) == 0])
    np.testing.assert_almost_equal(pruned_weights / total_weights, prune_ratio, decimal=2)

    # Check if output has changed after pruning
    # TO DO


def test_prune_weights_smallest_layer(filename):
    params = jnp.load(filename, allow_pickle=True).item()

    prune_ratio = 0.5  # 50% of the weights will be pruned
    pruned_params = prune_weights_smallest_layer(params, prune_ratio)

    # Check if shapes of all parameters are preserved
    chex.assert_trees_all_equal_shapes(params, pruned_params)

    # Check layer by layer
    for layer_name, layer_params in pruned_params.items():
        for param_name, param_values in layer_params.items():
            if param_name == 'w':  
                total_zero_weights = np.sum(np.sum(param_values == 0))
                np.testing.assert_almost_equal(total_zero_weights / param_values.shape, prune_ratio, decimal=2)

    # Check if output has changed after pruning
    # TO DO


def prune_weights_smallest_global(filename):
    params = jnp.load(filename, allow_pickle=True).item()

    prune_ratio = 0.5  # 50% of the weights will be pruned
    pruned_params = prune_weights_random_layer(params, prune_ratio)

    # Check if shapes of all parameters are preserved
    chex.assert_trees_all_equal_shapes(params, pruned_params)
    
    # Check if around 50% of weights have been pruned
    total_weights = np.sum([np.prod(v.shape) for k, v in params.items() if 'w' in k])
    pruned_weights = np.sum([np.prod(v.shape) for k, v in pruned_params.items() if np.sum(v) == 0])
    np.testing.assert_almost_equal(pruned_weights / total_weights, prune_ratio, decimal=2)

    # Check if output has changed after pruning
    # TO DO


def test_get_statistics():
    # Create a test params dictionary compatible with the get_statistics function.
    # Here we have a network with two layers, each having weights 'w' and biases 'b'.
    params = {
        'layer1': {'w': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 'b': jnp.array([0.1, 0.2])},
        'layer2': {'w': jnp.array([6.0, 7.0, 8.0, 9.0, 10.0]), 'b': jnp.array([0.3, 0.4])}
    }

    stats_params = get_statistics(params)
    
    # Test that stats_params has the same keys as params.
    assert set(stats_params.keys()) == set(params.keys())

    # Test that for each layer, the stats_params contains the same keys as params
    for layer_name in params.keys():
        assert set(stats_params[layer_name].keys()) == set(params[layer_name].keys())

    # Test that the statistics for the weights are computed correctly.
    # Test that the biases are preserved.
    for layer_name, layer_params in params.items():
        for param_name, param_values in layer_params.items():
            if param_name == 'w':
                true_stats = [np.mean(param_values), np.std(param_values)**2, skew(param_values), np.percentile(param_values, 1), 
                              np.percentile(param_values, 25), np.percentile(param_values, 50), np.percentile(param_values, 75), 
                              np.percentile(param_values, 99)]
                computed_stats = stats_params[layer_name][param_name]
                chex.assert_trees_all_close(true_stats, computed_stats, atol=5e-06)
            else:
                true_value = param_values
                computed_value = stats_params[layer_name][param_name]
                chex.assert_trees_all_close(true_value, computed_value, atol=5e-06)


def test_create_mask():
    # Create a test params dictionary compatible with the create_mask function.
    params = {
        'w1': jnp.array([1.0, -2.0, 3.0, -4.0, 5.0]),
        'b1': jnp.array([0.1, -0.2]),
        'w2': jnp.array([-6.0, 7.0, -8.0, 9.0, -10.0]),
        'b2': jnp.array([-0.3, 0.4])
    }

    prune_ratio = 0.4
    mask = create_mask(params, prune_ratio)

    # Test that mask has the same keys as params.
    assert set(mask.keys()) == set(params.keys())

    # Test that the mask has the correct shape and contains only values 0 or 1.
    for k, v in mask.items():
        assert v.shape == params[k].shape
        assert jnp.all((v == 0) | (v == 1))

    # Test that the proportion of values in the mask for the weights is close to the expected proportion.
    for k, v in mask.items():
        if 'w' in k:
            proportion = jnp.mean(v)
            assert jnp.isclose(proportion, 1 - prune_ratio, atol=0.1)



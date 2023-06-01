import chex
import jax
import numpy as np
import haiku as hk
import jax.numpy as jnp

from scipy.stats import skew

from pruning import prune_weights_random_layer, prune_weights_smallest_layer, prune_weights_smallest_global, get_statistics, create_mask, snr_pruning_layer, snr_pruning_global

"""
STILL WORK IN PROGRESS
- need to add the code to test whether output has changed after pruning
"""

# TESTED AND WORKS
def test_prune_weights_random_layer():
    # Create a test params dictionary compatible with the prune_weights_smallest_global function.
    # Here we have a network with two layers, each having weights 'w' and biases 'b'.
    params = {
        'ctc_net/linear_1': {'w': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 'b': jnp.array([0.1, 0.2])},
        'ctc_net/linear_2': {'w': jnp.array([6.0, 7.0, 8.0, 9.0, 10.0]), 'b': jnp.array([0.3, 0.4])}
    }

    prune_ratio = 0.4
    pruned_params = prune_weights_random_layer(params, prune_ratio)
    print(pruned_params)

    # Test that pruned_params has the same keys as params.
    assert set(pruned_params.keys()) == set(params.keys())

    # Test that for each layer, the pruned_params contains the same keys as params
    for layer_name in params.keys():
        assert set(pruned_params[layer_name].keys()) == set(params[layer_name].keys())

    # Test that the pruned_params have the correct shape.
    for layer, layer_param in pruned_params.items():
        for k, v in layer_param.items():
            assert v.shape == params[layer][k].shape

    # Test that the proportion of pruned weights is close to the expected proportion.
    for layer, layer_param in pruned_params.items():
        for k, v in layer_param.items():
            if 'w' in k:
                proportion = jnp.mean(v == 0)
                assert jnp.isclose(proportion, prune_ratio, atol=0.1)

    # Check if output has changed after pruning
    # TO DO

# TESTED AND WORKS
def test_prune_weights_smallest_layer():
    # Create a test params dictionary compatible with the prune_weights_smallest_global function.
    # Here we have a network with two layers, each having weights 'w' and biases 'b'.
    params = {
        'ctc_net/linear_1': {'w': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 'b': jnp.array([0.1, 0.2])},
        'ctc_net/linear_2': {'w': jnp.array([6.0, 7.0, 8.0, 9.0, 10.0]), 'b': jnp.array([0.3, 0.4])}
    }

    prune_ratio = 0.4
    pruned_params = prune_weights_smallest_layer(params, prune_ratio)
    print(pruned_params)

    # Test that pruned_params has the same keys as params.
    assert set(pruned_params.keys()) == set(params.keys())

    # Test that for each layer, the pruned_params contains the same keys as params
    for layer_name in params.keys():
        assert set(pruned_params[layer_name].keys()) == set(params[layer_name].keys())

    # Test that the pruned_params have the correct shape.
    for layer, layer_param in pruned_params.items():
        for k, v in layer_param.items():
            assert v.shape == params[layer][k].shape

    # Test that the proportion of pruned weights is close to the expected proportion.
    for layer, layer_param in pruned_params.items():
        for k, v in layer_param.items():
            if 'w' in k:
                proportion = jnp.mean(v == 0)
                assert jnp.isclose(proportion, prune_ratio, atol=0.1)

    # Check if output has changed after pruning
    # TO DO

# TESTED AND WORKS
def test_prune_weights_smallest_global():
    # Create a test params dictionary compatible with the prune_weights_smallest_global function.
    # Here we have a network with two layers, each having weights 'w' and biases 'b'.
    params = {
        'ctc_net/linear_1': {'w': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 'b': jnp.array([0.1, 0.2])},
        'ctc_net/linear_2': {'w': jnp.array([6.0, 7.0, 8.0, 9.0, 10.0]), 'b': jnp.array([0.3, 0.4])}
    }

    prune_ratio = 0.4
    pruned_params = prune_weights_smallest_global(params, prune_ratio)
    print(pruned_params)

    # Test that pruned_params has the same keys as params.
    assert set(pruned_params.keys()) == set(params.keys())

    # Test that for each layer, the pruned_params contains the same keys as params
    for layer_name in params.keys():
        assert set(pruned_params[layer_name].keys()) == set(params[layer_name].keys())

    # Test that the pruned_params have the correct shape.
    for layer, layer_param in pruned_params.items():
        for k, v in layer_param.items():
            assert v.shape == params[layer][k].shape

    # Total number of weights before pruning
    total_weights = np.sum([np.prod(v.shape) for layer, layer_param in params.items() for k, v in layer_param.items() if 'w' in k])

    # Total number of pruned weights
    total_pruned_weights = np.sum([np.sum(v == 0) for layer, layer_param in pruned_params.items() for k, v in layer_param.items() if 'w' in k])

    # The number of pruned parameters should be roughly equal to the prune_ratio
    assert np.isclose(total_pruned_weights / total_weights, prune_ratio, atol=0.1)

    # Check if output has changed after pruning
    # TO DO


# TESTED AND WORKS
def test_snr_pruning_global():
    # Create a test params dictionary compatible with the get_statistics function.
    # Here we have a network with two layers, each having weights 'w' and biases 'b'.
    params = {
        'ctc_net/linear_1': {'w': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 'b': jnp.array([0.1, 0.2])},
        'ctc_net/linear_2': {'w': jnp.array([6.0, 7.0, 8.0, 9.0, 10.0]), 'b': jnp.array([0.3, 0.4])}
    }

    prune_ratio = 0.4
    pruned_params = snr_pruning_global(params, prune_ratio)

    # Test that pruned_params has the same keys as params.
    assert set(pruned_params.keys()) == set(params.keys())

    # Test that for each layer, the pruned_params contains the same keys as params
    for layer_name in params.keys():
        assert set(pruned_params[layer_name].keys()) == set(params[layer_name].keys())

    # Test that the pruned_params have the correct shape.
    for layer, layer_param in pruned_params.items():
        for k, v in layer_param.items():
            assert v.shape == params[layer][k].shape

    # Total number of weights before pruning
    total_weights = np.sum([np.prod(v.shape) for layer, layer_param in params.items() for k, v in layer_param.items() if 'w' in k])

    # Total number of pruned weights
    total_pruned_weights = np.sum([np.sum(v == 0) for layer, layer_param in pruned_params.items() for k, v in layer_param.items() if 'w' in k])

    # The number of pruned parameters should be roughly equal to the prune_ratio
    assert np.isclose(total_pruned_weights / total_weights, prune_ratio, atol=0.1)

    # Check if output has changed after pruning
    # TO DO


# TESTED AND WORKS
def test_snr_pruning_layer():
    # Create a test params dictionary compatible with the get_statistics function.
    # Here we have a network with two layers, each having weights 'w' and biases 'b'.
    params = {
        'ctc_net/linear_1': {'w': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 'b': jnp.array([0.1, 0.2])},
        'ctc_net/linear_2': {'w': jnp.array([6.0, 7.0, 8.0, 9.0, 10.0]), 'b': jnp.array([0.3, 0.4])}
    }

    prune_ratio = 0.4
    pruned_params = snr_pruning_layer(params, prune_ratio)
    print(pruned_params)

    # Test that pruned_params has the same keys as params.
    assert set(pruned_params.keys()) == set(params.keys())

    # Test that for each layer, the pruned_params contains the same keys as params
    for layer_name in params.keys():
        assert set(pruned_params[layer_name].keys()) == set(params[layer_name].keys())

    # Test that the pruned_params have the correct shape.
    for layer, layer_param in pruned_params.items():
        for k, v in layer_param.items():
            assert v.shape == params[layer][k].shape

    # Test that the proportion of pruned weights is close to the expected proportion.
    for layer, layer_param in pruned_params.items():
        for k, v in layer_param.items():
            if 'w' in k:
                proportion = jnp.mean(v == 0)
                assert jnp.isclose(proportion, prune_ratio, atol=0.1)

    # Check if output has changed after pruning
    # TO DO



# TESTED AND WORKS
def test_get_statistics():
    # Create a test params dictionary compatible with the get_statistics function.
    # Here we have a network with two layers, each having weights 'w' and biases 'b'.
    params = {
        'ctc_net/linear_1': {'w': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 'b': jnp.array([0.1, 0.2])},
        'ctc_net/linear_2': {'w': jnp.array([6.0, 7.0, 8.0, 9.0, 10.0]), 'b': jnp.array([0.3, 0.4])}
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
            if 'w' in param_name:
                true_stats = [np.mean(param_values), np.std(param_values)**2, skew(param_values), np.percentile(param_values, 1), 
                              np.percentile(param_values, 25), np.percentile(param_values, 50), np.percentile(param_values, 75), 
                              np.percentile(param_values, 99)]
                computed_stats = stats_params[layer_name][param_name]
                chex.assert_trees_all_close(true_stats, computed_stats, atol=5e-06)
            else:
                true_value = param_values
                computed_value = stats_params[layer_name][param_name]
                chex.assert_trees_all_close(true_value, computed_value, atol=5e-06)

# TESTED AND WORKS
def test_create_mask():
    # Create a test params dictionary compatible with the create_mask function.
    params = {'ctc_net/linear_1' : {
        'w1': jnp.array([1.0, -2.0, 3.0, -4.0, 5.0]),
        'b1': jnp.array([0.1, -0.2]),
        'w2': jnp.array([-6.0, 7.0, -8.0, 9.0, -10.0]),
        'b2': jnp.array([-0.3, 0.4])
    }}

    prune_ratio = 0.4
    mask = create_mask(params, prune_ratio)

    # Test that mask has the same keys as params.
    assert set(mask.keys()) == set(params.keys())

    # Test that for each layer, the mask contains the same keys as params
    for layer_name in params.keys():
        assert set(mask[layer_name].keys()) == set(params[layer_name].keys())

    # Test that mask has the same keys as params.
    assert set(mask.keys()) == set(params.keys())

    # Test that the mask has the correct shape and contains only values 0 or 1.
    for layer, param_dict in mask.items():
        for k, v in param_dict.items():
            assert v.shape == params[layer][k].shape
            assert jnp.all((v == 0) | (v == 1))

    # Test that the proportion of values in the mask for the weights is close to the expected proportion.
    for layer, param_dict in mask.items():
        for k, v in param_dict.items():
            if 'w' in k:
                proportion = jnp.mean(v)
                assert jnp.isclose(proportion, 1 - prune_ratio, atol=0.1)


def main():
    test_prune_weights_random_layer()

# main()
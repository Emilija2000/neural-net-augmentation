import haiku as hk
import jax
import jax.numpy as jnp
import chex

from augmentations import permute_batch, permute_checkpoint

def test_permutations():
    rng = jax.random.PRNGKey(42)
    net = hk.without_apply_rng(hk.transform(net_fn))
    rng,subkey = jax.random.split(rng)
    params = net.init(subkey, jnp.ones([28, 28, 3]))
    rng,subkey = jax.random.split(rng)
    layers = list(params.keys())[:-1]
    
    # keep original works
    permutations = permute_checkpoint(subkey, params, permute_layers=layers,num_permutations=2,keep_original=True)
    chex.assert_trees_all_close(params, permutations[0], atol=5e-06)
    
    # good dimensions
    chex.assert_trees_all_equal_shapes(*permutations)
    
    # perserves output
    num_permute=5
    permutations = permute_checkpoint(subkey, params, permute_layers=layers,num_permutations=5,keep_original=True)
    rng,subkey = jax.random.split(rng)
    example_in = jax.random.normal(subkey, (28,28,3))
    out = net.apply(params, example_in)
    trees = []
    for i in range(num_permute):
        trees.append(net.apply(permutations[i], example_in))
    chex.assert_trees_all_close(*trees, out, atol=5e-06)

def test_batched_permutations():
    rng = jax.random.PRNGKey(42)
    net = hk.without_apply_rng(hk.transform(net_fn))
    param_batch = []
    for i in range(10):
        rng,subkey = jax.random.split(rng)
        param_batch.append(net.init(subkey, jnp.ones([28, 28, 3])))
    layers = list(param_batch[0].keys())[:-1]
    
    # perserves original
    permutations = permute_batch(subkey,param_batch,permute_layers=layers,num_permutations=1,keep_original=True)
    for i in range(10):
        chex.assert_trees_all_close(permutations[2*i],param_batch[i],atol=5e-06)
    
    # perserves output
    permutations = permute_batch(subkey,param_batch,permute_layers=layers,num_permutations=3,keep_original=False)
    assert len(permutations) == 3*len(param_batch)
    
    rng,subkey = jax.random.split(rng)
    example_in = jax.random.normal(subkey, (28,28,3))
    
    for i in range(10):
        out = net.apply(param_batch[i], example_in)
        trees = []
        for j in range(3):
            trees.append(net.apply(permutations[i*3+j], example_in))
        chex.assert_trees_all_close(*trees, out,atol=5e-06)    
    
def net_fn(x):
    model = hk.Sequential([
        hk.Conv2D(output_channels=8, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(window_shape=(2, 2, 1), strides=(2, 2, 1), padding='SAME'),
        hk.Conv2D(output_channels=16, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(window_shape=(2, 2, 1), strides=(2, 2, 1), padding='SAME'),
        hk.Flatten(),
        hk.Linear(50),
        hk.Linear(10),
    ])
    return model(x)

if __name__ == '__main__':
    test_permutations()
    test_batched_permutations()
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # JAX Quick (semi) live demo
#
# This notebook is a short demo to show how some of JAX's common core features can used in a toy machine learning example.

# +
import os

# If using CPU
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4' # for pmap
os.environ["JAX_PLATFORMS"]="cpu"

import timeit
import jax
from jax import grad, vmap, jit, pmap
from jax.flatten_util import ravel_pytree
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
# -

# ## Gradients of 1D functions $f:\mathbb{R} \to \mathbb{R}$
#
# Using `jax.grad` on 1D functions `f` return its derivative as a Python function.

# +
xs = jnp.linspace(0, 2*jnp.pi, 50)
f = jnp.sin
fprime = grad(f)
fprimeprime = grad(fprime)

plt.plot(xs, jax.vmap(f)(xs), label=r"$f\:(x)$") # sin x
plt.plot(xs, jax.vmap(fprime)(xs), label=r"$f\:'(x)$") # cos x
plt.plot(xs, jax.vmap(fprimeprime)(xs), label=r"$f\:''(x)$") # -sin x
plt.legend(); plt.show()


# -

# ## Computing averages
#
# Many applications require generating random numbers/arrays and then computing their average. 
#
# As JAX requires a PRNG key for functions involving random numbers, one approach to compute the average would be to:
#
# 1. Write a function (`one_sample` below) that takes a key and returns one random number sample.
# 2. Use `jax.vmap` to map the function over multiple keys to get multiple samples, and then compute their mean.

# +
def mc_gaussian_mean(key, N, normal_mean=0, normal_stdev=1):
    # For illustrative purposes, can technically pass N to jrandom.normal
    keys = jrandom.split(key, N)
    one_sample = lambda k: normal_mean + normal_stdev * jrandom.normal(k)
    samples = jax.vmap(one_sample)(keys)
    return jnp.mean(samples)

key = jrandom.PRNGKey(0)
mc_gaussian_mean(key, N=100, normal_mean=10, normal_stdev=2)
# -

# ## PyTrees
#
# A PyTree is any tree-like structure built out of container-like Python objects.
#
# Below is an example of a PyTree, alongside how to use some higher order functions like map and reduce on them.

# +
tree = {
    "d": 100, 
    "e":[10,23,102], 
    "f":[{"a": 12, "b": (1,2)}, 
         {"c": 98}]}

# Map a unary function
jax.tree.map(lambda x: x+1000, tree)

# Element wise binary function on PyTree with same structure
jax.tree.map(lambda x, y: x + y**2, tree, tree)

# Product of all elements
jax.tree.reduce(lambda x,y: x*y, tree) 

# Another approach for the product reduce
flattened_tree, _ = jax.flatten_util.ravel_pytree(tree)
jnp.prod(flattened_tree)
# -

# ## Simple machine learning task with a neural network
#
# Now we will use core functionality from JAX for a simple regression problem. In practice, consider using libraries such as flax, optax etc.
#
# The task is to fit a regression function to the following data, which is the sin curve with some gaussian noise added to it.

# +
data_seed = 0

data_key = jrandom.PRNGKey(data_seed)
n_samples = 50
xs = jnp.linspace(0, 2 * jnp.pi, n_samples)
f = lambda x: jnp.sin(x)
ys = jax.vmap(f)(xs) + 0.1 * jrandom.normal(data_key, shape=(xs.size,))
plt.scatter(xs, ys)
plt.show()


# -

# We will use a neural network, specifically a Multi-Layer Perceptron (MLP), to represent our regression function.
#
# An MLP can have many hidden layers - below is the definition of a 1 hidden layer MLP.
#
# $$
# f_\theta(x) = \psi(W_2\:\sigma(W_1x + b_1) + b_2)
# $$
#
# An MLP with 2 hidden layers is defined as
#
# $$
# f_\theta(x) = \psi(W_3\:\sigma(W_2\:\sigma(W_1x + b_1) + b_2) + b3)
# $$
#
# and so on.
#
# - $\sigma$ is a element-wise function that makes $f_\theta$ non-linear, allowing MLPs to represent intricate functions. In this example we will be using $\sigma(x)=\max(0, x)$ which is provided by `jax.nn.relu`.
# - $\psi$ is an aggregation function which maps the final hidden layer to the output space. Since the output space is 1D, a common choice is to use the average, so we will use `jnp.mean`.
# - The matrices/vectors $W_i\,,b_i$ are called the weights/parameters, and have dimensions such that the matrix multiplications are valid.
# - $W_1x+b_1$ maps $x$ to a vector, whose dimension we will call `hidden_dim`. This vector gets subsequently multiplied by $W_i$ and added with $b_i$ for each $i>1$, throughout which we assume the dimension does not change.
# - $\theta$ is a placeholder for all the parameters in the network. In code, we will represent this as a Python list of tuples $(W_i\,,\:b_i)$.
#
# We will start by writing a function `neural_net_naive` that takes a an input value `x` list of tuples `params` representing the weights of the neural network.

def neural_net_naive(x, params, activation=jax.nn.relu):
    # 1D input, 1D output
    # params :: [(W_1, b_1), ..., (W_{d+1}, b_{d+1})], d=hidden_layers
    x = jnp.expand_dims(x, 0)

    *initial_layers, final_layer = params
    for W, b in initial_layers:
        x = activation(W @ x + b)

    final_W, final_b = final_layer
    return jnp.mean(final_W @ x + final_b)


# To use `neural_net_naive` we need the parameters $(W_i\,, b_i)$. How to best initialize them is a craft in itself, luckily for us JAX provides some commonly used approaches in `jax.nn.initiliazers`.

# +
def initialize_params(key, hidden_dim, hidden_layers):
    w_keys = jrandom.split(key, hidden_layers + 1) # + 1 for input layer
    b_keys =  jrandom.split(key, hidden_layers + 1)

    initializer = jax.nn.initializers.lecun_normal()

    # Initial layer need to map 1D input to hidden_dim space
    params =  [(initializer(w_keys[0], shape=(hidden_dim, 1)),
                initializer(b_keys[0], shape=(hidden_dim, 1)))]

    for i in range(hidden_layers):
        params.append((initializer(w_keys[i+1], shape=(hidden_dim, hidden_dim)), 
                       initializer(b_keys[i+1], shape=(hidden_dim, 1))))
    return params

init_param_seed = 2
init_param_key = jrandom.PRNGKey(init_param_seed)

params = initialize_params(init_param_key, hidden_dim=50, hidden_layers=5)
plt.plot(xs, jax.vmap(lambda x: neural_net_naive(x, params))(xs))
plt.show()


# -

# To find a good set of parameters that 'fit' the data, we use gradient based learning, namely, stochastic gradient descent. This involves defining a loss function which, for some input $x$, compares the predictions from our neural to the true corresponding output value $y$. For regression problems, the squared loss function is a commonly used option:
#
# $$
# L_\theta(x,y) = (y - f_\theta(x))^2
# $$
#
# We implement this as a Python function below.
#
# As the loss function will be called multiple times, it is a good idea to pass it to `jax.jit`. We can see the speed comparison below.

# +
def loss_unjitted(x, y_true, params):
    # loss function for 1 sample
    prediction = neural_net_naive(x, params)
    return (y_true - prediction) ** 2

loss = jax.jit(loss_unjitted)

uj_time = timeit.timeit(lambda: loss_unjitted(xs[0], ys[0], params), number=5_000)
j_time = timeit.timeit(lambda: loss(xs[0], ys[0], params), number=5_000)
print(f"With jax.jit: {j_time:.3f}, Without: {uj_time:.3f}")
# -

# To find a set of parameters that minimize the loss function, we will use stochastic gradient descent. It is an iterative approach, where at step $i$, the parameters are updated using the following rule:
#
# $$
# \theta_i \leftarrow \theta_{i-1} - \eta \nabla_\theta L_\theta |_{\theta_{i-1}}
# $$
#
# - $L_\theta = \frac{1}{B}\sum_{i} L_\theta(x_i\,, y_i)$ is the average loss over a uniformly randomly selected minibatch of the data of size $B$.
# - $\nabla_\theta L_\theta |_{\theta_{i-1}}$ refers to the gradient of $L_\theta$ with respect to $\theta$ evaluated at $\theta_{i-1}$.
# - $\eta$ is called the learning rate, which determines how much of the gradient information to use when updating the parameters.
#
# By treating the loss function in Python as a function of just the parameters, we can pass it to `jax.grad` which would provide the Python function for $\nabla_\theta L_\theta$. When evaluated at some value for the parameters, it will return a PyTree with the same structure as the parameters. We can then use `jax.tree.map` to update the parameters iteratively. Also consider using `jax.value_and_grad` in practice.
#
# As a side-note, the noise in the training loss curve below is mainly an artefact of using a random subset/minibatch of the dataset at each training step. This can be verified by using the whole dataset, via setting `batch_ids = jnp.arange(xs.size)`, which will reveal a smoother loss curve.

# +
sgd_seed = 1
sgd_key = jrandom.PRNGKey(sgd_seed)

# Initialize parameters
params = initialize_params(init_param_key, hidden_dim=50, hidden_layers=4)

# Training loop
steps = 10_001
losses = jnp.zeros((steps,))
for i in range(steps):
    # Get minibatch
    sgd_key, subk = jrandom.split(sgd_key)
    batch_ids = jrandom.randint(subk, minval=0, maxval=xs.size, shape=(10,))
    xs_batch, ys_batch = xs[batch_ids], ys[batch_ids]

    # Loss as a function of model parameters
    avg_loss = lambda ps: jnp.mean(jax.vmap(loss, in_axes=(0, 0, None))(xs_batch, ys_batch, ps)) 

    # Gradient of loss function
    grad_loss = jax.grad(avg_loss)
    grads = grad_loss(params)

    # Update parameters
    params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)
    
    losses = losses.at[i].set(avg_loss(params))
    if i%1000==0:
        plt.title(f"Step {i}, loss={losses[i]:.5f}")
        plt.xlabel(r"$x$"); plt.ylabel(r"$y$")
        plt.plot(xs, jax.vmap(lambda x: neural_net_naive(x, params))(xs), label=r"$f_\theta(x)$")
        plt.scatter(xs, ys, label="Training data")
        plt.legend(); plt.show()

# Plot training loss
plt.title(f"Training loss curve")
plt.xlabel("Step"); plt.ylabel("Loss value")
plt.plot(losses)
plt.show()


# -

# In practice we mostly care about the performance of the trained neural network above on unseen data, however we will skip this for now. 
#
# We will now show an example use case of `jax.pmap` - to train a simpler version of the already simple deep ensemble model. A deep ensemble consists of different neural networks trained on the same data, which can be done in parallel. 
#
# To do this, we first wrap the training loop in a separate Python function `fit`. Note that there are 2 sources of randomness: Parameter initialization, and selecting minibatches in each training step. We will use a single key for both instead of a separate key for each of them. We will also save hyperparamters related to the neural network architecture and stochastic gradient descent in a dictionary `hparams`.

# Fit neural network for 1D data
def fit(key, xs, ys, hparams):
    # NOTE Neural network function and loss function defined outside

    key, subk = jax.random.split(key)
    params = initialize_params(subk, hparams["hidden_dim"], hparams["hidden_layers"])

    for i in range(hparams["steps"]):
        # Get minibatch
        key, subk = jrandom.split(key)
        batch_ids = jrandom.randint(subk, minval=0, maxval=xs.size, shape=(hparams["minibatch_size"],))
        xs_batch, ys_batch = xs[batch_ids], ys[batch_ids]

        # Loss as a function of model parameters
        avg_loss = lambda ps: jnp.mean(jax.vmap(loss, in_axes=(0, 0, None))(xs_batch, ys_batch, ps)) 
    
        # Gradient of loss function
        grad_loss = jax.grad(avg_loss)
        grads = grad_loss(params)
    
        # Update parameters
        params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)

    return params


# To see whether the deep ensemble provides some good notion of uncertainty, we remove some samples from the original dataset. *Ideally*, the prediction of each neural network should be spread for the region of the input space where there is not much data.

mask = jnp.ones(xs.shape, dtype=bool).at[10:20].set(False)
exs = xs[mask]
eys = jax.vmap(f)(exs) + 0.1 * jrandom.normal(data_key, shape=(exs.size, ))
plt.scatter(exs, eys)
plt.show()

# +
hparams = {"hidden_dim": 20, 
           "hidden_layers": 3,
           "lr":0.1,
           "minibatch_size":10,
           "steps": 5_000}

n_ensembles = len(jax.devices())
ensemble_seed = 1
ensemble_keys = jrandom.split(jrandom.PRNGKey(ensemble_seed), n_ensembles)
ensembles = jax.pmap(lambda k: fit(k, exs, eys, hparams), devices=jax.devices('cpu'))(ensemble_keys)
# -

# We can check how the parameters are organized in `ensembles`, which is a PyTree.

jax.tree.map(lambda p: p.shape, ensembles)

# We see that the number of ensembles is in the leading dimension of each tuple containing $W_i\,,b_i$. We can use `jax.tree.map` to extract the parameters as needed, and plot the results as shown below.

# +
params_model_i = lambda i: jax.tree.map(lambda p: p[i, ...], ensembles)
ensemble_network_i = lambda i: lambda x: neural_net_naive(x, params_model_i(i))

for i in range(n_ensembles):
    # Can technically be done using vmap again
    plt.xlabel(r"$x$"); plt.ylabel(r"$y$")
    plt.plot(xs, jax.vmap(lambda x: ensemble_network_i(i)(x))(xs))
plt.scatter(exs, eys, label="(Full) Training data")
plt.legend(); plt.show()

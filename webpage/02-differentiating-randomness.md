---
layout: home
---

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
```

# Differentiating through randomness

## Fixed noise distribution

In the previous notebook (01-automatic-differentiation) we covered the basics of automatic differentiation (AD). However, we left an important detail for ABMs out. How can we compute the derivative of stochastic programs, and what does that mean, exactly?

Consider the following program with an input parameter $\theta$

$$

\begin{align*}
    z &\sim \mathcal N(1, 2) \\
    y &= \theta^2(z+4)
\end{align*}

$$

where $\mathcal N(\mu, \sigma)$ is the Normal distribution with mean $\mu$ and standard deviation $\sigma^2$. This can be coded as:


```python
def f(theta):
    z = np.random.normal(1, 2)
    return theta**2 * (z + 4)
```

So, what does it mean to compute

$$
\frac{\partial f(\theta)}{\partial \theta} =  \; ?
$$

If we think about the standard definition of derivative,

$$
\frac{\partial f(\theta)}{\partial \theta} =  \lim_{\epsilon \to 0} \frac{f(\theta+\epsilon) - f(\theta)}{\epsilon}.
$$

we can observe that this is not well defined since the difference $f(\theta+\epsilon) - f(\theta)$ will take random values and it is not clear how to treat this limit.

However, in many cases, we are actually interested in the derivative of an **average** model realization, not just the derivative of a single run. In other words, we want to take gradients of the form

$$
\mathbb E_{p(z)} [f_\theta(z)],
$$

where $p$ is a probability density and $f$ is a (deterministic) simulator with structural parameters $\theta$. Taking the derivative we have

$$
\begin{align*}
    \nabla_\theta \mathbb E_{p(z)}\;\left[f_\theta(z)\right] &= \nabla_\theta\left[\int_z p(z) f_\theta(z)\mathrm{d}z\right]\\
                                                             &=  \int_z p(z)\left[\nabla_\theta f_\theta(z)\right]\mathrm{d} z \\
                                                             &= \mathbb E_{p(z)}[\nabla_\theta f_\theta(z)]
\end{align*}
$$

so that the gradient of the expectation is the expectation of the gradient.

In our considered example, this translates to

$$
\mathbb E_{z \sim \mathcal N(1, 2)}\;\left[\theta^2(z+4)\right]
$$

now we can take the derivative

$$
\nabla_\theta \mathbb E_{z \sim \mathcal N(1, 2)}\;\left[\theta(z+4)\right] = \mathbb E_{z\sim \mathcal N(1,2)} \left[\nabla_\theta \theta^2(z+4)\right] = 10\, \theta
$$

Let's check this result numerically.


```python
theta = 2.0
epsilons = [1.0, 0.5, 0.1]
n_samples = 10_000
samples_per_epsilon = {epsilon: [] for epsilon in epsilons}
for epsilon in epsilons:
    for i in range(n_samples):
        f_deriv = (f(theta + epsilon) - f(theta)) / (epsilon)
        samples_per_epsilon[epsilon].append(f_deriv)
```


```python
fig, ax = plt.subplots()
for i, (epsilon, samples) in enumerate(samples_per_epsilon.items()):
    ax.hist(
        samples,
        bins=100,
        alpha=0.6,
        label=f"$\epsilon={epsilon}$",
        density=True,
        color=f"C{i}",
    )
ax.axvline(
    theta * 10,
    color="black",
    linestyle="dashed",
    linewidth=1,
    label="Expected derivative",
)
ax.legend()
```




    <matplotlib.legend.Legend at 0x13fcacac0>




    
![png](02-differentiating-randomness_files/02-differentiating-randomness_5_1.png)
    


What is going on? The smaller $\epsilon$ is, the variance of our estimator is going $\to \infty$! 

This is caused by the fact that the two function calls $X(p)$ and $X(p+\epsilon)$ are uncorrelated (they have a different random seed) and so even calling the program for very similar values of $p$ can cause large differences. 

Nonetheless, the finite differences method gives an unbiased estimator.

This issue is not present in automatic differentiation engines, since they only require one evaluation of the model.


```python
def f_torch(theta):
    z = torch.distributions.Normal(1.0, 2.0).sample()
    return theta**2 * (z + 4)


n_samples = 10_000
samples_autograd = []
for i in range(n_samples):
    theta = torch.tensor(2.0, requires_grad=True)
    f_torch(theta).backward()
    samples_autograd.append(theta.grad)
```


```python
fig, ax = plt.subplots()
ax.hist(
    samples_autograd, bins=100, label="Autograd", density=True, color="C0"
)
ax.axvline(
    10*theta.item(), color="black", linestyle="dashed", linewidth=1, label="Expected derivative"
)
ax.legend()
```




    <matplotlib.legend.Legend at 0x15c053820>




    
![png](02-differentiating-randomness_files/02-differentiating-randomness_8_1.png)
    


As we can see, the variance is much lower than the finite difference case.

## Parameterizable noise

In the example above, we made the assumption that the randomness of our model $p(z)$ did not depend on the structural parameters $\theta$ that we want to differentiate by.

However, in most agent-based models this assumption does not hold. Consider, for instance, the case of an SIR model where agents become infected with some probability $q$. This probability may depend on structural parameters such as $R_0$ or social distancing measures that we want to calibrate.

So now we have

$$
\tag{1}
\nabla_\theta\mathbb E_{p_\theta(z)} [f_\theta(z)],
$$

where $p$ is parameterized by $\theta$. Expanding the same way as before we find

$$
\begin{align*}
    \nabla_\theta\mathbb E_{p_\theta(z)} [f_\theta(z)] &= \nabla_\theta \left[ \int_z p_\theta(z) f_\theta(z) \mathrm{d}z\right]\\
                                                       &= \int_z \nabla_\theta\left[p_\theta(z) f_\theta(z)\right] \mathrm{d}z\\
                                                       &= \int_z f_\theta(z)\nabla_\theta p_\theta(z)\mathrm{d} z + \int_z p_\theta(z) \nabla_\theta f_\theta(z)\mathrm{d}z\\
                                                       &= \int_z f_\theta(z) \nabla_\theta p_\theta(z) \mathrm{d} z + \mathbb E_{p_\theta(z)} \left[\nabla_\theta f_\theta(z)\right]
\end{align*}
$$

notice that now we have an additional term, $\int_z f_\theta(z) \nabla_\theta p_\theta(z) \mathrm{d} z$, that prevents us from commuting the gradient and the expectation. So, in general, **the gradient of the expectation is not the expectation of the gradient** when our structural parameters parameterize the randomness.


### The reparameterization trick

To circumvent this, we introduce the [reparameterization trick](https://arxiv.org/pdf/1312.6114.pdf). Continuous distributions have a property that allows an indirect way of sampling from them. One is direct sampling

$$ x \sim p_\theta(x) $$

and indirectly,

$$ \epsilon \sim p(\epsilon), \hspace{1cm} x = g(\epsilon, \theta) $$

where $g$ is a deterministic path that maps parameter-free noise ($\epsilon$) to $x$. For instance, to sample from a Normal distribution $\mathcal N(\mu, \sigma^2)$, we can do

$$ \epsilon \sim \mathcal N(0, 1), \hspace{0.5cm} x = g_\theta(\epsilon) = \mu + \sigma \epsilon. $$

What have we accomplished with this? We have untangled the "random" part of the distribution from the part that depends on the parameters.
This allows us to re-formulate (1) as

$$
\nabla_\theta\mathbb E_{p_\theta(z)} [f_\theta(z)] = \nabla_\theta\mathbb E_{p(\epsilon)}\left[f(g_\theta(\epsilon, x))\right],
$$

where now we can move the gradient inside,

$$
\nabla_\theta\mathbb E_{p(\epsilon)}\left[f(g_\theta(\epsilon, x))\right] = \mathbb E_{p(\epsilon)}\left[\nabla_\theta f(g_\theta(\epsilon, x))\right].
$$

So we have been able to transform the gradient of the expectation to the expectation of the gradient. This can be computed provided that $f$ and $g$ are differentiable. This expectation can now be estiated by Monte Carlo,

$$
\mathbb E_{p(\epsilon)}\left[\nabla_\theta f(g_\theta(\epsilon, x))\right] \approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta f(g_\theta(\epsilon_i, x_i)).
$$

Let's see an example. Consider the program with input $\theta$

$$
\begin{align*}
z \sim \mathcal N(\theta, 4) \\
y = \theta^2 (4 + z)
\end{align*}
$$

The expected answer is

$$
\mathbb E_{\epsilon \sim \mathcal N(0,1)}\left[ \nabla_\theta \theta^2 (4 + \theta + 2\epsilon )\right ] =\mathbb E_{\epsilon \sim \mathcal N(0,1)}[8\theta + 3\theta^2 + 4\theta\epsilon] = 8\theta + 3\theta^2
$$

Let's first check both the parameterized and unreparameterized version:


```python
def f_torch_norep(theta):
    z = torch.distributions.Normal(theta, 2.0).sample()
    return theta**2 * (z + 4)


def f_torch(theta):
    epsilon = torch.distributions.Normal(0, 1).sample()
    z = theta + epsilon * 2
    return theta**2 * (4 + z)


rep_samples = []
norep_samples = []
n_samples = 2_000
theta_value = 2.0
analytical_result = 8 * theta_value + 3 * theta_value**2
for i in range(n_samples):
    theta = torch.tensor(theta_value, requires_grad=True)
    f_torch_norep(theta).backward()
    norep_samples.append(theta.grad.item())
    theta = torch.tensor(theta_value, requires_grad=True)
    f_torch(theta).backward()
    rep_samples.append(theta.grad.item())
```


```python
fig, ax = plt.subplots()
ax.hist(
    norep_samples,
    bins=100,
    alpha=0.35,
    label="No Reparametrization",
    density=True,
    color="C0",
)
ax.hist(
    rep_samples,
    bins=100,
    alpha=0.35,
    label="Reparametrization",
    density=True,
    color="C1",
)
ax.axvline(
    analytical_result,
    color="black",
    linestyle="dashed",
    linewidth=1,
    label="Expected derivative",
)
ax.legend()
```




    <matplotlib.legend.Legend at 0x14fdd8280>




    
![png](02-differentiating-randomness_files/02-differentiating-randomness_12_1.png)
    


As we can see, we need to be careful, because PyTorch may not be applying the reparameterization trick and we may obtain the wrong results!!

By calling `dist.sample()` the gradient propagation stops and so we obtain the wrong result. Thankfully, PyTorch has a shortcut to implement the reparameterization trick for us so we don't have to do it manually. This can be done by using the `.rsample()` method.


```python
def f_torch(theta):
    z = torch.distributions.Normal(theta, 2.0).rsample()
    return theta**2 * (z + 4)


rep = (
    sum([torch.func.grad(f_torch)(torch.tensor(2.0)) for i in range(n_samples)])
    / n_samples
)
print(f"{rep} ~ {analytical_result}")
```

    27.93507194519043 ~ 28.0


# A note on reinforcement learning

Recall equation (1):

$$
\nabla_\theta\mathbb E_{p_\theta(z)} [f_\theta(z)].
$$

There is an alternative way to re-express this gradient:

$$
\begin{split}
\nabla_\theta\mathbb E_{p_\theta(z)} [f_\theta(z)] &= \nabla_\theta \int p_\theta(z) f_\theta(z) \mathrm{d} z \\
&= \int f(z) \nabla_\theta p(z) \mathrm{d}z \\
&= \int p_\theta(z) f_\theta(z) \nabla_\theta \log p_\theta(z) \mathrm{d} z \\
&= \mathbb E_{p_\theta(z)}\left[f(z) \nabla_\theta \log p_\theta(z)\right]
\end{split}
$$

The familiar reader will recognize the last expression as the score-based estimator of the gradient, also known as REINFORCE in the RL community.
The advantage of calculating the gradient this way, is that we only require the density $p$ to be differentiable, but we have no constraints on $f$. However, this gradient does not scale well for high dimensionality of $\theta$, compared to the path-wise approach, since the variance scales with the dimension of $\theta$. See [this reference](https://arxiv.org/abs/1906.10652) for an in-depth review.

# Discrete randomness

The above discussion about the reparameterization trick makes the assumption that the probability density is differentiable. This is true for continuous distributions but it does not hold for discrete ones such as Bernoulli or Categorical distributions.

Multiple methods exist to deal with this issue, the most common ones being the Stright-Through estimator [(Bengio et al. 2013)](https://arxiv.org/abs/1308.3432) or the Gumbel-Softmax trick [(Jang et al. 2016)](https://arxiv.org/abs/1611.01144). Newer methods are continuously being developed such as the [StochasticAD.jl](https://arxiv.org/abs/2210.08572) package in the Julia language.

Here, we focus on the Gumbel-Softmax (GS) trick. The GS distribution is a continuous relaxation of the Categorical distribution. The GS has a temperature parameter $\tau$ which controls the smoothness of the approximation such that the exact Categorical sampling is recovered for $\tau\to 0$. However, the variance of the gradient will grow as $\tau\to 0$ so $\tau$ acts as a bias-variance trade-off.

PyTorch has a standard implementation ready to use. In particular, it implements a "hard" version of GS. The idea is that we can use the samples from the categorical distribution in the forward simulated pass, and use the continuous relaxation to approximate the gradients in the backward pass. This way we guarantee that the forward simulation is exactly equivalent to the non-relaxed version.

Consider the program with input parameter $\theta$:

$$
x = 2 \mathrm{Bernoulli}(3\theta) + \mathrm{Bernoulli}(\theta)
$$

We have

$$
\nabla_\theta \mathbb E [2 \mathrm{Bernoulli}(3\theta) + \mathrm{Bernoulli}(\theta)] = \nabla_\theta (6\theta + \theta) = 7
$$


```python
def sample_bernoulli(theta, gs_tau):
    logits = torch.cat([theta, 1 - theta]).log()
    return torch.nn.functional.gumbel_softmax(logits=logits, tau=gs_tau, hard=True)


def f(theta, gs_tau):
    return 2 * sample_bernoulli(3 * theta, gs_tau) + sample_bernoulli(theta, gs_tau)
```


```python
n_samples = 1_000
taus = [0.1, 0.5, 0.9]
gradients_per_tau = {}
for tau in taus:
    gradients = []
    for i in range(n_samples):
        theta = torch.tensor([0.1], requires_grad=True)
        f(theta, tau)[0].backward()
        gradients.append(theta.grad.item())
    gradients_per_tau[tau] = gradients
```


```python
fig, ax = plt.subplots()
for i, (tau, gradients) in enumerate(gradients_per_tau.items()):
    ax.boxplot(gradients, showmeans=True, positions=[i], labels=[f"$\\tau={tau}$"], showfliers=False)
ax.axhline(7, color="blue", linestyle="dashed", linewidth=1, label="Expected gradient")
ax.legend()
```




    <matplotlib.legend.Legend at 0x15c2efe20>




    
![png](02-differentiating-randomness_files/02-differentiating-randomness_20_1.png)
    


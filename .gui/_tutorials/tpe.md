# <span class="tutorial_icon invert_in_dark_mode">🌳</span> Tree-structured Parzen Estimator (TPE)

<div id="toc"></div>

<!-- What is TPE and how does it work? -->

<!-- Category: Models -->

## What is TPE?

The **Tree-structured Parzen Estimator (TPE)** is a sequential model-based optimization algorithm commonly used for hyperparameter tuning. Instead of modeling the objective function directly, it models the probability densities of good and bad hyperparameter configurations separately, allowing efficient sampling of promising configurations.

## How does it operate?

- **Density estimation**: TPE builds two probabilistic models — one for hyperparameters associated with good results and one for all others.
- **Sequential optimization**: It samples new hyperparameters by maximizing the ratio of these densities, focusing on regions with a higher chance of improvement.
- **Adaptive search**: The algorithm iteratively updates its models as new observations come in, improving the quality of sampled hyperparameters over time.

This approach enables efficient exploration of complex, high-dimensional, and conditional search spaces often encountered in machine learning hyperparameter tuning.

## How does TPE work in practice?

TPE works by modeling the likelihood of hyperparameters leading to good versus bad outcomes, rather than directly modeling the objective function. At each step, it selects new hyperparameter candidates by maximizing the expected improvement, guided by its density models. This results in a focused and efficient search process that adapts as more data becomes available.

## When to use TPE

- When you have a **complex or high-dimensional hyperparameter space** with continuous, discrete, and conditional parameters.
- When your **objective function evaluations are expensive**, and you want to minimize the number of evaluations needed for good performance.
- When you need an optimizer that **adapts well as it learns more** about the search space, improving efficiency over time.

## When not to use TPE

- When you have a **very small search space** where exhaustive or grid search is feasible and simpler.
- When your **objective function is noisy or highly stochastic**, which can confuse the density models TPE relies on.
- When you require **extremely fast single-step decisions** and cannot afford the overhead of maintaining probabilistic models.

# Surrogate Models

<!-- What are Surrogate Models and when to use which one? -->

<div id="toc"></div>

In the context of the [Ax](https://ax.dev/) (Adaptive Experimentation) platform, **surrogate models** are statistical models that are used to approximate expensive-to-evaluate functions. These models serve as a proxy for the actual function being optimized in Bayesian optimization.

Bayesian optimization is often used for optimizing functions that are computationally expensive or time-consuming to evaluate, such as in machine learning hyperparameter tuning, engineering design, or scientific simulations. Instead of evaluating the function at every iteration, surrogate models allow the algorithm to predict the function’s behavior based on past evaluations, and make informed decisions about where to evaluate next.

Surrogate models in Ax are typically probabilistic models, such as Gaussian Processes (GP), that provide not only a prediction of the function value but also an estimate of uncertainty. This helps guide the optimization process by balancing exploration (evaluating untested regions) and exploitation (focusing on regions where the model predicts optimal values).

## What Are They Good For?

Surrogate models are particularly useful in scenarios where the function being optimized is:

- **Expensive to evaluate**: Surrogate models reduce the number of actual function evaluations needed by predicting function values at untested points, thus saving computational resources.
- **Noisy or uncertain**: Surrogate models can quantify uncertainty in predictions, which helps in understanding the variability in the actual function and making more robust optimization decisions.
- **High-dimensional**: Surrogate models can help optimize high-dimensional functions efficiently, where traditional grid search or random search methods would be computationally prohibitive.

In summary, surrogate models in Ax help optimize complex, expensive, or uncertain functions by approximating the objective function and guiding the optimization process in a more resource-efficient manner.

## Default Generators

### BOTORCH_MODULAR
`Models.BOTORCH_MODULAR` is the default Gaussian Process model using BoTorch under the hood. It fits a GP to observed data and proposes new candidates via acquisition function optimization (typically qEI/qNEI or qEHVI for multi-objective).
**Best suited for:** *Continuous* or low-cardinality *categorical* search spaces with *low to medium* dimensionality (typically <20 after one-hot encoding). Ideal when function evaluations are *expensive*.
**Use case:** Hyperparameter tuning of ML models with few parameters; optimization of quantum circuits, materials, or simulation parameters.
- **Pros:** Strong uncertainty modeling via GP; acquisition-driven sampling; supports multi-objective; checkpointable/resumable.
- **Cons:** Struggles with high-dimensional or high-cardinality categorical variables (dimensionality explosion via OHE); doesn't scale well to thousands of evaluations (GP training cost).

Sources: [BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization](https://arxiv.org/abs/1910.06403)

### SOBOL
Sobol sampling generates *quasi-random low-discrepancy sequences* over the search space. Typically used for initialization.
**Best suited for:** Any search space where *initial uniform exploration* is needed.
**Use case:** Initial design in BO pipelines; sensitivity analysis; uniform space exploration.
- **Pros:** Deterministic, reproducible, better space coverage than random sampling; works with mixed or constrained spaces.
- **Cons:** No learning or adaptive behavior; does not propose candidates based on past observations.

Sources: [Sampling based on Sobol' sequences for Monte Carlo techniques applied to building simulations](https://publica.fraunhofer.de/handle/publica/375074)

### FACTORIAL
The `FACTORIAL`-model generates a full factorial design over categorical variables. It enumerates all possible discrete combinations.
**Best suited for:** Low-cardinality, fully discrete search spaces (categorical/integer).
**Use case:** Grid search; controlled A/B testing; combinatorial experiments with small parameter spaces.
- **Pros:** Exhaustive; interpretable; no randomness.
- **Cons:** Grows exponentially with variable count/cardinality; not suitable for continuous parameters.

### SAASBO
"**Sparse Axis-Aligned Subspace BO**" uses a sparse GP prior (hierarchical Half-Cauchy on lengthscales) and Hamiltonian Monte Carlo (NUTS) for inference.
**Best suited for:** *High-dimensional* problems (hundreds of parameters) with *very limited evaluation budgets* (e.g., ≤100 trials).
**Use case:** Deep learning HP tuning with large config spaces; material design; scientific problems with few effective parameters.
- **Pros:** Automatically identifies relevant subspaces; excellent performance in sparse high-D regimes.
- **Cons:** Very slow per iteration (HMC is expensive); doesn’t scale beyond ~100–200 observations; impractical with many categorical variables.

Sources: [SAASBO Paper (Eriksson et al., 2021)](https://arxiv.org/abs/2006.04492), [SAASBO in the BoTorch-Documentation](https://botorch.org/docs/tutorials/saasbo)

### UNIFORM
`UNIFORM` generates uniformly random samples. Equivalent to PseudoRandom, just explicitly uniform.
**Best suited for:** Very cheap-to-evaluate functions; embarrassingly parallel sampling.
**Use case:** Random stress testing; large-scale exploration; fallback search method.
- **Pros:** Simple; unbiased; robust; cheap.
- **Cons:** No adaptivity; inefficient in high-dimensional or expensive settings.

### BO_MIXED
`BO_MIXED` is a BoTorch-powered model designed for *mixed spaces* (categorical + continuous). It encodes categories numerically (ordinal/integer) and uses a combined kernel.
**Best suited for:** Optimization with both continuous and categorical variables.
**Use case:** Neural architecture + HP tuning (e.g., architecture type + learning rate); mixed material design problems.
- **Pros:** Explicit handling of categorical parameters; avoids crude one-hot scaling; uses acquisition functions for adaptive proposals.
- **Cons:** Categorical optimization is harder than continuous; combinatoric explosion if too many categories; slower candidate generation.

Sources: [Mixed-Variable Bayesian Optimization](https://arxiv.org/abs/1907.01329)

## Special Generators

These generators cannot be [continued](tutorials?tutorial=continue_job) or used in [Custom Generation Strategies](tutorials?tutorial=custom_generation_strategy).

### RANDOMFOREST
Uses a Random Forest surrogate model for prediction (no candidate generation). Suitable for modeling nonlinear effects in categorical/mixed spaces.
**Best suited for:** Regression-only tasks; modeling complex, discrete-heavy datasets.
**Use case:** Model diagnostics; cross-validation predictions; analysis of parameter importance.
- **Pros:** Handles categorical features natively; nonlinear modeling; scalable to many observations.
- **Cons:** Predictive uncertainty is rough and less calibrated than GP; poor extrapolation behavior; slower training/prediction on large search spaces.

Sources: [Hyperparameters and Tuning Strategies for Random Forest](https://arxiv.org/abs/1804.03515);
[Hyperparameters, Tuning, and Meta-Learning for Random Forest](https://edoc.ub.uni-muenchen.de/24557/1/Probst_Philipp.pdf);
[Better Trees: An Empirical Study on Hyperparameter Tuning of Classification Decision Tree Induction Algorithms](https://arxiv.org/abs/1812.02207);
[Generalising Random Forest Parameter Optimisation to Include Stability and Cost](https://arxiv.org/abs/1706.09865)

### PSEUDORANDOM
Pseudo-random uniform sampling over the search space (via RNG). Like Sobol, but purely random.
**Best suited for:** Baseline methods, quick-and-dirty optimization, very cheap evaluations.
**Use case:** Random search for ML hyperparameters; brute-force sampling; baseline comparisons.
- **Pros:** Simple; cheap; easily parallelizable; often competitive with grid search.
- **Cons:** No learning or adaptive feedback; inefficient in high-dimensional spaces.

Sources: [Bayesian Optimization using Pseudo-Points](https://arxiv.org/abs/1910.05484)

### EXTERNAL_GENERATOR

External generators allow you to use *any* external program to generate new points. See [External Generators](tutorials?tutorial=external_generator) for more details.

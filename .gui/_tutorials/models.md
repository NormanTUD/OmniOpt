# Surrogate Models

<!-- What are Surrogate Models and when to use which one? -->

In the context of the [Ax](https://ax.dev/) (Adaptive Experimentation) platform, **surrogate models** are statistical models that are used to approximate expensive-to-evaluate functions. These models serve as a proxy for the actual function being optimized in Bayesian optimization.

Bayesian optimization is often used for optimizing functions that are computationally expensive or time-consuming to evaluate, such as in machine learning hyperparameter tuning, engineering design, or scientific simulations. Instead of evaluating the function at every iteration, surrogate models allow the algorithm to predict the function’s behavior based on past evaluations, and make informed decisions about where to evaluate next.

Surrogate models in Ax are typically probabilistic models, such as Gaussian Processes (GP), that provide not only a prediction of the function value but also an estimate of uncertainty. This helps guide the optimization process by balancing exploration (evaluating untested regions) and exploitation (focusing on regions where the model predicts optimal values).

## What Are They Good For?

Surrogate models are particularly useful in scenarios where the function being optimized is:

- **Expensive to evaluate**: Surrogate models reduce the number of actual function evaluations needed by predicting function values at untested points, thus saving computational resources.
- **Noisy or uncertain**: Surrogate models can quantify uncertainty in predictions, which helps in understanding the variability in the actual function and making more robust optimization decisions.
- **High-dimensional**: Surrogate models can help optimize high-dimensional functions efficiently, where traditional grid search or random search methods would be computationally prohibitive.

In summary, surrogate models in Ax help optimize complex, expensive, or uncertain functions by approximating the objective function and guiding the optimization process in a more resource-efficient manner.


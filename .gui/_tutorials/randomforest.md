# Random Forest

<div id="toc"></div>

<!-- What are random forests and how do they work? -->

## What is a Random Forest?

A **Random Forest** is a meta-algorithm that builds a collective of decision structures (trees) and aggregates their outcomes to form a more stable, generalized response. It leverages randomness both in data selection and feature consideration, aiming to reduce bias and variance simultaneously.

## How does it operate?

- **Stochastic construction**: Multiple learners are instantiated using randomly sampled views of the original data.
- **Independent reasoning**: Each learner (tree) forms a distinct internal model based on local data and partial feature information.
- **Synthesis of judgment**: The ensemble's final output is derived by merging the diverse predictions—typically via averaging (regression) or voting (classification).

This creates a robust, noise-resistant estimator that thrives in high-dimensional, non-linear spaces.

## How does it guide parameter selection?

In optimization contexts:

- **Model as proxy**: The forest learns a representation of how parameter configurations map to performance metrics.
- **Hypothesis generation**: A space of candidate parameters is stochastically explored.
- **Informed filtering**: Candidates are scored by the model, ranking them by predicted utility.
- **Constraint-aware selection**: Top candidates are filtered through domain constraints before selection.

In essence, the Random Forest acts as a structured intuition engine—guiding the search for optimal configurations without direct evaluation of every possibility.

## When and why to use it?

Random Forests excel in scenarios where:

- **Data is noisy, sparse, or irregular**.
- **Black-box evaluation is expensive**, and a quick approximation is preferred.
- **Uncertainty estimates are not strictly required**, or interpretability is more important than precision.
- **Computation needs to remain light**, or infrastructure is limited.

Compared to modular Bayesian approaches like **BoTorch**, Random Forests:

- Are **faster to fit**, especially with many categorical or discrete inputs.
- Can be **more robust to small datasets** and **less sensitive to prior specification**.

They are less suited when:

- Probabilistic modeling of the objective is central.
- Exploration–exploitation trade-offs need explicit control.
- Gradients or uncertainty quantification are essential to the problem structure.

In such cases, modular GP-based methods may be more appropriate—but at higher complexity and cost.

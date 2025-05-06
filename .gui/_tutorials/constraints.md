# Constraints
<!-- What are Constraints and how to use them? -->
<div id="toc"></div>
## Why Constraints Matter in Hyperparameter Optimization and Simulations

When performing hyperparameter optimization or running large-scale simulations, constraints allow you to embed domain knowledge directly into the search space. Instead of blindly exploring all possible combinations of parameters, constraints restrict the optimization process to *feasible* or *meaningful* regions — which improves efficiency, avoids invalid configurations, and reflects real-world limitations.

### What Are Constraints?

In the context of hyperparameter optimization, constraints are mathematical conditions that must be satisfied by the parameter values during the search. A common form of constraint is a linear inequality such as \( a \cdot x + b \cdot y \leq c \), where \(x\) and \(y\) are tunable parameters (e.g., learning rate, number of layers), and \(a\), \(b\), and \(c\) are fixed constants. These expressions define a subspace in which the optimizer is allowed to operate.

### Why Use Constraints?

Constraints are useful for several reasons:

- **Feasibility**: Some combinations of parameters are invalid due to physical, computational, or algorithmic limits. For example, in a simulation model, the sum of resource allocations may not exceed a total budget.
  
- **Domain knowledge**: You might know from experience or theory that certain parameter combinations don’t make sense or always lead to failure. Instead of learning this through wasted trials, you can encode this knowledge directly.

- **Safety**: Especially in simulations of real-world systems (like robotics, medicine, or finance), running configurations outside known-safe bounds can be dangerous or misleading.

- **Efficiency**: By reducing the size of the search space, constraints help optimizers converge faster and require fewer evaluations - which is especially important when each run is expensive or slow.

- **Logical conditions**: Sometimes constraints reflect logical rules, such as "if parameter A is high, parameter B must be low", which can be enforced with inequalities.

### Machine Learning Examples

- Limiting the total number of model parameters due to memory constraints: \(\text{hidden_units_layer}_1 + \text{hidden_units_layer}_2 \leq 1024 \)
- Ensuring a balance between training epochs and learning rate to prevent overfitting: \(0.1 \cdot \text{epochs} + 50 \cdot \text{learning_rate} \leq 10\)

### Simulation Examples

- Respecting physical conservation laws or budget limits: \(\text{power_alloc}_A + \text{power_alloc}_B <= \text{max_total_power}\)
- Enforcing time-step stability conditions in numerical models.

### Mathematical Form

In general, constraints can be written as:

$$
a_1 \cdot x_1 + a_2 \cdot x_2 + \dots + a_n \cdot x_n \leq c,
$$

where \(x_1, x_2, \dots, x_n\) are parameters and \(a_i, c\) are constants.

## Using Constraints in OmniOpt2

OmniOpt2 allows you to specify constraints in two ways: through the graphical user interface (GUI) or via the command-line interface (CLI).

### 1. Using the GUI

In the GUI, you can enter a list of constraints directly. This is done through an input field where you can specify each constraint in the following form:

$$
a \cdot x + b \cdot y + \dots \leq c
$$

For example, you might specify a constraint like:

$$
3 \cdot \text{learning_rate} + 2 \cdot \text{batch_size} \leq 100
$$

This constraint would limit the combination of the learning rate and batch size to ensure that they don't exceed a total of 100. To enter constraints in the GUI, see the screenshot below for guidance:

<img alt="Constraints GUI" data-lightsrc="imgs/constraints_light.png" data-darksrc="imgs/constraints_dark.png" /><br>

### 2. Using the CLI

In the CLI, constraints can be added using the `--experiment_constraints` argument. You need to encode each constraint in base64 format. Here’s an example:

```bash
--experiment_constraints $(echo "50 * learning_rate + 20 * batch_size >= 1000" | base64 -w0) $(echo "100 * learning_rate + 200 * num_layers >= 500" | base64 -w0)
```

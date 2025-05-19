# <span class="tutorial_icon invert_in_dark_mode">⚖️</span> Constraints

<!-- What are Constraints and how to use them? -->

<!-- Category: Advanced Usage -->

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

- Respecting physical conservation laws or budget limits: \(\text{power_alloc}_A + \text{power_alloc}_B \leq \text{max_total_power}\)
- Enforcing time-step stability conditions in numerical models.

## Ax-Constraints

Ax-constraints are taken into account already at the creation of new points, meaning parameter-arms that don't suffice the conditions will not be created in the first place.

### Mathematical Form

In general, ax-constraints can be written as:

- \(a_1 \cdot x_1 + a_2 \cdot x_2 + \dots + a_n \cdot x_n \leq c\)
- or \(x_1 \leq x_2\)

where \(x_1, x_2, \dots, x_n\) are parameters and \(a_i, c\) are constants (i.e., `int` or `float`). This is because the creation of new points with constraints is based on calculating the [linear span](https://en.wikipedia.org/wiki/Linear_span).

For a constraint to be treated as an ax-constraint:

- It must be mathematically valid (parsable and type-correct)
- It must follow the linear form rules described above

#### Examples of valid ax-constraints:

- \(3 \cdot \text{learning_rate} + 2 \cdot \text{dropout} <= 1.0 \)
- \( \text{batch_size} <= 512 \)
- \( \text{layers} <= \text{depth} \)

#### Examples **not** valid as ax-constraints:

- \( \text{sample_period} \cdot \text{window_size} >= 1 \)
- \( 1 \cdot \text{sample_period} >= 1 / \text{window_size} \)

These are valid constraints but will be evaluated *after* point creation, as Post-Generation-Constraints.

## Post-Generation-Constraints

For more complex constraints, Post-Generation-Constraints will be used. These are evaluated *after* point creation. If a point does not satisfy a Post-Generation-Constraint, the job will be immediately marked as *abandoned* and will not be executed.

### Valid operators for Post-Generation-Constraints include:

- `==`, `!=`, `<=`, `>=`

### Valid calculation types:

- Arithmetic sums and differences: `+`, `-`
- Scalar multiplications: `*`
- Divisions: `/`
- Unary operations: `+x`, `-x`
- Constants (e.g., `5`, `1.2`, etc.)

### Post-Generation-Constraints must:

- Be valid equations (syntax-checked and structurally correct)
- Include at least one comparison operator
- Reference only allowed parameter names and numeric constants
- Not be ax-compatible — otherwise, they will be treated as ax-constraints instead

#### Examples of valid Post-Generation-Constraints:

- \( \text{sample_period} \cdot \text{window_size} >= 1 \)
- \( 1 * \text{sample_period} >= 1 / \text{window_size} \)
- \( \text{training_steps} + \text{warmup_steps} <= 10000 \)
- \( \text{dropout_rate } != 0.5 \)

Multiple constraints can be defined. Each will automatically be interpreted as ax or non-ax depending on its structure.

## Using Constraints in OmniOpt2

OmniOpt2 allows you to specify constraints in two ways: through the graphical user interface (GUI) or via the command-line interface (CLI).

### 1. Using the GUI

In the GUI, you can enter a list of constraints directly. This is done through an input field where you can specify each constraint in the following forms (given, again that, \( x \) and \( y \) are parameters, and \( a, b, c \) are `int` or `float`):

$$
a \cdot x + b \cdot y + \dots \leq c,
$$

or:

$$
x \leq y.
$$

For example:

$$
3 \cdot \text{learning_rate} + 2 \cdot \text{batch_size} \leq 100
$$

This constraint limits the combination of learning rate and batch size to ensure they don't exceed a total of 100. See the screenshot below for input guidance:

<img alt="Constraints GUI" data-lightsrc="imgs/constraints_light.png" data-darksrc="imgs/constraints_dark.png" /><br>

### 2. Using the CLI

In the CLI, constraints can be added using the `--experiment_constraints` argument. Each constraint must be encoded in base64 format.

#### Example:

```bash
--experiment_constraints $(echo "50 * learning_rate + 20 * batch_size >= 1000" | base64 -w0) $(echo "100 * learning_rate + 200 * num_layers >= 500" | base64 -w0)
```

### Constraints in Continued Jobs

Given the option `--disable_previous_job_constraint` is not set, the constraints specified will be taken over to continued jobs as well. They can be overridden, though, by adding another `--experiment_constraints`. This will delete all old constraints and only work on the new ones.

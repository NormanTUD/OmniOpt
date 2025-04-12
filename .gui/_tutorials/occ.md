# What is optimization with combined criteria?

<!-- How to use OmniOpt2 with multiple results (OCC) -->

<div id="toc"></div>

## What is OCC?

Optimization with combined criteria, or OCC, is a way of optimizing neural networks and simulations for multiple parameters instead of only one parameter.

Usually, you optimize only one parameter. With OCC, multiple parameters can be optimized.

Your program needs to have one or multiple outputs, like this:

```
RESULT1: 123
RESULT2: 321
RESULT3: 1234
RESULT4: 4321
```

OmniOpt2 will automatically parse all RESULTs from your output string and will try to merge them together, by default, with Euclidean distance.

## What is OCC good for?

Sometimes you have conflicting goals, for example, a neural network's accuracy is much better when it has larger neurons, but it also takes potentially exponentially longer to train. To find a sweet-spot between, for example, learning time and accuracy, OCC may help. It allows you to find a good spot, where both options are as best (i.e. lowest) as possible. Another possible measure would be time and accuracy or loss and validation loss.

You have to specify the way your different results are outputted yourself in your program. It is recommended you normalize large numbers to a certain scale, so that for example all the result values are between 0 and 1. This is technically not needed, but large values may skew the results.

## Different types of OCC

The \( \text{sign} \)-variable-detection method is the same for all signed functions, while \( \_\text{args} \) being the set of all given parameters:

\[
\text{sign} =
\begin{cases}
        -1 & \text{if } \exists x \in \_\text{args} \text{ such that } x < 0, \\
        1 & \text{otherwise}
\end{cases}
\]

### Signed Euclidean Distance

\[
\text{distance} = \text{sign} \cdot \sqrt{\sum_{i=1}^{n} a_i^2}
\]

Computes the Euclidean distance, which is the square root of the sum of squared values.

### Signed Geometric Distance

\[
\text{distance} = \text{sign} \cdot \left( \prod_{i=1}^{n} |a_i| \right)^{\frac{1}{n}}
\]

Explanation:

- Computes the geometric mean instead of a sum-based distance.
- The geometric mean is the n-th root of the product of the absolute values.

### Signed Harmonic Distance

\[
\text{distance} = \text{sign} \cdot \frac{n}{\sum_{i=1}^{n} \frac{1}{|a_i|}}
\]

Explanation:

- Computes the harmonic mean instead of an arithmetic or geometric mean.
- The harmonic mean is the inverse of the average of reciprocals.

### Signed Minkowski Distance

\[
\text{distance} = \text{sign} \cdot \left( \sum_{i=1}^{n} |a_i|^p \right)^{\frac{1}{p}}
\]

Explanation:

- Generalization of Euclidean and Manhattan distances:
- When \( p = 1 \), it’s equivalent to Manhattan distance.
- When \( p = 2 \), it’s equivalent to Euclidean distance.
- When \( p > 2 \), it gives more weight to larger differences.

### Signed Weighted Euclidean Distance

\[
\text{distance} = \text{sign} \cdot \sqrt{\sum_{i=1}^{n} w_i \cdot a_i^2}
\]

where \( w_i \) is the weight assigned to each value, which can be specified by using `--signed_weighted_euclidean_weights`.

Explanation:

- Similar to Euclidean distance but weights each dimension differently.
- Gives more importance to certain hyperparameters.

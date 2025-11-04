# <img class='emoji_nav' src='emojis/crystal_ball.svg' /> What is Multi-Objective-Optimization?

<!-- How to use OmniOpt2 with Multi-Objective-Optimization (MOO)? -->

<!-- Category: Multiple Objectives -->

<div id="toc"></div>

## What is MOO?

Sometimes, you have several goals in mind when optimizing your neural network. For example, you may want to minimize two goals that conflict with each other. For example, you want to minimize your loss, but also minimize the time a prediction takes. Usually, better results mean that your network needs more time. With OmniOpt2, you can optimize for both. This will not give you a single result, but rather a so-called [Pareto-Front](https://en.wikipedia.org/wiki/Pareto_front) of results, of which you can then choose one that best fits your needs. OmniOpt2 allows you to use as as many RESULT-values as you wish.

## How to use Multi-Objective-Optimization with OmniOpt2?

It's very similar to using [single-optimization](tutorials?tutorial=run_sh), the only differences being that, instead of using

```python
print(f"RESULT: {loss}")
```

you now need two (or as many as you have Objectives) lines:

```python
print(f"LOSS: {loss}")
```

and

```python
print(f"PREDICTION_TIME: {prediction_time}")
```

and you need the option

```bash
--result_names LOSS PREDICTION_TIME
```

The Extra-option can be set in the GUI in the *Show additional parameters*-table at the bottom at the option called *Result-Names*. It accepts a space-separated list of result-names that are then used internally to search through the stdout of your program-to-be-optimized. You can use up to roughly 20 RESULT-names.

## How to minimize one and maximize the other parameter?

By default, OmniOpt2 minimizes all parameters. Minimizing one and maximizing another parameter can easily be done though, by specifying it in the RESULT-Names-Parameter:

```bash
--result_names LOSS=min PREDICTION_TIME=max
```

This way, LOSS is minimized, while PREDICTION_TIME is maximized.

## How to calculate Pareto-Fronts in cancelled jobs

When you have a job that was cancelled for whatever reason (i.e. time limit, memory limit, ...), but you still had results and want to calculate the Pareto-Front afterward, you can do that by simply calling:

```bash
omniopt --calculate_pareto_front_of_job runs/yourproject/0
```

This will calculate the pareto fronts, and, when you had `--live_share` enabled, will automatically push these results to the OmniOpt2-[Share](tutorials?tutorial=oo_share) server. This can also be done manually, by calling `omniopt_share` on the run folder.

## Normalization

In multi-objective optimization with Ax/BoTorch, it is highly recommended to normalize all objectives to a similar scale. Without normalization, the optimizer can be biased toward objectives with larger numeric ranges or steeper gradients, effectively neglecting others. Normalization ensures that the acquisition functions properly balance trade-offs between all objectives, allowing the search to explore the Pareto front more evenly and avoid being dominated by a single objective.

$$
\text{Normalized_value} = \frac{\text{Observed_value} - \text{Min_expected}}{\text{Max_expected} - \text{Min_expected}} \times 100
$$


It is highly recommended to normalize all objectives in multi-objective optimization. Without normalization, the optimizer may favor objectives with naturally larger scales, neglecting others. Normalization ensures that all objectives are treated fairly, allowing Ax/BoTorch to properly explore the Pareto front.

This can be done, for example, by mapping the expected minimum and maximum values of an objective to a normalized range (e.g., 0–100):

```python
# Example: normalize runtime
import time
start_time = time.time() # At the start of your program

import torch

# ... your code ...

end_time = time.time()

runtime_seconds = end_time - start_time                                                                                                                                                                                         

max_runtime = 3600  # expected maximum runtime in seconds
normalized_runtime = (runtime_seconds / max_runtime) * 100
normalized_runtime = round(normalized_runtime, 3)
print(f"NORMALIZED_RUNTIME: {normalized_runtime}")
```

or in bash with the `SECONDS` variable, which returns the number of second the bash process already ran:

```bash
#!/bin/bash

# ... your program...

max_runtime=3
normalized_runtime=$(echo "scale=3; ($SECONDS / $max_runtime) * 100" | bc)
echo "NORMALIZED_RUNTIME: $normalized_runtime"
```

## Caveats

<div class="caveat warning">
    <li> Using MOO prohibits most of the graphs you can usually plot with OmniOpt2, since the result-value is not unambiguous anymore and cannot be used for plotting easily. We'd recommend using [OmniOpt2-Share](tutorials?tutorial=oo_share) to plot Parallel plots of your data in the browser.</li>
    <li> It is very much recommended to normalize all results into the same range, as this makes it easier for the optimizer to find points that are along the pareto-minimum. It may happen that, while exploring the search space, the optimizer may lose focus on one or multiple of the goals and only optimize for the one which is easiest to maximize for.</li>
</div>

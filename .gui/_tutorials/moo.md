# What is Multi-Objective-Optimization?

## What is MOO?

Sometimes, you have several goals in mind when optimizing your neural network. For example, you may want to minimize two goals that conflict with each other. For example, you want to minimize your loss, but also minimize the time a prediction takes. Usually, better results mean that your network needs more time. With OmniOpt2, you can optimize for both. This will not give you a single result, but rather a so-called [Pareto-Front](https://en.wikipedia.org/wiki/Pareto_front) of results, of which you can then choose one that best fits your needs.

## How to use Multi-Objective-Optimization with OmniOpt2?

It's very similar to using [single-optimization](tutorials.php?tutorial=run_sh), the only differences being that, instead of using
```
print("RESULT: {loss}")
```

you now need two lines: 

```
print("LOSS: {loss}")
```

and

```
print("PREDICTION_TIME: {prediction_time}")
```

and you need the option

```
--result_names LOSS PREDICTION_TIME
```

The Extra-option can be set in the GUI in the *Show additional parameters*-table at the bottom at the option called *Result-Names*. It accepts a space-separated list of result-names that are then used internally to search through the stdout of your program-to-be-optimized. You can use up to roughly 20 RESULT-names.

## How to minimize one and maximize the other parameter?

By default, OmniOpt2 minimizes all parameters. Minimizing one and maximizing another parameter can easily be done though, by specifying it in the RESULT-Names-Parameter:

```
--result_names LOSS=min PREDICTION_TIME=max
```

This way, LOSS is minimized, while PREDICTION_TIME is maximized.

## Caveats

Using MOO prohibits most of the graphs you can usually plot with OmniOpt2, since the result-value is not unambiguous anymore and cannot be used for plotting easily. We'd recommend using OmniOpt2-Share to plot Parallel plots of your data in the browser.

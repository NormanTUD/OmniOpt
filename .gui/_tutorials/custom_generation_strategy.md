# <span class="tutorial_icon invert_in_dark_mode">🧩</span> Custom Generation Strategy

<!-- How to run custom generation strategies -->

<!-- Category: Advanced Usage -->

<div id="toc"></div>

## What are Custom Generation Strategies?

The default generation strategy is *generate n points randomly, then, m points with a systematic model*.

But under certain circumstances, you may need a more complex strategy, for example, first generating 10 pseudorandom points,
then searching them with BOTORCH_MODULAR, then searching 20 pseudorandom again, and then 10 SAASBO. This is possible with
OmniOpt2, and the way to do it is to use a *custom generation strategy*.

## How to enter them in the GUI

Click the *Show additional parameters* button, then scroll down to *Generation Strategy*. Enter the list of models with an equal-sign
and the number of jobs that should be executed with them, like given in the example.

<img alt="Custom Generation Strategy in GUI" data-lightsrc="imgs/custom_generation_strategy_light.png" data-darksrc="imgs/custom_generation_strategy_dark.png" /><br>

## How to add it manually to an OmniOpt2-job

It's simple: just add this command line argument to your OmniOpt2-call:

```bash
--generation_strategy SOBOL=10,BOTORCH_MODULAR=20,SOBOL=10
```

Of cource, change it according to your needs.

## Caveats
<div class="caveat warning">
- Currently, the custom Generation Strategy does not work with `PSEUDORANDOM`, `RANDOMFOREST` and `EXTERNAL_GENERATOR`
- Jobs with custom generation strategies cannot be continued
</div>

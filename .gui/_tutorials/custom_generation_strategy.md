# Custom Generation Strategy

<!-- How to run custom generation strategies -->

## What are Custom Generation Strategies?

The default generation strategy is *generate n points randomly, then, m points with a systematic model*.

But under certain circumstances, you may need a more complex strategy, for example, first generating 10 pseudorandom points,
then searching them with BOTORCH_MODULAR, then searching 20 pseudorandom again, and then 10 SAASBO. This is possible with
OmniOpt2, and the way to do it is to use a *custom generation strategy*.

# How to enter them in the GUI



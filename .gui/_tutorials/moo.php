<h1>What is Multi-Objective-Optimization?</h1>

<!-- How to use OmniOpt2 with Multi-Objective-Optimization (MOO)? -->

<div id="toc"></div>

<h2 id="what_is_occ">What is MOO?</h2>

<p>Sometimes, you have several goals in mind when optimizing your neural network. For example,
you may want to minimize two goals that conflict with each other. For example, you want to
minimize your loss, but also minimize the time a prediction takes. Usually, better results
mean that your network needs more time. With OmniOpt2, you can optimize for both. This will
not give you a single result, but rather a so-called
<a href="https://en.wikipedia.org/wiki/Pareto_front" target="_blank">Pareto-Front</a> of results,
of which you can then chose one that best fits your needs.</p>

<h2 id="how_to_use_moo">How to use Multi-Objective-Optimization with OmniOpt2?</h2>

<p>It's very similar to using <a href="tutorials.php?tutorial=run_sh" target="_blank">single-optimization</a>,
the only differences being that, instead of using
<span class="invert_in_dark_mode"><code class='language-python'>print("RESULT: {loss}")</code></span>,
you now need two lines: 
<span class="invert_in_dark_mode"><code class='language-python'>print("LOSS: {loss}")</code></span> and
<span class="invert_in_dark_mode"><code class='language-python'>print("PREDICTION_TIME: {prediction_time}")</code></span>,
and you need the option
<span class="invert_in_dark_mode"><code class='language-bash'>--result_names LOSS PREDICTION_TIME</code></span>.</p>

<p>The Extra-option can be set in the GUI in the <i>Show additional parameters</i>-table at the bottom at the option
called <i>Result-Names</i>. It accepts a space-seperated list of result-names that are then used internally to search
through the stdout of your program-to-be-optimized. You can use up to roughly 20 RESULT-names.</p>

<h2 id="min_max">How to minimize one and maximize the other parameter?</h2>

<p>By default, OmniOpt2 minimizes all parameters. Minimizing one and maximizing another
parameter can easily be done though, by specifying it in the RESULT-Names-Parameter:
<span class="invert_in_dark_mode"><code class='language-bash'>--result_names LOSS=min PREDICTION_TIME=max</code></span>.
This way, LOSS is minimized, while PREDICTION_TIME is maximized.</p>

<h2 id="caveats">Caveats</h2>

<p>Using MOO prohibits most of the graphs you can usually plot with OmniOpt2, since the result-value is not
unambiguous anymore and cannot be used for plotting easily. We'd recommend using OmniOpt2-Share to plot
Parallel plots of your data in the browser.</p>

<h1>What is multi-objective-optimization and how to use it?</h1>
    
<div id="toc"></div>

<h2 id="what_is_moo">What is MOO?</h2>

<p>Multi-objective-optimization, or MOO, is a way of optimizing neural networks and simulations for
multiple parameters instead of only one parameter.</p>

<p>Usually, you optimize only one parameter. With MOO, multiple parameters can be optimized.</p>

<p>Your program needs to have one or multiple outputs, like this:</p>

<pre>
RESULT1: 123
RESULT2: 321
RESULT3: 1234
RESULT4: 4321
</pre>

<p>OmniOpt2 will automatically parse all RESULTs from your output string and will try to merge them together,
by default, with euclidean distance.</p>

<h2 id='what_is_it_good_for'>What is MOO good for?</h2>

<p>Sometimes you have conflicting goals, for example, a neural network's accuracy is much better when it
has larger neurons, but it also takes potentially exponentially longer to train. To find a sweet-spot between, 
for example, learning time and accuracy, MOO may help. It allows you to find a good spot, where both options are
as best (i.e. lowest) as possible. Another possible measure would be time and accuracy or loss and validation loss.</p>

<p>You have to specify the way your different results are outputted yourself in your program. It is recommended you
normalize large numbers to a certain scale, so that for example all the result values are between 0 and 1. This is
technically not needed, but large values may skew the results.</p>

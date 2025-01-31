<h1>What is optimization with combined criteria?</h1>

<!-- How to use OmniOpt2 with multiple results (OCC) -->

<div id="toc"></div>

<h2 id="what_is_occ">What is OCC?</h2>

<p>Optimization with combined criteria, or OCC, is a way of optimizing neural networks and simulations for
multiple parameters instead of only one parameter.</p>

<p>Usually, you optimize only one parameter. With OCC, multiple parameters can be optimized.</p>

<p>Your program needs to have one or multiple outputs, like this:</p>

<pre>
RESULT1: 123
RESULT2: 321
RESULT3: 1234
RESULT4: 4321
</pre>

<p>OmniOpt2 will automatically parse all RESULTs from your output string and will try to merge them together,
by default, with euclidean distance.</p>

<h2 id='what_is_it_good_for'>What is OCC good for?</h2>

<p>Sometimes you have conflicting goals, for example, a neural network's accuracy is much better when it
has larger neurons, but it also takes potentially exponentially longer to train. To find a sweet-spot between,
for example, learning time and accuracy, OCC may help. It allows you to find a good spot, where both options are
as best (i.e. lowest) as possible. Another possible measure would be time and accuracy or loss and validation loss.</p>

<p>You have to specify the way your different results are outputted yourself in your program. It is recommended you
normalize large numbers to a certain scale, so that for example all the result values are between 0 and 1. This is
technically not needed, but large values may skew the results.</p>

<h2 id="occ_types">Different types of OCC</h2>
The \( \text{sign} \)-variable-detection method is the same for all signed functions:

$$
\text{sign} = 
\begin{cases} 
	-1 & \text{if } \exists x \in \_\text{args} \text{ such that } x < 0, \\
	1 & \text{otherwise}.
\end{cases}
$$.

<h3 id="signed_euclidean_distance">Signed Euclidean Distance</h3>

$$ \text{distance} = \text{sign} \cdot \sqrt{\sum_{i=1}^{n} a_i^2} $$

Explanation:

<ul>
	<li>Computes the Euclidean distance, which is the square root of the sum of squared values.</li>
	<li>Maintains the sign:</li>
	<li>Otherwise, it remains positive.</li>
</ul>

<h3 id="signed_geometric_distance">Signed Geometric Distance</h3>

$$ \text{distance} = \text{sign} \cdot \left( \prod_{i=1}^{n} |a_i| \right)^{\frac{1}{n}} $$

Explanation:

<ul>
	<li>Computes the geometric mean instead of a sum-based distance.</li>
	<li>The geometric mean is the th root of the product of the absolute values.</li>
	<li>Sign Handling:</li>
	<ul>
		<li>If the number of negative values is odd, the result is negative.</li>
		<li>Otherwise, it’s positive.</li>
	</ul>
</ul>


<h3 id="signed_harmonic">Signed Harmonic Distance</h3>

$$ \text{distance} = \text{sign} \cdot \frac{n}{\sum_{i=1}^{n} \frac{1}{|a_i|}} $$

Explanation:

<ul>
	<li>Computes the harmonic mean instead of an arithmetic or geometric mean.</li>
	<li>The harmonic mean is the inverse of the average of reciprocals.</li>
	<li>Sign Handling:</li>
	<li>If the number of negative values is odd, the result is negative.</li>
	<li>Otherwise, it’s positive.</li>
</ul>

<h3 id="signed_minkowski_distance">Signed Minkowski Distance</h3>

$$ \text{distance} = \text{sign} \cdot \left( \sum_{i=1}^{n} |a_i|^p \right)^{\frac{1}{p}} $$

Explanation:

<ul>
	<li>Generalization of Euclidean and Manhattan distances:</li>
	<li>When p = 1, it’s equivalent to Manhattan distance.</li>
	<li>When p = 2, it’s equivalent to Euclidean distance.</li>
	<li>When p > 2, it gives more weight to larger differences.</li>
</ul>

<h3 id="signed_weighted_euclidean_distance">Signed Weighted Euclidean Distance</h3>

$$ \text{distance} = \text{sign} \cdot \sqrt{\sum_{i=1}^{n} w_i \cdot a_i^2} $$ 

where \( w_i \) is the weight assigned to each value, which can be specified by using <samp>--signed_weighted_euclidean_weights</samp>.

Explanation:

<ul>
	<li>Similar to Euclidean distance but weights each dimension differently.</li>
	<li>Gives more importance to certain hyperparameters.</li>
</ul>

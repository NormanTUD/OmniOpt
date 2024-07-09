<?php
	include("../_header_base.php");
?>
<h1>Continue jobs</h1>

<div id="toc"></div>

<h2 id="continue_with_same_options">Continue an old job with the same options</h2>
<p>Continuing an old job with the same options as previously, but with awareness of the hyperparameter-constellations
that have already been tested, is as simple as this, assuming your job is in <tt>runs/my_experiment/0</tt>:

<pre><code class="language-bash">./omniopt --continue runs/my_experiment/0</code></pre>

<p>This will start a new run with the same settings as the old one, but load all already tried out data points, and 
continue the search from there.</p>

<h2 id="continue_with_changed_options">Continue an old job with the changed options</h2>

<p>In continued runs, some options can be changed. For example,</p>

<pre><code class="language-bash">./omniopt --continue runs/my_experiment/0 --time=360</code></pre>

<p>Will run the continuation for 6 hours ( = 360 minutes), independent of how long the old job ran.</p>

<p>It is also possible to change parameter borders, though narrowing is not currently supported, i.e.
the parameters need to be equal or wider than in the previous run. You cannot remove parameters here or add
names of parameters that have previously not existed, though.</p>

<p>Also, parameter names and types must stay the same. This code, for example, runs a continued run but changes
the <tt>epochs</tt> parameter that was previously defined and could have been, for example, between 0 and 10. In
this new run, all old values will be considered, but new values of epochs can be between 0 and 1000.</p>

<pre><code class="language-bash">./omniopt --continue runs/my_experiment/0 --parameter epochs range 0 1000 int</code></pre>

<p>All parameters that are not specified here are taken out of the old run, and thus stay in the same borders.</p>

<h2 id="Folder">In which folder will a run continue in?</h2>
<p>It will create a new folder. Imagine there are already the subfolders <tt>0</tt>, <tt>1</tt> and <tt>2</tt> for your
experiment. If you continue the job <tt>0</tt>, it's job data will be in the subfolder <tt>3</tt> then, since it is the first
non-existing folder for that project..</p>

<h2 id="Caveat">Caveat</h2>
<p>It is currently not possible to decrease the search space on a continued run. Attempts to do that will be ignored and the
original limits will automatically be restored.</p>

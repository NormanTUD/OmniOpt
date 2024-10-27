<h1>Orchestrator</h1>

<div id="toc"></div>

<h2 id="what_is_the_orchestrator">What is the Orchestrator?</h2>

<p>Sometimes, partitions contain instable nodes that have, for example, hardware issues. The Orchestrator allows you to react to those circumstances by detecting certain configurable strings in the stdout and stderr of your programs. Given the strings are contained, certain actions are possible. For example,
restarting on a different node, restarting in general, and just excluding the node. This allows to detect defective nodes and skip them in production after only a single test job that failed on them automatically.</p>

<h2 id="example_file">Example <samp>orchestrator.yaml</samp>-file</h2>

<pre>
errors:
  - name: GPUDisconnected
    match_strings:
      - "AssertionError: ``AmpOptimizerWrapper`` is only available"
    behavior: ExcludeNode

  - name: Timeout
    match_strings:
      - "Timeout"
    behavior: RestartOnDifferentNode

  - name: ExampleRestart
    match_strings:
      - "StartAgain"
    behavior: Restart

  - name: StorageError
    match_strings:
      - "Read/Write failure"
    behavior: ExcludeNodeAndRestartAll
</pre>

<p>This configuration file does the following. When a job ends, and in the output the string...</p>

<ul>
	<li>... <samp>AssertionError: ``AmpOptimizerWrapper`` is only available</samp> appears, it will exclude that node from all future executions inside the current and all continued jobs from it</li>
	<li>... <samp>Timeout</samp> appears, it will exclude that node from all future executions inside the current and all continued jobs from it, and restart the job on the list of nodes on that partition excluding the one that ran this timeout job</li>
	<li>... <samp>StartAgain</samp> appears, it will restart that job (may end up on the same node)</li>
	<li>... <samp>Read/Write failure</samp> appears, it will exclude the node the job started on and restart it on a different node</li>
</ul>

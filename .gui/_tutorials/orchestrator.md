# Orchestrator

<!-- How to orchestrate how failing jobs should restart or be treated in general -->

<div id="toc"></div>

## What is the Orchestrator?

Sometimes, partitions contain instable nodes that have, for example, hardware issues. The Orchestrator allows you to react to those circumstances by detecting certain configurable strings in the stdout and stderr of your programs. Given the strings are contained, certain actions are possible. For example,
restarting on a different node, restarting in general, and just excluding the node. This allows to detect defective nodes and skip them in production after only a single test job that failed on them automatically.

## Example <samp>orchestrator.yaml</samp>-file

```
errors:
  - name: GPUDisconnected
    match_strings:
      - "AssertionError: AmpOptimizerWrapper is only available"
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
```

This configuration file does the following. When a job ends, and in the output the string...

- ... `AssertionError: AmpOptimizerWrapper is only available` appears, it will exclude that node from all future executions inside the current and all continued jobs from it
- ... `Timeout` appears, it will exclude that node from all future executions inside the current and all continued jobs from it, and restart the job on the list of nodes on that partition excluding the one that ran this timeout job
- ... `StartAgain` appears, it will restart that job (may end up on the same node)
- ... `Read/Write failure` appears, it will exclude the node the job started on and restart it on a different node

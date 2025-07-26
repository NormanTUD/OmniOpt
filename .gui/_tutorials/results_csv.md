# 📊 <tt>results.csv</tt>

<!-- What is the results.csv and what do its columns mean? -->

<!-- Category: Preparations, Basics and Setup -->

<div id="toc"></div>

# What is a CSV?

A **CSV** (Comma-Separated Values) file is a simple, text-based file format used to store tabular data. Each line represents a row, and each value is separated by a comma. CSVs are widely used because they are human-readable and compatible with many tools.

## What is `results.csv`?

In **OmniOpt**, the `results.csv` file is used to store the outcomes of all executed trials. It contains:

- The **hyperparameter configurations** tested during the optimization process.
- The **results** (objective values) produced by these configurations.
- Metadata about the optimization process such as the trial index, arm name, trial status, and generation method used.

### Example content:

```csv
trial_index,arm_name,trial_status,generation_node,RESULT,int_param
0,0_0,COMPLETED,SOBOL,-41764.12,-59
1,1_0,COMPLETED,BOTORCH_MODULAR,-916.12,10
```

### Column Explanations

#### `trial_index`
A unique, sequential number identifying the trial.

#### `arm_name`
A name identifying the specific configuration (or **arm**) tested during the trial.
In Ax (the optimization framework used by OmniOpt), an **arm** represents a single set of parameter values to be evaluated.

#### `trial_status`
Indicates the current or final state of a trial. Possible values are:

- **COMPLETED** – The trial ran successfully and returned a result.
- **FAILED** – The evaluation failed due to an error (e.g., crash, invalid config).
- **ABANDONED** – The trial was skipped or terminated early, often due to constraints ([post-generation-constraints](tutorials?tutorial=constraints#post-generation-constraints)) or poor performance.
- **RUNNING** – The trial is still in progress and has not yet returned a result.

#### `generation_node`
This shows which model or strategy was responsible for generating the configuration. It reflects the generation logic used for that trial.

### Generation Models (`generation_node`)

Below is a list of possible values in the `generation_node` column and a brief description of each; they correspond to the [models](tutorials?tutorial=models) available:

<table>
  <thead>
    <tr>
      <th>Generation Node</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MANUAL</strong></td>
      <td>A configuration that was inserted by previous jobs.</td>
    </tr>
    <tr>
      <td><strong>SOBOL</strong></td>
      <td>Sobol sequence sampling – a quasi-random method for uniform space coverage.</td>
    </tr>
    <tr>
      <td><strong>SAASBO</strong></td>
      <td>SAASBO – Sparse Axis-Aligned Subspace Bayesian Optimization.</td>
    </tr>
    <tr>
      <td><strong>UNIFORM</strong></td>
      <td>Uniform random sampling – purely random selection of parameters.</td>
    </tr>
    <tr>
      <td><strong>LEGACY_GPEI</strong></td>
      <td>Gaussian Process Expected Improvement (legacy version).</td>
    </tr>
    <tr>
      <td><strong>BO_MIXED</strong></td>
      <td>Bayesian Optimization with mixed models – hybrid of multiple models.</td>
    </tr>
    <tr>
      <td><strong>TPE</strong></td>
      <td>[TPE](tutorials?tutorial=tpe) (Tree-structured Parzen Estimator) is a sequential model-based optimization algorithm used for hyperparameter tuning.</td>
    </tr>
    <tr>
      <td><strong>RANDOMFOREST</strong></td>
      <td>[Random Forest model](tutorials?tutorial=randomforest) used for prediction and optimization.</td>
    </tr>
    <tr>
      <td><strong>EXTERNAL_GENERATOR</strong></td>
      <td>A trial configuration provided by an external generator (custom logic).</td>
    </tr>
    <tr>
      <td><strong>BOTORCH_MODULAR</strong></td>
      <td>A modular, flexible BoTorch strategy – highly customizable.</td>
    </tr>
  </tbody>
</table>

These options are color-coded in the GUI for visual clarity.

## Summary

The `results.csv` file is central to understanding what was tried, how it was generated, and how it performed. It enables:

- Tracking progress and failures.
- Analyzing performance across generations.
- Debugging model behavior and identifying patterns.

The combination of metadata and results makes it a valuable artifact in any optimization workflow.

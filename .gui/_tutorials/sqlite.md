# <span class="tutorial_icon invert_in_dark_mode">🗃️</span> Using the SQLite3 store

<!-- What is SQLite3 and how to use it? -->

<!-- Category: Advanced Usage -->

<div id="toc"></div>

## SQLite3 Usage in OmniOpt2

SQLite3 is an optional but always-enabled format for saving OmniOpt2 optimization results. By default, OmniOpt2 automatically writes all trial data and results into a local SQLite3 database file as an archival measure. Users who don't need it can simply ignore it without any impact on their workflow.

### Key Points:

- **Always active but optional to use:** Results are saved in SQLite3, but you can choose whether to query or analyze this data.
- **Archive-focused:** Primarily serves as a persistent backup of results, but users can query or analyze the data however they want.
- **Automatic and transparent:** No extra setup or manual export is needed; saving to SQLite3 happens seamlessly during optimization.

## Benefits of Using SQLite3 for OmniOpt2 Results

Using SQLite3 to save OmniOpt2 optimization results unlocks powerful possibilities for data analysis and aggregation:

### 1. Structured Data Storage
- Results from each trial, including hyperparameters and objective values, are stored in structured tables.
- Enables consistent, organized access to experiment data without manual file parsing.

### 2. Efficient Querying
- You can run complex SQL queries to filter, group, and sort trials based on specific hyperparameter values or objective metrics.
- Example: Find the best trials where learning rate is within a range or select only converged runs.

### 3. Aggregation and Summarization
- Use SQL aggregation functions like `AVG()`, `MAX()`, `MIN()`, `COUNT()`, `GROUP BY` to compute summary statistics.
- Example: Calculate average validation loss per model architecture or maximum accuracy achieved.

### 4. Cross-Experiment Comparison
- Store multiple experiment runs in the same or linked SQLite files to compare different optimization settings side by side.
- Enables easier identification of trends and best hyperparameter configurations across experiments.

### 5. Portability and Integration
- The entire dataset is in a single file, easily transferred or shared with collaborators.
- Many tools and programming languages support SQLite3, enabling integration with Python, R, or visualization tools.

### 6. Post-Processing and Reporting
- Automate report generation or dashboards by querying SQLite3 and feeding results into plots or summary tables.
- Supports custom analyses without modifying the original OmniOpt2 optimization workflow.

In summary, SQLite3 provides a lightweight, yet powerful database option for archiving, analyzing, and aggregating OmniOpt2 hyperparameter optimization results, making it easier to extract insights and improve models.

## How to access

Go to your run folder, and run `sqlite3 database.db`. This allows you to use SQLite3 to view the database.

## Other DB-systems than SQLite3

You can add any other system that `sqlalchemy` supports by creating it's connect-string and passing it to OmniOpt2 with `--db_urls postgresql://user:password@host:port/database mysql+pymysql://user:password@host:port/database ...`. The SQLite3 one will be saved independently. It may be possible that, for certain databases, you need to install further plugins to the [venv](tutorials?tutorial=virtualenv).

## Tables and Meaning of Columns

<table>
<caption>experiment_v2</caption>
<thead><tr><th>Column</th><th>Description</th></tr></thead>
<tbody>
<tr><td>description</td><td>Textual description of the experiment</td></tr>
<tr><td>experiment_type</td><td>Type or category of the experiment</td></tr>
<tr><td>id</td><td>Unique experiment identifier</td></tr>
<tr><td>is_test</td><td>Flag indicating if this is a test experiment</td></tr>
<tr><td>name</td><td>Name of the experiment</td></tr>
<tr><td>properties</td><td>JSON properties/configuration of the experiment</td></tr>
<tr><td>status_quo_name</td><td>Name of the baseline or control setup</td></tr>
<tr><td>status_quo_parameters</td><td>Parameters of the baseline/control setup</td></tr>
<tr><td>time_created</td><td>Timestamp when the experiment was created</td></tr>
<tr><td>default_trial_type</td><td>Default trial type associated with this experiment</td></tr>
<tr><td>default_data_type</td><td>Default data type for trials in the experiment</td></tr>
<tr><td>auxiliary_experiments_by_purpose</td><td>JSON mapping for auxiliary experiments grouped by purpose</td></tr>
</tbody>
</table>

<table>
<caption>generation_strategy</caption>
<thead><tr><th>Column</th><th>Description</th></tr></thead>
<tbody>
<tr><td>id</td><td>Unique identifier for the generation strategy</td></tr>
<tr><td>name</td><td>Name or label of the generation strategy</td></tr>
<tr><td>steps</td><td>Configuration or steps defining the strategy</td></tr>
<tr><td>curr_index</td><td>Current index or step in the strategy sequence</td></tr>
<tr><td>experiment_id</td><td>Associated experiment identifier</td></tr>
<tr><td>nodes</td><td>Details of nodes used in the strategy (e.g. parallel generation nodes)</td></tr>
<tr><td>curr_node_name</td><td>Name of the current node executing</td></tr>
</tbody>
</table>

<table>
<caption>trial_v2</caption>
<thead><tr><th>Column</th><th>Description</th></tr></thead>
<tbody>
<tr><td>abandoned_reason</td><td>Reason why the trial was abandoned (if any)</td></tr>
<tr><td>failed_reason</td><td>Reason why the trial failed (if any)</td></tr>
<tr><td>deployed_name</td><td>Name of the deployed model/configuration</td></tr>
<tr><td>experiment_id</td><td>Identifier of the experiment this trial belongs to</td></tr>
<tr><td>id</td><td>Unique trial identifier</td></tr>
<tr><td>index</td><td>Index or sequence number of the trial</td></tr>
<tr><td>is_batched</td><td>Flag indicating if the trial is batched</td></tr>
<tr><td>lifecycle_stage</td><td>Current lifecycle stage (e.g. running, completed)</td></tr>
<tr><td>num_arms_created</td><td>Number of arms (parameter configurations) created for this trial</td></tr>
<tr><td>ttl_seconds</td><td>Time to live for the trial, in seconds</td></tr>
<tr><td>run_metadata</td><td>JSON metadata about the run</td></tr>
<tr><td>stop_metadata</td><td>JSON metadata about the stop event</td></tr>
<tr><td>status</td><td>Current status (e.g. completed, running, failed)</td></tr>
<tr><td>status_quo_name</td><td>Name of baseline status quo configuration</td></tr>
<tr><td>time_completed</td><td>Timestamp when the trial completed</td></tr>
<tr><td>time_created</td><td>Timestamp when the trial was created</td></tr>
<tr><td>time_staged</td><td>Timestamp when the trial was staged/prepared</td></tr>
<tr><td>time_run_started</td><td>Timestamp when the trial run started</td></tr>
<tr><td>trial_type</td><td>Type of trial (e.g. default, special)</td></tr>
<tr><td>generation_step_index</td><td>Index of the generation step used</td></tr>
<tr><td>properties</td><td>Additional JSON properties for the trial</td></tr>
</tbody>
</table>

<table>
<caption>analysis_card</caption>
<thead><tr><th>Column</th><th>Description</th></tr></thead>
<tbody>
<tr><td>id</td><td>Unique identifier of the analysis card</td></tr>
<tr><td>name</td><td>Name of the analysis card</td></tr>
<tr><td>title</td><td>Title text for the card</td></tr>
<tr><td>subtitle</td><td>Subtitle text for the card</td></tr>
<tr><td>level</td><td>Level or hierarchy depth of the card</td></tr>
<tr><td>dataframe_json</td><td>JSON representing data in tabular format for analysis</td></tr>
<tr><td>blob</td><td>Binary large object, e.g. charts or serialized data</td></tr>
<tr><td>blob_annotation</td><td>Annotation or metadata for the blob</td></tr>
<tr><td>time_created</td><td>Timestamp when the analysis card was created</td></tr>
<tr><td>experiment_id</td><td>ID of the experiment associated</td></tr>
<tr><td>attributes</td><td>JSON of additional attributes or metadata</td></tr>
<tr><td>category</td><td>Category or grouping of the analysis card</td></tr>
</tbody>
</table>

<table>
<caption>abandoned_arm_v2</caption>
<thead><tr><th>Column</th><th>Description</th></tr></thead>
<tbody>
<tr><td>abandoned_reason</td><td>Reason why the arm (parameter configuration) was abandoned</td></tr>
<tr><td>id</td><td>Unique identifier of the abandoned arm</td></tr>
<tr><td>name</td><td>Name or label of the abandoned arm</td></tr>
<tr><td>time_abandoned</td><td>Timestamp when it was abandoned</td></tr>
<tr><td>trial_id</td><td>Identifier of the trial this arm belonged to</td></tr>
</tbody>
</table>

<table>
<caption>generator_run_v2</caption>
<thead><tr><th>Column</th><th>Description</th></tr></thead>
<tbody>
<tr><td>best_arm_name</td><td>Name of the best arm (parameter set) from this generation run</td></tr>
<tr><td>best_arm_parameters</td><td>JSON of parameters of the best arm</td></tr>
<tr><td>best_arm_predictions</td><td>JSON of predictions for the best arm</td></tr>
<tr><td>generator_run_type</td><td>Type of the generator run (e.g. optimization algorithm)</td></tr>
<tr><td>id</td><td>Unique identifier of the generator run</td></tr>
<tr><td>index</td><td>Index of this generator run in the trial</td></tr>
<tr><td>model_predictions</td><td>JSON predictions produced by the model</td></tr>
<tr><td>time_created</td><td>Timestamp when the generator run was created</td></tr>
<tr><td>trial_id</td><td>Trial ID associated with this generator run</td></tr>
<tr><td>weight</td><td>Weight or importance assigned to this generator run</td></tr>
<tr><td>fit_time</td><td>Time spent fitting the model (seconds)</td></tr>
<tr><td>gen_time</td><td>Time spent generating arms (seconds)</td></tr>
<tr><td>model_key</td><td>Key identifying the model used</td></tr>
<tr><td>model_kwargs</td><td>JSON of model-specific parameters</td></tr>
<tr><td>bridge_kwargs</td><td>JSON of parameters related to bridging strategies</td></tr>
<tr><td>gen_metadata</td><td>JSON metadata about the generation process</td></tr>
</tbody>
</table>

<table>
<caption>arm_v2</caption>
<thead><tr><th>Column</th><th>Description</th></tr></thead>
<tbody>
<tr><td>experiment_id</td><td>ID of the experiment this arm belongs to</td></tr>
<tr><td>id</td><td>Unique identifier of the arm</td></tr>
<tr><td>name</td><td>Name of the arm</td></tr>
<tr><td>parameters</td><td>JSON of parameter values for this arm</td></tr>
<tr><td>trial_id</td><td>ID of the trial associated</td></tr>
<tr><td>time_created</td><td>Timestamp when the arm was created</td></tr>
<tr><td>time_removed</td><td>Timestamp when the arm was removed (if any)</td></tr>
<tr><td>properties</td><td>Additional JSON properties for the arm</td></tr>
</tbody>
</table>

<table>
<caption>generation_step</caption>
<thead><tr><th>Column</th><th>Description</th></tr></thead>
<tbody>
<tr><td>id</td><td>Unique identifier of the generation step</td></tr>
<tr><td>generation_strategy_id</td><td>ID of the generation strategy</td></tr>
<tr><td>index</td><td>Index/order of the generation step in the strategy</td></tr>
<tr><td>name</td><td>Name or label of the generation step</td></tr>
<tr><td>type</td><td>Type of generation step (e.g. initial, iterative)</td></tr>
<tr><td>model_key</td><td>Model key used for this step</td></tr>
<tr><td>model_kwargs</td><td>JSON of model parameters</td></tr>
<tr><td>bridge_kwargs</td><td>JSON of bridging parameters</td></tr>
<tr><td>use_update</td><td>Flag if this step uses update mechanisms</td></tr>
<tr><td>minimum_trials_observed</td><td>Minimum number of trials before this step can run</td></tr>
<tr><td>is_dedicated</td><td>Flag if step is dedicated to specific tasks</td></tr>
<tr><td>should_save</td><td>Flag indicating if results from this step should be saved</td></tr>
<tr><td>is_active</td><td>Flag indicating if this step is currently active</td></tr>
</tbody>
</table>

<table>
<caption>data_row</caption>
<thead><tr><th>Column</th><th>Description</th></tr></thead>
<tbody>
<tr><td>id</td><td>Unique identifier of the data row</td></tr>
<tr><td>data</td><td>JSON or serialized data content</td></tr>
<tr><td>trial_id</td><td>Associated trial identifier</td></tr>
<tr><td>arm_name</td><td>Name of the arm corresponding to this data row</td></tr>
<tr><td>time_created</td><td>Timestamp when the data row was created</td></tr>
</tbody>
</table>

## Example data

# `experiment_v2`

<table>
  <thead>
    <tr>
      <th>description</th>
      <th>experiment_type</th>
      <th>id</th>
      <th>is_test</th>
      <th>name</th>
      <th>properties</th>
      <th>status_quo_name</th>
      <th>status_quo_parameters</th>
      <th>time_created</th>
      <th>default_trial_type</th>
      <th>default_data_type</th>
      <th>auxiliary_experiments_by_purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td></td>
      <td>1</td>
      <td>0</td>
      <td>__main__tests__BOTORCH_MODULAR___nogridsearch</td>
      <td>{"immutable_search_space_and_opt_config": true}</td>
      <td></td>
      <td></td>
      <td>1750172209</td>
      <td></td>
      <td>1</td>
      <td>{}</td>
    </tr>
  </tbody>
</table>

### `generation_strategy` (empty)

<table>
  <thead>
    <tr>
      <th>id</th>
      <th>name</th>
      <th>steps</th>
      <th>curr_index</th>
      <th>experiment_id</th>
      <th>nodes</th>
      <th>curr_node_name</th>
    </tr>
  </thead>
  <tbody>
    <!-- Keine Daten vorhanden -->
  </tbody>
</table>

### `trial_v2`

<table>
  <thead>
    <tr>
      <th>abandoned_reason</th>
      <th>failed_reason</th>
      <th>deployed_name</th>
      <th>experiment_id</th>
      <th>id</th>
      <th>index</th>
      <th>is_batched</th>
      <th>lifecycle_stage</th>
      <th>num_arms_created</th>
      <th>ttl_seconds</th>
      <th>run_metadata</th>
      <th>stop_metadata</th>
      <th>status</th>
      <th>status_quo_name</th>
      <th>time_completed</th>
      <th>time_created</th>
      <th>time_staged</th>
      <th>time_run_started</th>
      <th>trial_type</th>
      <th>generation_step_index</th>
      <th>properties</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td>1</td>
      <td></td>
      <td>{}</td>
      <td>{}</td>
      <td>3</td>
      <td></td>
      <td>1750172230</td>
      <td>1750172214</td>
      <td></td>
      <td>1750172214</td>
      <td></td>
      <td></td>
      <td>{}</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td></td>
      <td>1</td>
      <td></td>
      <td>{}</td>
      <td>{}</td>
      <td>3</td>
      <td></td>
      <td>1750172249</td>
      <td>1750172237</td>
      <td></td>
      <td>1750172237</td>
      <td></td>
      <td></td>
      <td>{}</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td></td>
      <td>1</td>
      <td></td>
      <td>{}</td>
      <td>{}</td>
      <td>3</td>
      <td></td>
      <td>1750172274</td>
      <td>1750172260</td>
      <td></td>
      <td>1750172260</td>
      <td></td>
      <td></td>
      <td>{}</td>
    </tr>
  </tbody>
</table>

### `analysis_card` (empty)

<table>
  <thead>
    <tr>
      <th>id</th>
      <th>name</th>
      <th>title</th>
      <th>subtitle</th>
      <th>level</th>
      <th>dataframe_json</th>
      <th>blob</th>
      <th>blob_annotation</th>
      <th>time_created</th>
      <th>experiment_id</th>
      <th>attributes</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <!-- Keine Daten vorhanden -->
  </tbody>
</table>

### `abandoned_arm_v2` (empty)

<table>
  <thead>
    <tr>
      <th>abandoned_reason</th>
      <th>id</th>
      <th>name</th>
      <th>time_abandoned</th>
      <th>trial_id</th>
    </tr>
  </thead>
  <tbody>
    <!-- Keine Daten vorhanden -->
  </tbody>
</table>

### `generator_run_v2`

<table>
  <thead>
    <tr>
      <th>best_arm_name</th>
      <th>best_arm_parameters</th>
      <th>best_arm_predictions</th>
      <th>generator_run_type</th>
      <th>id</th>
      <th>index</th>
      <th>model_predictions</th>
      <th>time_created</th>
      <th>trial_id</th>
      <th>weight</th>
      <th>fit_time</th>
      <th>gen_time</th>
      <th>model_key</th>
      <th>model_kwargs</th>
      <th>bridge_kwargs</th>
      <th>gen_metadata</th>
      <th>model_state_after_gen</th>
      <th>generation_strategy_id</th>
      <th>generation_step_index</th>
      <th>candidate_metadata_by_arm_signature</th>
      <th>generation_node_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>1</td>
      <td>0</td>
      <td></td>
      <td>1750172214</td>
      <td>1</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>SOBOL</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>2</td>
      <td>0</td>
      <td></td>
      <td>1750172237</td>
      <td>2</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>SOBOL</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>3</td>
      <td>0</td>
      <td></td>
      <td>1750172260</td>
      <td>3</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>BOTORCH_MODULAR</td>
    </tr>
  </tbody>
</table>

### `runner` (empty)

<table>
  <thead>
    <tr>
      <th>id</th>
      <th>experiment_id</th>
      <th>properties</th>
      <th>runner_type</th>
      <th>trial_id</th>
      <th>trial_type</th>
    </tr>
  </thead>
  <tbody>
    <!-- Keine Daten vorhanden -->
  </tbody>
</table>

### `data_v2`

<table>
  <thead>
    <tr>
      <th>id</th>
      <th>data_json</th>
      <th>description</th>
      <th>experiment_id</th>
      <th>time_created</th>
      <th>trial_index</th>
      <th>generation_strategy_id</th>
      <th>structure_metadata_json</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>{"trial_index":{"0":0},"arm_name":{"0":"0_0"},"metric_name":{"0":"RESULT"},"mean":{"0":-93199.1948576998},"sem":{"0":null}}</td>
      <td>1</td>
      <td>1750172230632</td>
      <td>0</td>
      <td></td>
      <td>0</td>
      <td>{"df": {"__type": "DataFrame", "value": "{\"trial_index\":{\"0\":0},\"arm_name\":{\"0\":\"0_0\"},\"metric_name\":{\"0\":\"RESULT\"},\"mean\":{\"0\":-93199.1948576998},\"sem\":{\"0\":null}}"}, "description": null}</td>
    </tr>
    <tr>
      <td>2</td>
      <td>{"trial_index":{"0":1},"arm_name":{"0":"1_0"},"metric_name":{"0":"RESULT"},"mean":{"0":-79151.8159297052},"sem":{"0":null}}</td>
      <td>1</td>
      <td>1750172249564</td>
      <td>1</td>
      <td></td>
      <td>0</td>
      <td>{"df": {"__type": "DataFrame", "value": "{\"trial_index\":{\"0\":1},\"arm_name\":{\"0\":\"1_0\"},\"metric_name\":{\"0\":\"RESULT\"},\"mean\":{\"0\":-79151.8159297052},\"sem\":{\"0\":null}}"}, "description": null}</td>
    </tr>
    <tr>
      <td>3</td>
      <td>{"trial_index":{"0":2},"arm_name":{"0":"2_0"},"metric_name":{"0":"RESULT"},"mean":{"0":-111380.52},"sem":{"0":null}}</td>
      <td>1</td>
      <td>1750172274070</td>
      <td>2</td>
      <td></td>
      <td>0</td>
      <td>{"df": {"__type": "DataFrame", "value": "{\"trial_index\":{\"0\":2},\"arm_name\":{\"0\":\"2_0\"},\"metric_name\":{\"0\":\"RESULT\"},\"mean\":{\"0\":-111380.52},\"sem\":{\"0\":null}}"}, "description": null}</td>
    </tr>
  </tbody>
</table>

### `parameter_v2`

<table>
  <thead>
    <tr>
      <th>domain_type</th>
      <th>experiment_id</th>
      <th>id</th>
      <th>generator_run_id</th>
      <th>name</th>
      <th>parameter_type</th>
      <th>is_fidelity</th>
      <th>target_value</th>
      <th>digits</th>
      <th>log_scale</th>
      <th>lower</th>
      <th>upper</th>
      <th>choice_values</th>
      <th>is_ordered</th>
      <th>is_task</th>
      <th>dependents</th>
      <th>fixed_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td></td>
      <td>int_param</td>
      <td>1</td>
      <td>0</td>
      <td></td>
      <td>0</td>
      <td>-100.0</td>
      <td>10.0</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td></td>
      <td>float_param</td>
      <td>2</td>
      <td>0</td>
      <td></td>
      <td>0</td>
      <td>-100.0</td>
      <td>10.0</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td></td>
      <td>choice_param</td>
      <td>3</td>
      <td>0</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>["1", "2", "4", "8", "16", "hallo"]</td>
      <td>0</td>
      <td>0</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td></td>
      <td>int_param_two</td>
      <td>1</td>
      <td>0</td>
      <td></td>
      <td>0</td>
      <td>-100.0</td>
      <td>10.0</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

### `parameter_constraint_v2`

<table>
  <thead>
    <tr>
      <th>bound</th>
      <th>constraint_dict</th>
      <th>experiment_id</th>
      <th>id</th>
      <th>generator_run_id</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <!-- Keine Daten vorhanden -->
  </tbody>
</table>

### `metric_v2`

<table>
  <thead>
    <tr>
      <th>experiment_id</th>
      <th>generator_run_id</th>
      <th>id</th>
      <th>lower_is_better</th>
      <th>intent</th>
      <th>metric_type</th>
      <th>name</th>
      <th>properties</th>
      <th>minimize</th>
      <th>op</th>
      <th>bound</th>
      <th>relative</th>
      <th>trial_type</th>
      <th>canonical_name</th>
      <th>scalarized_objective_id</th>
      <th>scalarized_objective_weight</th>
      <th>scalarized_outcome_constraint_id</th>
      <th>scalarized_outcome_constraint_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td></td>
      <td>1</td>
      <td>1</td>
      <td>objective</td>
      <td>0</td>
      <td>RESULT</td>
      <td>{"name": "RESULT", "lower_is_better": true, "properties": {}}</td>
      <td>1</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

### `arm_v2`

<table>
  <thead>
    <tr>
      <th>generator_run_id</th>
      <th>id</th>
      <th>name</th>
      <th>parameters</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>0_0</td>
      <td>{"int_param": 1, "float_param": -69.76193875074387, "int_param_two": -56, "choice_param": "hallo"}</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>1_0</td>
      <td>{"int_param": -66, "float_param": -32.97011447139084, "int_param_two": -31, "choice_param": "16"}</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>2_0</td>
      <td>{"int_param": 10, "float_param": -100.0, "int_param_two": 10, "choice_param": "hallo"}</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>

# 🗃️ Using the SQLite3 store

<!-- What is SQLite3 and how to use it? -->

<!-- Category: Advanced Usage -->

<div id="toc"></div>

## SQLite3 Usage in OmniOpt2

SQLite3 is an optional but always-enabled format for saving OmniOpt2 optimization results. By default, OmniOpt2 automatically writes all trial data and results into a local SQLite3 database file as an archival measure. Users who don't need it can simply ignore it without any impact on their workflow.

### Key Points

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

You can add any other system that `sqlalchemy` supports by creating it's connect-string and passing it to OmniOpt2 with `--db_url`.

These include, but are not limited to `postgresql://user:password@host:port/database`, `mysql+pymysql://user:password@host:port/database ...` and so on.

It may be possible that, for certain databases, you need to install further plugins to the [venv](tutorials?tutorial=virtualenv).

Using this option disables the saving into the SQLite3 database.

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

This command was run:

```bash
omniopt \
	--live_share \
	--send_anonymized_usage_stats \
	--partition alpha \
	--experiment_name ExampleDatabase \
	--mem_gb=4 \
	--time 60 \
	--worker_timeout=5 \
	--max_eval 4 \
	--num_parallel_jobs 2 \
	--gpus 0 \
	--run_program Li8udGVzdHMvb3B0aW1pemF0aW9uX2V4YW1wbGUgIC0taW50X3BhcmFtPSclKGludF9wYXJhbSknIC0tZmxvYXRfcGFyYW09JyUoZmxvYXRfcGFyYW0pJyAtLWNob2ljZV9wYXJhbT0nJShjaG9pY2VfcGFyYW0pJyAtLWludF9wYXJhbV90d289JyUoaW50X3BhcmFtX3R3byknIC0tbnJfcmVzdWx0cz0x \
	--parameter int_param range -100 10 int \
	--parameter float_param range -100 10 float \
	--parameter choice_param choice 1,2,4,8,16,hallo \
	--parameter int_param_two range -100 10 int \
	--num_random_steps 2 \
	--model BOTORCH_MODULAR \
	--auto_exclude_defective_hosts \
	--generate_all_jobs_at_once \
	--follow \
	--experiment_constraints MjAqaW50X3BhcmFtID49IDEwMAo= \
	--show_generate_time_table

```

And resulted in this database:

```run_php
/**
 * Dump every SQLite table (with headers) as HTML.
 *
 * Usage: place this file in the same directory as database.sqlite3
 * and open it in the browser through your PHP server.
 */

$dbFile = __DIR__ . '/database.db';
if (!is_readable($dbFile)) {
    die("Database file not found: $dbFile");
}

$db = new SQLite3($dbFile, SQLITE3_OPEN_READONLY);

// 1. fetch all user tables (omit sqlite internal tables)
$tablesStmt = $db->query("
    SELECT name
    FROM sqlite_master
    WHERE type='table'
      AND name NOT LIKE 'sqlite_%'
    ORDER BY name;
");

while ($tableRow = $tablesStmt->fetchArray(SQLITE3_ASSOC)) {
    $table = $tableRow['name'];
    echo "<h3>" . htmlspecialchars($table, ENT_QUOTES, 'UTF-8') . "</h3>\n";

    // 2. fetch rows
    $dataStmt = @$db->query("SELECT * FROM \"$table\"");
    if (!$dataStmt) {
        echo "<p><em>Could not read data.</em></p>\n";
        continue;
    }

    // 3. print table
    echo "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;margin-bottom:2em;'>\n";

    // column headers
    $firstRow = $dataStmt->fetchArray(SQLITE3_ASSOC);
    if ($firstRow !== false) {
        echo "  <tr>";
        foreach (array_keys($firstRow) as $col) {
            echo "<th>" . htmlspecialchars($col, ENT_QUOTES, 'UTF-8') . "</th>";
        }
        echo "</tr>\n";

        // first row data
        echo "  <tr>";
        foreach ($firstRow as $val) {
            echo "<td>" . htmlspecialchars(strval($val), ENT_QUOTES, 'UTF-8') . "</td>";
        }
        echo "</tr>\n";

        // remaining rows
        while ($row = $dataStmt->fetchArray(SQLITE3_ASSOC)) {
            echo "  <tr>";
            foreach ($row as $val) {
                echo "<td>" . htmlspecialchars(strval($val), ENT_QUOTES, 'UTF-8') . "</td>";
            }
            echo "</tr>\n";
        }
    } else {
        // empty table
        echo "  <tr><td><em>No rows</em></td></tr>\n";
    }

    echo "</table>\n";
}

$db->close();
```

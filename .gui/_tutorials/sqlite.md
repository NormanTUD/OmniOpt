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

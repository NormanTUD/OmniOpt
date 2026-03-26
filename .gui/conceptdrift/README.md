# Concept Drift Benchmark Web Platform

This repository contains an interactive web platform for exploring the computational performance of semi-supervised and unsupervised concept drift detectors under multi-objective settings.

The website supports experiment inspection at three levels:
- single detector per dataset,
- cross-detector Pareto-front analysis,
- ensemble-oriented visualization modules.

The platform is designed as a static website and can be served locally without a backend service.

## Scientific Context

The website demonstrates the benchmark study:

**Computational Performance of Semi- and Unsupervised Concept Drift Detection: A Survey and Multiobjective Benchmark using Bayesian Optimization**

It provides a reproducible and inspectable interface for:
- objective trade-off analysis,
- Pareto-optimal configuration exploration,
- detector-wise and dataset-wise comparison,
- downloadable benchmark result files.

## Website Sections

The interface has four top-level navigation entries:

1. **Home**  
Landing page with title and authorship information.

2. **Single Detector Experiments**  
Detailed experiment analysis for one `(detector, dataset, metric, model)` selection at a time, including:
- summary cards,
- scatter plot(s) with Pareto highlighting,
- parallel coordinates plot,
- experiment parameter and overview tables,
- interactive result table with sorting/pagination,
- dataset- and detector-scoped download actions.

3. **Multiple Detector Pareto Fronts**  
Aggregated Pareto analysis across all detectors for a selected dataset and metric family:
- ACC_RT and MTR_RT: 2D scatter,
- ACC_RT_REQL: 3D and 2D views.

4. **Ensemble Estimation**  
Two visualization modes:
- **Ensemble of DDs** (dataset/model selectable),
- **Single Variate DDs on Multiple Features**.

Both are embedded as iframe-based views backed by static assets.

## Data and Directory Structure

The current website uses the `data/` directory as the runtime data root:

```text
data/
  all_benchmark_results/
    <detector>/
      <dataset>/
        <detector>_<dataset>_ACC_RT.csv
        <detector>_<dataset>_ACC_RT_REQL.csv
        <detector>_<dataset>_MTR_RT.csv
  overview_data/
    <detector>/
      <dataset>/
        <detector>_<dataset>_ACC_RT_overview.json
        <detector>_<dataset>_ACC_RT_REQL_overview.json
        <detector>_<dataset>_MTR_RT_overview.json
  ensemble_estimation/
    ensemble_DDs/
      ensemble.html
      single_predictions/
    sv_mf/
      sv_mf.html
      single_predictions/
```

### Runtime Data Usage

- **CSV files** in `data/all_benchmark_results/` are the primary source for plotted and tabulated result values.
- **Overview JSON files** in `data/overview_data/` provide the precomputed overview tables shown in the single-detector section.
- **Ensemble visual modules** are loaded from `data/ensemble_estimation/...`.

## Metric Families

The website uses three metric families:

- `ACCURACY-RUNTIME` (ACC_RT) for real datasets
- `ACCURACY-RUNTIME-REQLABELS` (ACC_RT_REQL) for real datasets
- `MEANTIMERATIO-RUNTIME` (MTR_RT) for synthetic datasets

Metric availability is dataset-aware in the selectors.

## Implementation Notes

- Main entry page: `index.html`
- Main interaction logic: `script.js`
- Main styles: `styles.css`, `expstyle.css`
- Reusable UI components: `components/`
  - `components/dropdown/`
  - `components/show_button/`

External libraries are loaded by CDN at runtime (e.g., Plotly, jQuery, Grid.js, JSZip).

## Running Locally

Because the project uses dynamic `fetch()` calls for CSV/JSON/HTML assets, run it through a local HTTP server (do not open `index.html` directly from filesystem).

### Option 1 (recommended): Python built-in server

From the project root:

```powershell
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/
```

### Option 2: any static server

Any equivalent static file server is suitable, as long as it serves the repository root.


## Reproducibility and Maintenance

- Keep file naming conventions unchanged for CSV and overview JSON files.
- Preserve detector and dataset directory names exactly as referenced by selectors.
- If adding new datasets/metrics, update selector options and path builders consistently in `index.html` and `script.js`.

## Contact

For scientific or technical inquiries, please use the contact route shown in the website footer.

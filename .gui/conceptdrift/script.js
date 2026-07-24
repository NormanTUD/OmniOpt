const REAL_DATASETS = new Set([
  'Electricity',
  'ForestCovertype',
  'GasSensor',
  'NOAAWeather',
  'OutdoorObjects',
  'Ozone',
  'PokerHand',
  'RialtoBridgeTimelapse',
  'SensorStream'
]);

// 

const SYNTHETIC_DATASETS = new Set([
  'SineClusters',
  'WaveformDrift2'
]);

const REAL_METRIC_VALUES = [
  'ACCURACY-RUNTIME',
  'ACCURACY-RUNTIME-REQLABELS'
];

const SYNTHETIC_METRIC_VALUE = 'MEANTIMERATIO-RUNTIME';

function setCustomDropdownValue(dropdownName, value) {
  const $dropdown = $(`custom-dropdown[name="${dropdownName}"]`);
  const $container = $dropdown.find('.dropdown-container');
  const $label = $container.find('.dropdown-label');
  const $input = $container.find('input[type="hidden"]');
  const $content = $container.find('.dropdown-content');
  const $option = $content.find(`p[data-value="${value}"]`).first();

  if (!$dropdown.length || !$option.length) {
    return;
  }

  const useExternalLabel = $dropdown.is('[external-label]');
  const baseText = $label.data('base') || $label.text().split(':')[0].trim();
  const text = $option.text();

  $content.find('p').removeClass('selected');
  $option.addClass('selected');

  $label.text(useExternalLabel ? text : `${baseText}: ${text}`);
  $input.val(value).trigger('change');
}

function updateSingleDDMetricAvailability() {
  const datasetValue = $('custom-dropdown[name="dataset"] input[type="hidden"]').val();
  const $metricsDropdown = $('custom-dropdown[name="metrics"]');
  const $metricsContent = $metricsDropdown.find('.dropdown-content');

  if (!$metricsDropdown.length || !$metricsContent.length || !datasetValue) {
    return;
  }

  const isSyntheticDataset = SYNTHETIC_DATASETS.has(datasetValue);

  REAL_METRIC_VALUES.forEach(metricValue => {
    const $option = $metricsContent.find(`p[data-value="${metricValue}"]`);
    $option.toggleClass('disabled-option', isSyntheticDataset);
  });

  const $syntheticOption = $metricsContent.find(`p[data-value="${SYNTHETIC_METRIC_VALUE}"]`);
  $syntheticOption.toggleClass('disabled-option', !isSyntheticDataset);

  if (isSyntheticDataset) {
    setCustomDropdownValue('metrics', SYNTHETIC_METRIC_VALUE);
  } else {
    setCustomDropdownValue('metrics', REAL_METRIC_VALUES[0]);
  }
}

function updateParetoMetricAvailability() {
  const datasetValue = $('custom-dropdown[name="dataset_par"] input[type="hidden"]').val();
  const $metricsDropdown = $('custom-dropdown[name="metrics_par"]');
  const $metricsContent = $metricsDropdown.find('.dropdown-content');

  if (!$metricsDropdown.length || !$metricsContent.length || !datasetValue) {
    return;
  }

  const isSyntheticDataset = SYNTHETIC_DATASETS.has(datasetValue);

  REAL_METRIC_VALUES.forEach(metricValue => {
    const $option = $metricsContent.find(`p[data-value="${metricValue}"]`);
    $option.toggleClass('disabled-option', isSyntheticDataset);
  });

  const $syntheticOption = $metricsContent.find(`p[data-value="${SYNTHETIC_METRIC_VALUE}"]`);
  $syntheticOption.toggleClass('disabled-option', !isSyntheticDataset);

  if (isSyntheticDataset) {
    setCustomDropdownValue('metrics_par', SYNTHETIC_METRIC_VALUE);
  } else {
    setCustomDropdownValue('metrics_par', REAL_METRIC_VALUES[0]);
  }
}

document.addEventListener('DOMContentLoaded', function () {
  const $datasetInput = $('custom-dropdown[name="dataset"] input[type="hidden"]');
  const $paretoDatasetInput = $('custom-dropdown[name="dataset_par"] input[type="hidden"]');

  if ($datasetInput.length) {
    $datasetInput.on('change', updateSingleDDMetricAvailability);
    updateSingleDDMetricAvailability();
  }

  if ($paretoDatasetInput.length) {
    $paretoDatasetInput.on('change', updateParetoMetricAvailability);
    updateParetoMetricAvailability();
  }
});


// 



const DETECTOR_OPTIONS = [
  'BNDM',
  'CDBD',
  'CDLEEDS',
  'CSDDM',
  'D3',
  'DAWIDD',
  'DDAL',
  'EDFS',
  'HDDDM',
  'IBDD',
  'IKS',
  'NNDVI',
  'OCDD',
  'PCACD',
  'SlidShaps',
  'SPLL',
  'STUDD',
  'UCDD',
  'UDetect',
  'WindowKDE'
];

const DETECTOR_COLOR_SCHEME = {
  CSDDM: '#1f77b4',
  BNDM: '#ff7f0e',
  D3: '#2ca02c',
  IBDD: '#d62728',
  OCDD: '#9467bd',
  SPLL: '#8c564b',
  UDetect: '#e377c2',
  EDFS: '#7f7f7f',
  NNDVI: '#bcbd22',
  UCDD: '#17becf',
  STUDD: '#ffbb78',
  DDAL: '#ff9896',
  DAWIDD: '#c5b0d5',
  IKS: '#c49c94',
  HDDDM: '#f7b6d2',
  PCACD: '#dbdb8d',
  CDBD: '#9edae5',
  CDLEEDS: '#f5b0b0',
  SlidShaps: '#f7f7f7',
  WindowKDE: '#e5e5e5'
};

function isCsvBackedAccuracyRuntimeSelection(dataset, metrics) {
  return REAL_DATASETS.has(dataset) && metrics === 'ACCURACY-RUNTIME';
}

function parseCsvText(csvText) {
  const lines = csvText
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    return { headers: [], rows: [] };
  }

  const headers = lines[0].split(',').map(header => header.trim());
  const rows = lines.slice(1).map(line => {
    const values = line.split(',').map(value => value.trim());
    const row = {};

    headers.forEach((header, index) => {
      const rawValue = values[index] ?? '';
      const numericValue = Number(rawValue);
      row[header] = rawValue !== '' && !Number.isNaN(numericValue) ? numericValue : rawValue;
    });

    return row;
  });

  return { headers, rows };
}

function computeParetoFront(rows) {
  const sortableRows = rows
    .filter(row => typeof row.ACCURACY === 'number' && typeof row.RUNTIME === 'number')
    .slice()
    .sort((left, right) => {
      if (left.RUNTIME !== right.RUNTIME) {
        return left.RUNTIME - right.RUNTIME;
      }
      return right.ACCURACY - left.ACCURACY;
    });

  const pareto = [];
  let bestAccuracy = -Infinity;

  sortableRows.forEach(row => {
    if (row.ACCURACY > bestAccuracy) {
      pareto.push(row);
      bestAccuracy = row.ACCURACY;
    }
  });

  return pareto;
}

function computeMetricRuntimeParetoFront(rows, metricKey) {
  const sortableRows = rows
    .filter(row => typeof row[metricKey] === 'number' && typeof row.RUNTIME === 'number')
    .slice()
    .sort((left, right) => {
      if (left.RUNTIME !== right.RUNTIME) {
        return left.RUNTIME - right.RUNTIME;
      }
      return right[metricKey] - left[metricKey];
    });

  const pareto = [];
  let bestMetric = -Infinity;

  sortableRows.forEach(row => {
    if (row[metricKey] > bestMetric) {
      pareto.push(row);
      bestMetric = row[metricKey];
    }
  });

  return pareto;
}

function formatMetricValue(value, digits = 3) {
  return typeof value === 'number' ? value.toFixed(digits) : 'N/A';
}

function buildPointDetailText(row, headers) {
  const parameterSummary = headers
    .filter(header => !['Status', 'ACCURACY', 'RUNTIME', 'REQLABELS', 'MTR'].includes(header))
    .map(header => `${header}: ${row[header]}`)
    .join('<br>');

  return [
    typeof row.ACCURACY === 'number' ? `Accuracy: ${row.ACCURACY}` : null,
    typeof row.RUNTIME === 'number' ? `Runtime: ${row.RUNTIME}` : null,
    typeof row.REQLABELS === 'number' ? `ReqLabels: ${row.REQLABELS}` : null,
    typeof row.MTR === 'number' ? `MTR: ${row.MTR}` : null,
    parameterSummary || null
  ].filter(Boolean).join('<br>');
}

function buildParetoPointDetailText(row, headers, detector) {
  const parameterSummary = headers
    .filter(header => !['Status', 'ACCURACY', 'RUNTIME', 'REQLABELS', 'MTR'].includes(header))
    .map(header => `${header}: ${row[header]}`)
    .join('<br>');

  return [
    detector ? `Detector: ${detector}` : null,
    typeof row.ACCURACY === 'number' ? `Accuracy: ${row.ACCURACY}` : null,
    typeof row.RUNTIME === 'number' ? `Runtime: ${row.RUNTIME}` : null,
    typeof row.REQLABELS === 'number' ? `ReqLabels: ${row.REQLABELS}` : null,
    typeof row.MTR === 'number' ? `MTR: ${row.MTR}` : null,
    parameterSummary || null
  ].filter(Boolean).join('<br>');
}

function computeAxisRange(values, { floor = null, ceiling = null } = {}) {
  const numericValues = values.filter(value => typeof value === 'number' && !Number.isNaN(value));
  if (!numericValues.length) {
    return null;
  }

  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);
  const spread = max - min;
  const padding = spread > 0 ? spread * 0.08 : Math.max(Math.abs(max) * 0.08, 0.05);

  let lower = min - padding;
  let upper = max + padding;

  if (floor != null) {
    lower = Math.max(floor, lower);
  }
  if (ceiling != null) {
    upper = Math.min(ceiling, upper);
  }

  if (lower === upper) {
    upper = lower + 1;
  }

  return [lower, upper];
}

function applyCommonPlotLayout(layout = {}) {
  return {
    ...layout,
    hoverlabel: {
      bgcolor: '#ffffff',
      bordercolor: '#ced4da',
      font: {
        family: 'Inter, system-ui, -apple-system, "Segoe UI", sans-serif',
        size: 13,
        color: '#002D3F'
      },
      ...(layout.hoverlabel || {})
    }
  };
}

function ensureCsvResultStyles() {
  if (document.getElementById('csv-results-inline-styles')) {
    return;
  }

  const style = document.createElement('style');
  style.id = 'csv-results-inline-styles';
  style.textContent = `
    .csv-summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }

    .csv-summary-card {
      background: white;
      border: 1px solid #e9ecef;
      border-radius: 10px;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
      padding: 16px;
      text-align: center;
    }

    .csv-summary-label {
      color: #6c757d;
      font-size: 0.85rem;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }

    .csv-summary-value {
      color: #002D3F;
      font-size: 1.6rem;
      font-weight: 600;
    }

    .csv-plot-block {
      margin-bottom: 30px;
    }

    .csv-table-wrapper {
      background: white;
      border: 1px solid #e9ecef;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
      margin-bottom: 18px;
    }

    .csv-table-scroll {
      overflow-x: auto;
    }

    .csv-results-table {
      width: 100%;
      border-collapse: collapse;
      margin: 0;
    }

    .csv-results-table th.sortable-header {
      cursor: pointer;
      position: relative;
      padding-right: 30px !important;
    }

    .csv-results-table th.sortable-header::after {
      content: attr(data-sort-indicator);
      position: absolute;
      right: 12px;
      top: 50%;
      transform: translateY(-50%);
      font-size: 0.8rem;
      opacity: 0.9;
    }

    .csv-table-pagination {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      padding: 12px 16px;
      border-top: 1px solid #e9ecef;
      background: #f8f9fa;
    }

    .csv-table-pagination button {
      background: #015E80;
      color: white;
      border: none;
      border-radius: 6px;
      padding: 6px 12px;
      cursor: pointer;
      font-size: 0.9rem;
    }

    .csv-table-pagination button:disabled {
      background: #adb5bd;
      cursor: not-allowed;
    }

    .csv-table-status {
      color: #495057;
      font-size: 0.9rem;
    }

    .csv-muted-note {
      color: #6c757d;
      margin-bottom: 18px;
    }

    .csv-download-group {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 10px;
    }

    .csv-download-button {
      background: #015E80;
      color: white;
      border: none;
      border-radius: 8px;
      padding: 10px 16px;
      cursor: pointer;
      font-size: 0.95rem;
      transition: background 0.2s ease;
    }

    .csv-download-button:hover {
      background: #016A8E;
    }

    .csv-download-button:disabled {
      background: #adb5bd;
      cursor: not-allowed;
    }

    .csv-parallel-controls {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }

    .csv-parallel-controls label {
      margin: 0;
      font-weight: 600;
      color: #002D3F;
    }

    .csv-parallel-controls select {
      appearance: none;
      -webkit-appearance: none;
      -moz-appearance: none;
      padding: 10px 14px;
      padding-right: 42px;
      min-height: 46px;
      border-radius: 6px;
      border: 1px solid #ced4da;
      min-width: 260px;
      width: auto;
      font-size: 0.96rem;
      line-height: 1.25;
      color: #002D3F;
      background: #ffffff;
      background-image: linear-gradient(45deg, transparent 50%, #002D3F 50%), linear-gradient(135deg, #002D3F 50%, transparent 50%);
      background-position: calc(100% - 20px) calc(50% - 2px), calc(100% - 14px) calc(50% - 2px);
      background-size: 6px 6px, 6px 6px;
      background-repeat: no-repeat;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
      max-width: 100%;
    }

    .csv-download-panel {
      margin: 0 0 20px 0;
    }

    .csv-download-trigger {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 180px;
    }

    .csv-download-modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(2, 45, 63, 0.42);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
      z-index: 10050;
    }

    .csv-download-modal-overlay.open {
      display: flex;
    }

    .csv-download-modal {
      width: min(100%, 520px);
      background: #ffffff;
      border-radius: 14px;
      box-shadow: 0 20px 48px rgba(2, 45, 63, 0.22);
      border: 1px solid #d8e2e8;
      padding: 20px;
      position: relative;
    }

    .csv-download-modal-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 16px;
    }

    .csv-download-modal-title {
      margin: 0;
      color: #002D3F;
      font-size: 1.1rem;
      font-weight: 600;
    }

    .csv-download-modal-close {
      border: none;
      background: transparent;
      color: #002D3F;
      font-size: 1.4rem;
      line-height: 1;
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 6px;
    }

    .csv-download-modal-close:hover {
      background: #eef4f7;
    }

    .csv-download-modal-actions {
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 12px;
    }

    .js-plotly-plot .parcoords text {
      font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif !important;
      font-weight: 700 !important;
      text-shadow: none !important;
      stroke: none !important;
      paint-order: normal !important;
    }
  `;
  document.head.appendChild(style);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderOverviewTableHtml(tableData, fallbackMessage) {
  if (!tableData || !Array.isArray(tableData.columns) || !Array.isArray(tableData.rows) || !tableData.columns.length) {
    return `<p class="csv-muted-note">${fallbackMessage}</p>`;
  }

  const headerHtml = tableData.columns
    .map(column => `<th>${escapeHtml(column)}</th>`)
    .join('');

  const bodyHtml = tableData.rows
    .map(row => `
      <tr>
        ${row.map(cell => `<td>${escapeHtml(cell)}</td>`).join('')}
      </tr>
    `)
    .join('');

  return `
    <h2>${escapeHtml(tableData.title || 'Table')}</h2>
    <table cellpadding="5" cellspacing="0">
      <thead>
        <tr>${headerHtml}</tr>
      </thead>
      <tbody>
        ${bodyHtml}
      </tbody>
    </table>
  `;
}

function compareTableValues(left, right) {
  const leftNumber = Number(left);
  const rightNumber = Number(right);
  const leftIsNumber = !Number.isNaN(leftNumber) && left !== '';
  const rightIsNumber = !Number.isNaN(rightNumber) && right !== '';

  if (leftIsNumber && rightIsNumber) {
    return leftNumber - rightNumber;
  }

  return String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: 'base' });
}

function renderInteractiveResultsTable(containerId, headers, rows, emptyMessage) {
  const container = document.getElementById(containerId);
  if (!container) {
    return;
  }

  const state = {
    page: 1,
    pageSize: 15,
    sortStack: []
  };

  function getSortedRows() {
    const sortedRows = rows.slice();
    if (!state.sortStack.length) {
      return sortedRows;
    }

    sortedRows.sort((left, right) => {
      for (const sortRule of state.sortStack) {
        const leftValue = left[sortRule.column];
        const rightValue = right[sortRule.column];
        const leftEmpty = leftValue == null || leftValue === '';
        const rightEmpty = rightValue == null || rightValue === '';

        if (leftEmpty || rightEmpty) {
          if (leftEmpty && rightEmpty) {
            continue;
          }
          return leftEmpty ? 1 : -1;
        }

        const comparison = compareTableValues(leftValue, rightValue);
        if (comparison !== 0) {
          return sortRule.direction === 'asc' ? comparison : -comparison;
        }
      }
      return 0;
    });

    return sortedRows;
  }

  function render() {
    if (!rows.length) {
      container.innerHTML = `<p class="csv-muted-note">${emptyMessage}</p>`;
      return;
    }

    const sortedRows = getSortedRows();
    const totalPages = Math.max(1, Math.ceil(sortedRows.length / state.pageSize));
    if (state.page > totalPages) {
      state.page = totalPages;
    }

    const startIndex = (state.page - 1) * state.pageSize;
    const pageRows = sortedRows.slice(startIndex, startIndex + state.pageSize);

    const headerHtml = headers.map(header => {
      let indicator = ' ';
      const sortRule = state.sortStack.find(rule => rule.column === header);
      if (sortRule) {
        indicator = sortRule.direction === 'asc' ? '^' : 'v';
      }

      return `<th class="sortable-header" data-column="${escapeHtml(header)}" data-sort-indicator="${indicator}">${escapeHtml(header)}</th>`;
    }).join('');

    const bodyHtml = pageRows.map(row => `
      <tr>
        ${headers.map(header => `<td>${escapeHtml(row[header])}</td>`).join('')}
      </tr>
    `).join('');

    container.innerHTML = `
      <div class="csv-table-wrapper">
        <div class="csv-table-scroll">
          <table class="csv-results-table">
            <thead>
              <tr>${headerHtml}</tr>
            </thead>
            <tbody>${bodyHtml}</tbody>
          </table>
        </div>
        <div class="csv-table-pagination">
          <div class="csv-table-status">
            Showing ${startIndex + 1} to ${Math.min(startIndex + state.pageSize, sortedRows.length)} of ${sortedRows.length} results
          </div>
          <div>
            <button type="button" data-action="prev" ${state.page === 1 ? 'disabled' : ''}>Previous</button>
            <button type="button" data-action="next" ${state.page === totalPages ? 'disabled' : ''}>Next</button>
          </div>
        </div>
      </div>
    `;

    container.querySelectorAll('th.sortable-header').forEach(headerElement => {
      headerElement.addEventListener('click', () => {
        const column = headerElement.dataset.column;
        const existingIndex = state.sortStack.findIndex(rule => rule.column === column);

        if (existingIndex === 0) {
          state.sortStack[0].direction = state.sortStack[0].direction === 'asc' ? 'desc' : 'asc';
        } else if (existingIndex > 0) {
          const existingRule = state.sortStack.splice(existingIndex, 1)[0];
          state.sortStack.unshift(existingRule);
        } else {
          state.sortStack.unshift({ column, direction: 'asc' });
        }

        if (column === 'ACCURACY' && !state.sortStack.some(rule => rule.column === 'RUNTIME')) {
          state.sortStack.push({ column: 'RUNTIME', direction: 'asc' });
        } else if (column === 'MTR' && !state.sortStack.some(rule => rule.column === 'RUNTIME')) {
          state.sortStack.push({ column: 'RUNTIME', direction: 'asc' });
        } else if (column === 'RUNTIME' && !state.sortStack.some(rule => rule.column === 'ACCURACY')) {
          const fallbackMetricColumn = headers.includes('ACCURACY') ? 'ACCURACY' : (headers.includes('MTR') ? 'MTR' : null);
          if (fallbackMetricColumn) {
            state.sortStack.push({ column: fallbackMetricColumn, direction: 'desc' });
          }
        } else if (column !== 'ACCURACY' && column !== 'RUNTIME' && column !== 'MTR') {
          state.sortStack = state.sortStack.slice(0, 1);
        }

        if (state.sortStack.length > 2) {
          state.sortStack = state.sortStack.slice(0, 2);
        }

        state.page = 1;
        render();
      });
    });

    container.querySelector('[data-action="prev"]')?.addEventListener('click', () => {
      if (state.page > 1) {
        state.page -= 1;
        render();
      }
    });

    container.querySelector('[data-action="next"]')?.addEventListener('click', () => {
      if (state.page < totalPages) {
        state.page += 1;
        render();
      }
    });
  }

  render();
}

function buildParallelPlotRows(rows, headers, metricKey = 'ACCURACY') {
  const candidateHeaders = headers.filter(header => header !== 'Status');
  return rows
    .filter(row => candidateHeaders.every(header => row[header] !== '' && row[header] != null))
    .filter(row => typeof row[metricKey] === 'number' && typeof row.RUNTIME === 'number');
}

function renderParallelPlot(containerId, rows, headers, options = {}) {
  const metricKey = options.metricKey || 'ACCURACY';
  const metricColorValue = options.metricColorValue || 'accuracy';
  const metricLabel = options.metricLabel || 'Accuracy (max)';
  const plotRows = buildParallelPlotRows(rows, headers, metricKey);
  if (!plotRows.length || typeof Plotly === 'undefined') {
    const container = document.getElementById(containerId);
    if (container) {
      container.innerHTML = '<p class="csv-muted-note">Parallel plot data is not available for this experiment.</p>';
    }
    return;
  }

  const dimensions = headers
    .filter(header => header !== 'Status')
    .map(header => ({
      label: header,
      values: plotRows.map(row => Number(row[header]))
    }))
    .filter(dimension => dimension.values.every(value => !Number.isNaN(value)));

  const selector = document.getElementById('parallel-color-mode');

  function buildLineSettings(colorMode) {
    if (colorMode === metricColorValue) {
      return {
        color: plotRows.map(row => row[metricKey]),
        colorscale: [[0, 'rgba(220, 53, 69, 0.42)'], [1, 'rgba(40, 167, 69, 0.42)']],
        showscale: true,
        colorbar: { title: metricLabel }
      };
    }

    if (colorMode === 'runtime') {
      return {
        color: plotRows.map(row => row.RUNTIME),
        colorscale: [[0, 'rgba(40, 167, 69, 0.42)'], [1, 'rgba(220, 53, 69, 0.42)']],
        showscale: true,
        colorbar: { title: 'Runtime (min)' }
      };
    }

    return {
      color: 'rgba(1, 94, 128, 0.32)',
      showscale: false
    };
  }

  function renderPlot(colorMode) {
    Plotly.newPlot(
      containerId,
      [{
        type: 'parcoords',
        dimensions,
        labelfont: {
          size: 18,
          color: '#002D3F',
          family: 'Inter, sans-serif'
        },
        tickfont: {
          size: 15,
          color: '#002D3F',
          family: 'Inter, sans-serif'
        },
        line: buildLineSettings(colorMode)
      }],
      applyCommonPlotLayout({
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        margin: { l: 60, r: 60, t: 110, b: 40 }
      }),
      { responsive: true }
    );
  }

  renderPlot(selector?.value || 'none');

  if (selector && !selector.dataset.bound) {
    selector.dataset.bound = 'true';
    selector.addEventListener('change', () => {
      renderPlot(selector.value);
    });
  }
}

async function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

async function withBusyButton(button, loadingText, action) {
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = loadingText;
  try {
    await action();
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

async function fetchCsvFilesForZipWithResolver(detectors, fileResolver, datasets = Array.from(REAL_DATASETS), zipBasePath = '') {
  const requests = [];

  detectors.forEach(detector => {
    datasets.forEach(dataset => {
      const path = fileResolver(detector, dataset);
      requests.push(
        fetch(path)
          .then(response => response.ok ? response.text() : null)
          .then(content => content ? {
            relativePath: `${zipBasePath}${path.replace(/^data\/all_benchmark_results\//, '')}`,
            content
          } : null)
          .catch(() => null)
      );
    });
  });

  const files = await Promise.all(requests);
  return files.filter(Boolean);
}

async function downloadZipOfCsvs(detectors, zipName, fileResolver, datasets, zipBasePath = '') {
  if (typeof JSZip === 'undefined') {
    throw new Error('JSZip is not available.');
  }

  const files = await fetchCsvFilesForZipWithResolver(detectors, fileResolver, datasets, zipBasePath);
  const zip = new JSZip();

  files.forEach(file => {
    zip.file(file.relativePath, file.content);
  });

  const blob = await zip.generateAsync({ type: 'blob' });
  await downloadBlob(blob, zipName);
}

async function downloadZipFromConfigs(zipName, configs) {
  if (typeof JSZip === 'undefined') {
    throw new Error('JSZip is not available.');
  }

  const zip = new JSZip();

  for (const config of configs) {
    const files = await fetchCsvFilesForZipWithResolver(
      config.detectors,
      config.fileResolver,
      config.datasets,
      config.zipBasePath || ''
    );
    files.forEach(file => {
      zip.file(file.relativePath, file.content);
    });
  }

  const blob = await zip.generateAsync({ type: 'blob' });
  await downloadBlob(blob, zipName);
}

function buildAccuracyRuntimeCsvPath(detector, dataset) {
  return `data/all_benchmark_results/${detector}/${dataset}/${detector}_${dataset}_ACC_RT.csv`;
}

function buildReqLabelsCsvPath(detector, dataset) {
  return `data/all_benchmark_results/${detector}/${dataset}/${detector}_${dataset}_ACC_RT_REQL.csv`;
}

function buildMtrRuntimeCsvPath(detector, dataset) {
  return `data/all_benchmark_results/${detector}/${dataset}/${detector}_${dataset}_MTR_RT.csv`;
}

function buildOverviewDataPath(detector, dataset, metricCode) {
  return `data/overview_data/${detector}/${dataset}/${detector}_${dataset}_${metricCode}_overview.json`;
}

function buildDownloadPanelHtml(detector, dataset, metricLabel) {
  return `
    <div class="csv-download-panel">
      <div class="csv-download-group">
        <button type="button" class="csv-download-button csv-download-trigger" id="download-open-modal">Download Results</button>
      </div>
      <div class="csv-download-modal-overlay" id="csv-download-modal-overlay" aria-hidden="true">
        <div class="csv-download-modal" role="dialog" aria-modal="true" aria-labelledby="csv-download-modal-title">
          <div class="csv-download-modal-header">
            <h2 class="csv-download-modal-title" id="csv-download-modal-title">Download Options</h2>
            <button type="button" class="csv-download-modal-close" id="csv-download-modal-close" aria-label="Close download options">x</button>
          </div>
          <div class="csv-download-modal-actions">
            <button type="button" class="csv-download-button" id="download-all-csvs">All Benchmark Results</button>
            <button type="button" class="csv-download-button" id="download-detector-csvs">All ${detector} Results</button>
            <button type="button" class="csv-download-button" id="download-single-csv">${detector}_${dataset}_${metricLabel} Results</button>
          </div>
        </div>
      </div>
    </div>
  `;
}

function bindDownloadButtons({ detector, dataset, singlePathBuilder }) {
  const overlay = document.getElementById('csv-download-modal-overlay');
  const openButton = document.getElementById('download-open-modal');
  const closeButton = document.getElementById('csv-download-modal-close');

  function closeDownloadModal() {
    overlay?.classList.remove('open');
    overlay?.setAttribute('aria-hidden', 'true');
  }

  function openDownloadModal() {
    overlay?.classList.add('open');
    overlay?.setAttribute('aria-hidden', 'false');
  }

  openButton?.addEventListener('click', openDownloadModal);
  closeButton?.addEventListener('click', closeDownloadModal);
  overlay?.addEventListener('click', event => {
    if (event.target === overlay) {
      closeDownloadModal();
    }
  });

  document.getElementById('download-single-csv')?.addEventListener('click', async event => {
    closeDownloadModal();
    await withBusyButton(event.currentTarget, 'Preparing download...', async () => {
      const response = await fetch(singlePathBuilder(detector, dataset));
      if (!response.ok) {
        throw new Error('Failed to load the selected CSV file.');
      }
      const blob = await response.blob();
      await downloadBlob(blob, singlePathBuilder(detector, dataset).split('/').pop());
    }).catch(error => alert(error.message));
  });

  document.getElementById('download-detector-csvs')?.addEventListener('click', async event => {
    closeDownloadModal();
    await withBusyButton(event.currentTarget, 'Creating ZIP...', async () => {
      await downloadZipFromConfigs(`${detector.toLowerCase()}_results.zip`, [
        {
          detectors: [detector],
          datasets: Array.from(REAL_DATASETS),
          fileResolver: buildAccuracyRuntimeCsvPath
        },
        {
          detectors: [detector],
          datasets: Array.from(REAL_DATASETS),
          fileResolver: buildReqLabelsCsvPath
        },
        {
          detectors: [detector],
          datasets: Array.from(SYNTHETIC_DATASETS),
          fileResolver: buildMtrRuntimeCsvPath
        }
      ]);
    }).catch(error => alert(error.message));
  });

  document.getElementById('download-all-csvs')?.addEventListener('click', async event => {
    closeDownloadModal();
    await withBusyButton(event.currentTarget, 'Creating ZIP...', async () => {
      await downloadZipFromConfigs('all_benchmark_results.zip', [
        {
          detectors: DETECTOR_OPTIONS,
          datasets: Array.from(REAL_DATASETS),
          fileResolver: buildAccuracyRuntimeCsvPath,
          zipBasePath: 'all_benchmark_results/'
        },
        {
          detectors: DETECTOR_OPTIONS,
          datasets: Array.from(REAL_DATASETS),
          fileResolver: buildReqLabelsCsvPath,
          zipBasePath: 'all_benchmark_results/'
        },
        {
          detectors: DETECTOR_OPTIONS,
          datasets: Array.from(SYNTHETIC_DATASETS),
          fileResolver: buildMtrRuntimeCsvPath,
          zipBasePath: 'all_benchmark_results/'
        }
      ]);
    }).catch(error => alert(error.message));
  });
}

function computeParetoFrontMulti(rows, dimensions) {
  function dominates(candidate, target) {
    let strictlyBetter = false;

    for (const dimension of dimensions) {
      const candidateValue = candidate[dimension.key];
      const targetValue = target[dimension.key];

      if (typeof candidateValue !== 'number' || typeof targetValue !== 'number') {
        return false;
      }

      if (dimension.goal === 'max') {
        if (candidateValue < targetValue) {
          return false;
        }
        if (candidateValue > targetValue) {
          strictlyBetter = true;
        }
      } else {
        if (candidateValue > targetValue) {
          return false;
        }
        if (candidateValue < targetValue) {
          strictlyBetter = true;
        }
      }
    }

    return strictlyBetter;
  }

  return rows.filter((row, rowIndex) =>
    !rows.some((candidate, candidateIndex) =>
      candidateIndex !== rowIndex && dominates(candidate, row)
    )
  );
}

function buildParetoScatterTrace(detector, points, options = {}) {
  return {
    x: points.map(point => point[options.xKey]),
    y: points.map(point => point[options.yKey]),
    mode: 'markers',
    type: 'scatter',
    name: detector,
    marker: {
      size: options.markerSize || 8,
      color: DETECTOR_COLOR_SCHEME[detector] || '#000000'
    },
    text: points.map(point => buildParetoPointDetailText(point, options.headers || [], detector)),
    hovertemplate: '%{text}<extra></extra>'
  };
}

function buildParetoScatter3dTrace(detector, points, headers, keys) {
  return {
    x: points.map(point => point[keys.xKey]),
    y: points.map(point => point[keys.yKey]),
    z: points.map(point => point[keys.zKey]),
    mode: 'markers',
    type: 'scatter3d',
    name: detector,
    marker: {
      size: 6,
      color: DETECTOR_COLOR_SCHEME[detector] || '#000000'
    },
    text: points.map(point => buildParetoPointDetailText(point, headers, detector)),
    hovertemplate: '%{text}<extra></extra>'
  };
}

function createParetoSectionHtml(title, description, plotId, plotHeight = 650) {
  return `
    <div class="section-header results">${title}</div>
    <div class="section-content">
      <p class="csv-muted-note">${description}</p>
      <div id="${plotId}" class="csv-plot-block" style="width:100%; height:${plotHeight}px;"></div>
    </div>
  `;
}

async function loadParetoCsvDataByDetector(dataset, metrics) {
  const pathBuilder = metrics === 'ACCURACY-RUNTIME'
    ? buildAccuracyRuntimeCsvPath
    : metrics === 'ACCURACY-RUNTIME-REQLABELS'
      ? buildReqLabelsCsvPath
      : buildMtrRuntimeCsvPath;

  const requests = DETECTOR_OPTIONS.map(detector => {
    const primaryFetch = fetch(pathBuilder(detector, dataset))
      .then(response => response.ok ? response.text() : null);

    return primaryFetch
      .then(csvText => {
        if (!csvText) {
          return null;
        }

        const parsed = parseCsvText(csvText);

        if (!parsed.rows.length) {
          return null;
        }

        return {
          detector,
          rows: parsed.rows,
          headers: parsed.headers
        };
      })
      .catch(() => null);
  });

  const results = await Promise.all(requests);
  return results.filter(Boolean);
}

function renderParetoAccuracyRuntimeView(dataset, detectorData) {
  ensureCsvResultStyles();
  const content = document.getElementById('content');
  const traces = [];
  const allAccuracies = [];
  const allRuntimes = [];

  detectorData.forEach(({ detector, rows, headers }) => {
    const completedRows = rows.filter(row => String(row.Status || '').toLowerCase() === 'completed');
    const paretoRows = computeParetoFront(completedRows.length ? completedRows : rows);
    if (!paretoRows.length) {
      return;
    }

    paretoRows.forEach(row => {
      allAccuracies.push(row.ACCURACY);
      allRuntimes.push(row.RUNTIME);
    });

    traces.push(buildParetoScatterTrace(detector, paretoRows, {
      xKey: 'ACCURACY',
      yKey: 'RUNTIME',
      headers
    }));
  });

  content.innerHTML = createParetoSectionHtml(
    'Pareto Fronts',
    'Pareto-optimal points are computed separately for each detector on the selected dataset, then combined for plotting.',
    'pareto-plot',
    680
  );

  if (!traces.length) {
    document.getElementById('pareto-plot').innerHTML = '<p class="csv-muted-note">No Pareto-optimal points are available for this selection.</p>';
    content.style.visibility = 'visible';
    content.classList.add('visible');
    return;
  }

  if (typeof Plotly !== 'undefined') {
    Plotly.newPlot('pareto-plot', traces, applyCommonPlotLayout({
      title: { text: dataset, x: 0.5, xanchor: 'center' },
      xaxis: {
        title: { text: 'ACCURACY' },
        range: computeAxisRange(allAccuracies, { floor: 0, ceiling: 1 }) || undefined
      },
      yaxis: {
        title: { text: 'RUNTIME (seconds)' },
        range: computeAxisRange(allRuntimes, { floor: 0 }) || undefined
      },
      hovermode: 'closest',
      showlegend: true,
      paper_bgcolor: 'white',
      plot_bgcolor: 'white'
    }), { responsive: true });
  }

  content.style.visibility = 'visible';
  content.classList.add('visible');
}

function renderParetoReqLabelsView(dataset, detectorData) {
  ensureCsvResultStyles();
  const content = document.getElementById('content');
  const traces3d = [];
  const traces2d = [];
  const combinedParetoRows = [];
  const allAccuracies = [];
  const allRuntimes = [];
  const allReqLabels = [];

  detectorData.forEach(({ detector, rows, headers }) => {
    const completedRows = rows.filter(row =>
      String(row.Status || '').toLowerCase() === 'completed' &&
      typeof row.ACCURACY === 'number' &&
      typeof row.RUNTIME === 'number' &&
      typeof row.REQLABELS === 'number'
    );

    const paretoRows = computeParetoFrontMulti(completedRows, [
      { key: 'ACCURACY', goal: 'max' },
      { key: 'RUNTIME', goal: 'min' },
      { key: 'REQLABELS', goal: 'min' }
    ]);

    if (!paretoRows.length) {
      return;
    }

    paretoRows.forEach(row => {
      allAccuracies.push(row.ACCURACY);
      allRuntimes.push(row.RUNTIME);
      allReqLabels.push(row.REQLABELS);
      combinedParetoRows.push({
        ...row,
        detector,
        headers
      });
    });

    traces3d.push(buildParetoScatter3dTrace(detector, paretoRows, headers, {
      xKey: 'RUNTIME',
      yKey: 'ACCURACY',
      zKey: 'REQLABELS'
    }));

    traces2d.push({
      ...buildParetoScatterTrace(detector, paretoRows, {
        xKey: 'ACCURACY',
        yKey: 'REQLABELS',
        headers
      }),
      yaxis: 'y'
    });
  });

  content.innerHTML = `
    ${createParetoSectionHtml(
      '3D visualization of Pareto points per DD',
      'This section shows the 3D Pareto front visualization with Runtime, Accuracy, and Required Labels as the three dimensions. Each point represents a Pareto-optimal configuration for a drift detection method.',
      'pareto-plot-3d',
      720
    )}
    ${createParetoSectionHtml(
      '2D visualization of Pareto points per DD',
      'This section provides a 2D visualization of the Pareto front data with dual y-axes. The left axis shows Required Labels, the right axis shows Runtime, and dotted lines indicate the runtime values linked to the same Pareto points.',
      'pareto-plot-2d',
      640
    )}
  `;

  if (!traces3d.length || !traces2d.length) {
    document.getElementById('pareto-plot-3d').innerHTML = '<p class="csv-muted-note">No Pareto-optimal points are available for this selection.</p>';
    document.getElementById('pareto-plot-2d').innerHTML = '<p class="csv-muted-note">No Pareto-optimal points are available for this selection.</p>';
    content.style.visibility = 'visible';
    content.classList.add('visible');
    return;
  }

  if (typeof Plotly !== 'undefined') {
    const accuracyRange = computeAxisRange(allAccuracies, { floor: 0, ceiling: 1 }) || undefined;
    const reqLabelsRange = computeAxisRange(allReqLabels, { floor: 0, ceiling: 1 }) || [0, 1];
    const runtimeRange = computeAxisRange(allRuntimes, { floor: 0 }) || [0, 1];
    const anchorX = accuracyRange ? accuracyRange[1] : Math.max(...allAccuracies);
    const global3dParetoRows = computeParetoFrontMulti(combinedParetoRows, [
      { key: 'ACCURACY', goal: 'max' },
      { key: 'RUNTIME', goal: 'min' },
      { key: 'REQLABELS', goal: 'min' }
    ]);
    const globalRuntimeAxisRange = computeAxisRange(
      global3dParetoRows.map(row => row.RUNTIME),
      { floor: 0 }
    ) || runtimeRange;

    function mapReqLabelsToRuntimeAxis(value) {
      const [reqMin, reqMax] = reqLabelsRange;
      const [runtimeMin, runtimeMax] = globalRuntimeAxisRange;

      if (reqMax === reqMin) {
        return runtimeMin;
      }

      const normalized = (value - reqMin) / (reqMax - reqMin);
      return runtimeMin + normalized * (runtimeMax - runtimeMin);
    }

    global3dParetoRows.forEach(row => {
      traces2d.push({
        x: [anchorX, row.ACCURACY],
        y: [row.RUNTIME, mapReqLabelsToRuntimeAxis(row.REQLABELS)],
        mode: 'lines',
        type: 'scatter',
        line: {
          color: DETECTOR_COLOR_SCHEME[row.detector] || '#000000',
          width: 1,
          dash: 'dot'
        },
        hoverinfo: 'skip',
        showlegend: false,
        yaxis: 'y2'
      });
    });

    Plotly.newPlot('pareto-plot-3d', traces3d, applyCommonPlotLayout({
      title: { text: dataset, x: 0.5, xanchor: 'center' },
      scene: {
        xaxis: { title: { text: 'RUNTIME (seconds)' }, range: runtimeRange || undefined },
        yaxis: { title: { text: 'ACCURACY' }, range: accuracyRange || undefined },
        zaxis: { title: { text: 'REQLABELS' }, range: reqLabelsRange || undefined }
      },
      margin: { l: 80, r: 180, b: 80, t: 100 }
    }), { responsive: true });

    Plotly.newPlot('pareto-plot-2d', traces2d, applyCommonPlotLayout({
      title: { text: dataset, x: 0.5, xanchor: 'center' },
      xaxis: {
        title: { text: 'ACCURACY' },
        range: accuracyRange || undefined
      },
      yaxis: {
        title: { text: 'REQLABELS' },
        range: reqLabelsRange || undefined,
        rangemode: 'tozero'
      },
      yaxis2: {
        title: { text: 'RUNTIME (seconds)' },
        side: 'right',
        overlaying: 'y',
        color: '#d62728',
        range: globalRuntimeAxisRange || undefined,
        rangemode: 'tozero',
        showgrid: false
      },
      margin: { l: 80, r: 260, b: 80, t: 100 },
      legend: {
        x: 1.08,
        y: 1,
        xanchor: 'left',
        yanchor: 'top'
      },
      paper_bgcolor: 'white',
      plot_bgcolor: 'white'
    }), { responsive: true });
  }

  content.style.visibility = 'visible';
  content.classList.add('visible');
}

function renderParetoMtrRuntimeView(dataset, detectorData) {
  ensureCsvResultStyles();
  const content = document.getElementById('content');
  const traces = [];
  const allMtr = [];
  const allRuntimes = [];

  detectorData.forEach(({ detector, rows, headers }) => {
    const completedRows = rows.filter(row => String(row.Status || '').toLowerCase() === 'completed');
    const paretoRows = computeMetricRuntimeParetoFront(completedRows.length ? completedRows : rows, 'MTR');
    if (!paretoRows.length) {
      return;
    }

    paretoRows.forEach(row => {
      allMtr.push(row.MTR);
      allRuntimes.push(row.RUNTIME);
    });

    traces.push(buildParetoScatterTrace(detector, paretoRows, {
      xKey: 'MTR',
      yKey: 'RUNTIME',
      headers
    }));
  });

  content.innerHTML = createParetoSectionHtml(
    'Pareto Fronts',
    'Pareto-optimal points are computed separately for each detector on the selected synthetic dataset, then combined for plotting.',
    'pareto-plot',
    680
  );

  if (!traces.length) {
    document.getElementById('pareto-plot').innerHTML = '<p class="csv-muted-note">No Pareto-optimal points are available for this selection.</p>';
    content.style.visibility = 'visible';
    content.classList.add('visible');
    return;
  }

  if (typeof Plotly !== 'undefined') {
    Plotly.newPlot('pareto-plot', traces, applyCommonPlotLayout({
      title: { text: dataset, x: 0.5, xanchor: 'center' },
      xaxis: {
        title: { text: 'MTR' },
        range: computeAxisRange(allMtr, { floor: 0 }) || undefined
      },
      yaxis: {
        title: { text: 'RUNTIME (seconds)' },
        range: computeAxisRange(allRuntimes, { floor: 0 }) || undefined
      },
      hovermode: 'closest',
      showlegend: true,
      paper_bgcolor: 'white',
      plot_bgcolor: 'white'
    }), { responsive: true });
  }

  content.style.visibility = 'visible';
  content.classList.add('visible');
}

function loadCsvParetoExperiment({ dataset, metrics, button }) {
  const content = document.getElementById('content');

  if (typeof startLoadingAnimation === 'function') {
    startLoadingAnimation(button);
    window.activeButton = button;
  }

  content.style.visibility = 'hidden';
  content.classList.remove('visible');

  loadParetoCsvDataByDetector(dataset, metrics)
    .then(detectorData => {
      if (!detectorData.length) {
        throw new Error('No CSV files were found for the selected Pareto view.');
      }

      if (metrics === 'ACCURACY-RUNTIME') {
        renderParetoAccuracyRuntimeView(dataset, detectorData);
        return;
      }

      if (metrics === 'ACCURACY-RUNTIME-REQLABELS') {
        renderParetoReqLabelsView(dataset, detectorData);
        return;
      }

      if (metrics === SYNTHETIC_METRIC_VALUE) {
        renderParetoMtrRuntimeView(dataset, detectorData);
        return;
      }

      throw new Error('Unsupported Pareto metric selection.');
    })
    .catch(error => {
      content.innerHTML = `
        <div class="alert alert-danger">
          <h4>Error loading Pareto content</h4>
          <p>${error.message}</p>
        </div>
      `;
      content.style.visibility = 'visible';
      content.classList.add('visible');
    })
    .finally(() => {
      if (typeof stopLoadingAnimation === 'function' && button) {
        stopLoadingAnimation(button);
      }
    });
}

function renderReqLabelsScatterPlot(rows, headers) {
  const selector = document.getElementById('scatter-plot-type');
  const containerId = 'csv-accuracy-runtime-plot';
  if (!selector || typeof Plotly === 'undefined') {
    return;
  }

  function renderPlot(plotType) {
    const completedRows = rows.filter(row =>
      typeof row.ACCURACY === 'number' &&
      typeof row.RUNTIME === 'number' &&
      typeof row.REQLABELS === 'number'
    );

    if (!completedRows.length) {
      return;
    }

    if (plotType === 'reqlabels_accuracy_runtime_3d') {
      const paretoRows = computeParetoFrontMulti(completedRows, [
        { key: 'ACCURACY', goal: 'max' },
        { key: 'RUNTIME', goal: 'min' },
        { key: 'REQLABELS', goal: 'min' }
      ]);

      Plotly.newPlot(
        containerId,
        [
          {
            x: completedRows.map(row => row.ACCURACY),
            y: completedRows.map(row => row.RUNTIME),
            z: completedRows.map(row => row.REQLABELS),
            mode: 'markers',
            type: 'scatter3d',
            name: 'Completed Runs',
            marker: { color: '#015E80', size: 4, opacity: 0.7 },
            text: completedRows.map(row => buildPointDetailText(row, headers)),
            hovertemplate: '%{text}<extra></extra>'
          },
          {
            x: paretoRows.map(row => row.ACCURACY),
            y: paretoRows.map(row => row.RUNTIME),
            z: paretoRows.map(row => row.REQLABELS),
            mode: 'markers',
            type: 'scatter3d',
            name: 'Pareto-optimal Runs',
            marker: { color: '#dc3545', size: 5, opacity: 0.9 },
            text: paretoRows.map(row => buildPointDetailText(row, headers)),
            hovertemplate: '%{text}<extra></extra>'
          }
        ],
        applyCommonPlotLayout({
          title: 'REQLABELS vs ACCURACY vs RUNTIME',
          scene: {
            xaxis: { title: { text: 'ACCURACY (max)' }, range: computeAxisRange(completedRows.map(row => row.ACCURACY), { floor: 0, ceiling: 1 }) || undefined },
            yaxis: { title: { text: 'RUNTIME (min)' }, range: computeAxisRange(completedRows.map(row => row.RUNTIME), { floor: 0 }) || undefined },
            zaxis: { title: { text: 'REQLABELS (min)' }, range: computeAxisRange(completedRows.map(row => row.REQLABELS), { floor: 0, ceiling: 1 }) || undefined }
          },
          legend: { orientation: 'h', y: 1.05 }
        }),
        { responsive: true }
      );
      return;
    }

    const isAccuracyPlot = plotType === 'reqlabels_accuracy';
    const xKey = isAccuracyPlot ? 'ACCURACY' : 'RUNTIME';
    const xLabel = isAccuracyPlot ? 'ACCURACY (max)' : 'RUNTIME (min)';
    const title = isAccuracyPlot ? 'ACCURACY vs REQLABELS' : 'RUNTIME vs REQLABELS';
    const paretoRows = computeParetoFrontMulti(completedRows, [
      { key: 'ACCURACY', goal: 'max' },
      { key: 'RUNTIME', goal: 'min' },
      { key: 'REQLABELS', goal: 'min' }
    ]);

    Plotly.newPlot(
      containerId,
      [
        {
          x: completedRows.map(row => row[xKey]),
          y: completedRows.map(row => row.REQLABELS),
          mode: 'markers',
          type: 'scatter',
          name: 'Completed Runs',
          marker: { color: '#015E80', size: 10, opacity: 0.75 },
          text: completedRows.map(row => buildPointDetailText(row, headers)),
          hovertemplate: '%{text}<extra></extra>'
        },
        {
          x: paretoRows.map(row => row[xKey]),
          y: paretoRows.map(row => row.REQLABELS),
          mode: 'markers',
          type: 'scatter',
          name: 'Pareto-optimal Runs',
          marker: { color: '#dc3545', size: 11, opacity: 0.9 },
          text: paretoRows.map(row => buildPointDetailText(row, headers)),
          hovertemplate: '%{text}<extra></extra>'
        }
      ],
      applyCommonPlotLayout({
        title,
        xaxis: {
          title: { text: xLabel },
          range: computeAxisRange(completedRows.map(row => row[xKey]), { floor: xKey === 'RUNTIME' ? 0 : 0, ceiling: xKey === 'ACCURACY' ? 1 : null }) || undefined
        },
        yaxis: {
          title: { text: 'REQLABELS (min)' },
          range: computeAxisRange(completedRows.map(row => row.REQLABELS), { floor: 0, ceiling: 1 }) || undefined
        },
        legend: { orientation: 'h', y: 1.1 },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        hovermode: 'closest'
      }),
      { responsive: true }
    );
  }

  renderPlot(selector.value);
  if (!selector.dataset.bound) {
    selector.dataset.bound = 'true';
    selector.addEventListener('change', () => renderPlot(selector.value));
  }
}

function renderCsvAccuracyRuntimeExperiment({ detector, dataset, model, rows: rawRows, headers, metadata }) {
  ensureCsvResultStyles();
  const content = document.getElementById('content');
  const rows = rawRows.filter(row => String(row.Status || '').toLowerCase() !== 'abandoned');
  const completedRows = rows.filter(row => String(row.Status || '').toLowerCase() === 'completed');
  const failedRows = rows.filter(row => String(row.Status || '').toLowerCase() === 'failed');
  const plotRows = completedRows.length ? completedRows : rows.filter(row => typeof row.ACCURACY === 'number' && typeof row.RUNTIME === 'number');
  const paretoRows = computeParetoFront(plotRows);
  const overviewData = metadata || {};

  const bestAccuracy = plotRows.length
    ? Math.max(...plotRows.map(row => row.ACCURACY).filter(value => typeof value === 'number'))
    : null;
  const minRuntime = plotRows.length
    ? Math.min(...plotRows.map(row => row.RUNTIME).filter(value => typeof value === 'number'))
    : null;

  content.innerHTML = `
    <div class="section-header results">Results</div>
    <div class="experiment-subcaption" style="font-size: 0.7em; font-weight: normal; margin: 8px 0 16px 0; opacity: 0.9; letter-spacing: 0.3px; color: #666;">
      ${detector} + ${dataset} + ${model} + ACCURACY-RUNTIME
    </div>
    <div class="section-content">
      ${buildDownloadPanelHtml(detector, dataset, 'Accuracy-Runtime', Array.from(REAL_DATASETS))}
      <div class="csv-summary-grid">
        <div class="csv-summary-card">
          <div class="csv-summary-label">Total Runs</div>
          <div class="csv-summary-value">${rows.length}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Completed Runs</div>
          <div class="csv-summary-value">${completedRows.length}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Failed Runs</div>
          <div class="csv-summary-value">${failedRows.length}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Best Accuracy</div>
          <div class="csv-summary-value">${formatMetricValue(bestAccuracy, 3)}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Best Runtime</div>
          <div class="csv-summary-value">${formatMetricValue(minRuntime, 0)}</div>
        </div>
      </div>
      <h2>Scatter Plot</h2>
      <div id="csv-accuracy-runtime-plot" class="csv-plot-block" style="width:100%; height:520px;"></div>
      <h2>Parallel Plot</h2>
      <div class="csv-parallel-controls">
        <label for="parallel-color-mode">Color lines by</label>
        <select id="parallel-color-mode">
          <option value="none">Default</option>
          <option value="accuracy">Accuracy (max)</option>
          <option value="runtime">Runtime (min)</option>
        </select>
      </div>
      <div id="csv-parallel-plot" class="csv-plot-block" style="width:100%; height:860px;"></div>
      <h2>Trials</h2>
      <div id="csv-all-runs-table"></div>
    </div>
    <div class="section-header overview">Overview</div>
    <div class="section-content">
      ${renderOverviewTableHtml(overviewData.overview, 'Experiment overview is not available.')}
      ${renderOverviewTableHtml(overviewData.parameters, 'Experiment parameters are not available.')}
    </div>
  `;

  bindDownloadButtons({
    detector,
    dataset,
    singlePathBuilder: buildAccuracyRuntimeCsvPath
  });

  if (typeof Plotly !== 'undefined') {
    const runtimeRange = computeAxisRange(plotRows.map(row => row.RUNTIME), { floor: 0 });
    const accuracyRange = computeAxisRange(plotRows.map(row => row.ACCURACY), { floor: 0, ceiling: 1 });

    const allTrialsTrace = {
      x: plotRows.map(row => row.RUNTIME),
      y: plotRows.map(row => row.ACCURACY),
      text: plotRows.map(row => buildPointDetailText(row, headers)),
      mode: 'markers',
      type: 'scatter',
      name: 'Completed Runs',
      marker: {
        color: '#015E80',
        size: 10,
        opacity: 0.75
      },
      hovertemplate: '%{text}<extra></extra>'
    };

    const paretoTrace = {
      x: paretoRows.map(row => row.RUNTIME),
      y: paretoRows.map(row => row.ACCURACY),
      mode: 'markers',
      type: 'scatter',
      name: 'Pareto-optimal Runs',
      marker: {
        color: '#dc3545',
        size: 11,
        symbol: 'circle'
      },
      hovertemplate: 'Runtime: %{x}<br>Accuracy: %{y}<extra>Pareto Front</extra>'
    };

    Plotly.newPlot(
      'csv-accuracy-runtime-plot',
      [allTrialsTrace, paretoTrace],
      applyCommonPlotLayout({
        title: `${detector} on ${dataset}`,
        xaxis: { title: { text: 'Runtime (min)' }, range: runtimeRange || undefined },
        yaxis: { title: { text: 'Accuracy' }, range: accuracyRange || undefined },
        hovermode: 'closest',
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        legend: { orientation: 'h', y: 1.1 }
      }),
      { responsive: true }
    );
  }

  renderParallelPlot('csv-parallel-plot', completedRows, headers);
  renderInteractiveResultsTable('csv-all-runs-table', headers, rows, 'No runs are available.');

  content.style.visibility = 'visible';
  content.classList.add('visible');
}

function renderCsvReqLabelsExperiment({ detector, dataset, model, rows: rawRows, headers, metadata }) {
  ensureCsvResultStyles();
  const content = document.getElementById('content');
  const rows = rawRows.filter(row => String(row.Status || '').toLowerCase() !== 'abandoned');
  const completedRows = rows.filter(row => String(row.Status || '').toLowerCase() === 'completed');
  const failedRows = rows.filter(row => String(row.Status || '').toLowerCase() === 'failed');
  const plotRows = completedRows.filter(row =>
    typeof row.ACCURACY === 'number' &&
    typeof row.RUNTIME === 'number' &&
    typeof row.REQLABELS === 'number'
  );
  const overviewData = metadata || {};

  const bestAccuracy = completedRows.length
    ? Math.max(...completedRows.map(row => row.ACCURACY).filter(value => typeof value === 'number'))
    : null;
  const minRuntime = completedRows.length
    ? Math.min(...completedRows.map(row => row.RUNTIME).filter(value => typeof value === 'number'))
    : null;

  content.innerHTML = `
    <div class="section-header results">Results</div>
    <div class="experiment-subcaption" style="font-size: 0.7em; font-weight: normal; margin: 8px 0 16px 0; opacity: 0.9; letter-spacing: 0.3px; color: #666;">
      ${detector} + ${dataset} + ${model} + ACCURACY-RUNTIME-REQLABELS
    </div>
    <div class="section-content">
      ${buildDownloadPanelHtml(detector, dataset, 'Accuracy-Runtime-ReqLabels', Array.from(REAL_DATASETS))}
      <div class="csv-summary-grid">
        <div class="csv-summary-card">
          <div class="csv-summary-label">Total Runs</div>
          <div class="csv-summary-value">${rows.length}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Completed Runs</div>
          <div class="csv-summary-value">${completedRows.length}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Failed Runs</div>
          <div class="csv-summary-value">${failedRows.length}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Best Accuracy</div>
          <div class="csv-summary-value">${formatMetricValue(bestAccuracy, 3)}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Best Runtime</div>
          <div class="csv-summary-value">${formatMetricValue(minRuntime, 0)}</div>
        </div>
      </div>
      <h2>Scatter Plot</h2>
      <div class="csv-parallel-controls">
        <label for="scatter-plot-type">Plot Type</label>
        <select id="scatter-plot-type">
          <option value="reqlabels_accuracy">REQLABELS-ACCURACY</option>
          <option value="reqlabels_runtime">REQLABELS-RUNTIME</option>
          <option value="reqlabels_accuracy_runtime_3d">REQLABELS-ACCURACY-RUNTIME (3D)</option>
        </select>
      </div>
      <div id="csv-accuracy-runtime-plot" class="csv-plot-block" style="width:100%; height:620px;"></div>
      <h2>Parallel Plot</h2>
      <div class="csv-parallel-controls">
        <label for="parallel-color-mode">Color lines by</label>
        <select id="parallel-color-mode">
          <option value="none">Default</option>
          <option value="accuracy">Accuracy (max)</option>
          <option value="runtime">Runtime (min)</option>
        </select>
      </div>
      <div id="csv-parallel-plot" class="csv-plot-block" style="width:100%; height:860px;"></div>
      <h2>Trials</h2>
      <div id="csv-all-runs-table"></div>
    </div>
    <div class="section-header overview">Overview</div>
    <div class="section-content">
      ${renderOverviewTableHtml(overviewData.overview, 'Experiment overview is not available.')}
      ${renderOverviewTableHtml(overviewData.parameters, 'Experiment parameters are not available.')}
    </div>
  `;

  bindDownloadButtons({
    detector,
    dataset,
    singlePathBuilder: buildReqLabelsCsvPath
  });

  renderReqLabelsScatterPlot(plotRows, headers);
  renderParallelPlot('csv-parallel-plot', completedRows, headers);
  renderInteractiveResultsTable('csv-all-runs-table', headers, rows, 'No runs are available.');

  content.style.visibility = 'visible';
  content.classList.add('visible');
}

function renderCsvMtrRuntimeExperiment({ detector, dataset, model, rows: rawRows, headers, metadata }) {
  ensureCsvResultStyles();
  const content = document.getElementById('content');
  const rows = rawRows.filter(row => String(row.Status || '').toLowerCase() !== 'abandoned');
  const completedRows = rows.filter(row => String(row.Status || '').toLowerCase() === 'completed');
  const failedRows = rows.filter(row => String(row.Status || '').toLowerCase() === 'failed');
  const plotRows = completedRows.length ? completedRows : rows.filter(row => typeof row.MTR === 'number' && typeof row.RUNTIME === 'number');
  const paretoRows = computeMetricRuntimeParetoFront(plotRows, 'MTR');
  const overviewData = metadata || {};

  const bestMtr = plotRows.length
    ? Math.max(...plotRows.map(row => row.MTR).filter(value => typeof value === 'number'))
    : null;
  const minRuntime = plotRows.length
    ? Math.min(...plotRows.map(row => row.RUNTIME).filter(value => typeof value === 'number'))
    : null;

  content.innerHTML = `
    <div class="section-header results">Results</div>
    <div class="experiment-subcaption" style="font-size: 0.7em; font-weight: normal; margin: 8px 0 16px 0; opacity: 0.9; letter-spacing: 0.3px; color: #666;">
      ${detector} + ${dataset} + ${model} + MEANTIMERATIO-RUNTIME
    </div>
    <div class="section-content">
      ${buildDownloadPanelHtml(detector, dataset, 'MTR-Runtime', Array.from(SYNTHETIC_DATASETS))}
      <div class="csv-summary-grid">
        <div class="csv-summary-card">
          <div class="csv-summary-label">Total Runs</div>
          <div class="csv-summary-value">${rows.length}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Completed Runs</div>
          <div class="csv-summary-value">${completedRows.length}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Failed Runs</div>
          <div class="csv-summary-value">${failedRows.length}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Best MTR</div>
          <div class="csv-summary-value">${formatMetricValue(bestMtr, 3)}</div>
        </div>
        <div class="csv-summary-card">
          <div class="csv-summary-label">Best Runtime</div>
          <div class="csv-summary-value">${formatMetricValue(minRuntime, 0)}</div>
        </div>
      </div>
      <h2>Scatter Plot</h2>
      <div id="csv-accuracy-runtime-plot" class="csv-plot-block" style="width:100%; height:520px;"></div>
      <h2>Parallel Plot</h2>
      <div class="csv-parallel-controls">
        <label for="parallel-color-mode">Color lines by</label>
        <select id="parallel-color-mode">
          <option value="none">Default</option>
          <option value="mtr">MTR (max)</option>
          <option value="runtime">Runtime (min)</option>
        </select>
      </div>
      <div id="csv-parallel-plot" class="csv-plot-block" style="width:100%; height:860px;"></div>
      <h2>Trials</h2>
      <div id="csv-all-runs-table"></div>
    </div>
    <div class="section-header overview">Overview</div>
    <div class="section-content">
      ${renderOverviewTableHtml(overviewData.overview, 'Experiment overview is not available.')}
      ${renderOverviewTableHtml(overviewData.parameters, 'Experiment parameters are not available.')}
    </div>
  `;

  bindDownloadButtons({
    detector,
    dataset,
    singlePathBuilder: buildMtrRuntimeCsvPath
  });

  if (typeof Plotly !== 'undefined') {
    const runtimeRange = computeAxisRange(plotRows.map(row => row.RUNTIME), { floor: 0 });
    const mtrRange = computeAxisRange(plotRows.map(row => row.MTR), { floor: 0 });

    const allTrialsTrace = {
      x: plotRows.map(row => row.RUNTIME),
      y: plotRows.map(row => row.MTR),
      text: plotRows.map(row => buildPointDetailText(row, headers)),
      mode: 'markers',
      type: 'scatter',
      name: 'Completed Runs',
      marker: {
        color: '#015E80',
        size: 10,
        opacity: 0.75
      },
      hovertemplate: '%{text}<extra></extra>'
    };

    const paretoTrace = {
      x: paretoRows.map(row => row.RUNTIME),
      y: paretoRows.map(row => row.MTR),
      mode: 'markers',
      type: 'scatter',
      name: 'Pareto-optimal Runs',
      marker: {
        color: '#dc3545',
        size: 11,
        symbol: 'circle'
      },
      hovertemplate: 'Runtime: %{x}<br>MTR: %{y}<extra>Pareto Front</extra>'
    };

    Plotly.newPlot(
      'csv-accuracy-runtime-plot',
      [allTrialsTrace, paretoTrace],
      applyCommonPlotLayout({
        title: `${detector} on ${dataset}`,
        xaxis: { title: { text: 'Runtime (min)' }, range: runtimeRange || undefined },
        yaxis: { title: { text: 'MTR (max)' }, range: mtrRange || undefined },
        hovermode: 'closest',
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        legend: { orientation: 'h', y: 1.1 }
      }),
      { responsive: true }
    );
  }

  renderParallelPlot('csv-parallel-plot', completedRows, headers, {
    metricKey: 'MTR',
    metricColorValue: 'mtr',
    metricLabel: 'MTR (max)'
  });
  renderInteractiveResultsTable('csv-all-runs-table', headers, rows, 'No runs are available.');

  content.style.visibility = 'visible';
  content.classList.add('visible');
}

function loadCsvAccuracyRuntimeExperiment({ detector, dataset, model, button }) {
  const accRtPath = buildAccuracyRuntimeCsvPath(detector, dataset);
  const overviewDataPath = buildOverviewDataPath(detector, dataset, 'ACC_RT');
  const content = document.getElementById('content');

  if (typeof startLoadingAnimation === 'function') {
    startLoadingAnimation(button);
    window.activeButton = button;
  }

  content.style.visibility = 'hidden';
  content.classList.remove('visible');

  Promise.all([
    fetch(accRtPath).then(response => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return response.text();
    }),
    fetch(overviewDataPath)
      .then(response => response.ok ? response.json() : null)
      .catch(() => null)
  ])
    .then(([accRtCsvText, overviewData]) => {
      const parsed = parseCsvText(accRtCsvText);

      renderCsvAccuracyRuntimeExperiment({
        detector,
        dataset,
        model,
        rows: parsed.rows,
        headers: parsed.headers,
        metadata: overviewData
      });
    })
    .catch(error => {
      content.innerHTML = `
        <div class="alert alert-danger">
          <h4>Error loading CSV content</h4>
          <p>Failed to load: ${accRtPath}</p>
          <p>${error.message}</p>
        </div>
      `;
      content.style.visibility = 'visible';
      content.classList.add('visible');
    })
    .finally(() => {
      if (typeof stopLoadingAnimation === 'function' && button) {
        stopLoadingAnimation(button);
      }
    });
}

function loadCsvReqLabelsExperiment({ detector, dataset, model, button }) {
  const path = buildReqLabelsCsvPath(detector, dataset);
  const overviewDataPath = buildOverviewDataPath(detector, dataset, 'ACC_RT_REQL');
  const content = document.getElementById('content');

  if (typeof startLoadingAnimation === 'function') {
    startLoadingAnimation(button);
    window.activeButton = button;
  }

  content.style.visibility = 'hidden';
  content.classList.remove('visible');

  Promise.all([
    fetch(path).then(response => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return response.text();
    }),
    fetch(overviewDataPath)
      .then(response => response.ok ? response.json() : null)
      .catch(() => null)
  ])
    .then(([csvText, overviewData]) => {
      const parsed = parseCsvText(csvText);
      renderCsvReqLabelsExperiment({
        detector,
        dataset,
        model,
        rows: parsed.rows,
        headers: parsed.headers,
        metadata: overviewData
      });
    })
    .catch(error => {
      content.innerHTML = `
        <div class="alert alert-danger">
          <h4>Error loading CSV content</h4>
          <p>Failed to load: ${path}</p>
          <p>${error.message}</p>
        </div>
      `;
      content.style.visibility = 'visible';
      content.classList.add('visible');
    })
    .finally(() => {
      if (typeof stopLoadingAnimation === 'function' && button) {
        stopLoadingAnimation(button);
      }
    });
}

function loadCsvMtrRuntimeExperiment({ detector, dataset, model, button }) {
  const path = buildMtrRuntimeCsvPath(detector, dataset);
  const overviewDataPath = buildOverviewDataPath(detector, dataset, 'MTR_RT');
  const content = document.getElementById('content');

  if (typeof startLoadingAnimation === 'function') {
    startLoadingAnimation(button);
    window.activeButton = button;
  }

  content.style.visibility = 'hidden';
  content.classList.remove('visible');

  Promise.all([
    fetch(path).then(response => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return response.text();
    }),
    fetch(overviewDataPath)
      .then(response => response.ok ? response.json() : null)
      .catch(() => null)
  ])
    .then(([csvText, overviewData]) => {
      const parsed = parseCsvText(csvText);
      renderCsvMtrRuntimeExperiment({
        detector,
        dataset,
        model,
        rows: parsed.rows,
        headers: parsed.headers,
        metadata: overviewData
      });
    })
    .catch(error => {
      content.innerHTML = `
        <div class="alert alert-danger">
          <h4>Error loading CSV content</h4>
          <p>Failed to load: ${path}</p>
          <p>${error.message}</p>
        </div>
      `;
      content.style.visibility = 'visible';
      content.classList.add('visible');
    })
    .finally(() => {
      if (typeof stopLoadingAnimation === 'function' && button) {
        stopLoadingAnimation(button);
      }
    });
}

$(document).ready(function () {
  $('.header-section').click(function(){
    $('.header-section').removeClass("active");
    $(this).addClass("active");
  });

  // Function to fade out landing title with animation
  function hideLandingTitle() {
    const $landingTitle = $('#landing-title');
    $landingTitle.addClass('fade-out');
    setTimeout(() => {
      $landingTitle.addClass('hidden');
    }, 600); // Wait for fade animation to complete
  }

  // Function to show landing title with fade in
  function showLandingTitle() {
    const $landingTitle = $('#landing-title');
    $landingTitle.removeClass('hidden').removeClass('fade-out');
  }

  // Handle click on "Home"
  $('#home').click(function () {
    // Show landing title and hide all content sections
    showLandingTitle();
    $('#single-dds-content').addClass('hidden');
    $('#pareto-fronts-content').addClass('hidden');
    $('#ensemble-estimation-content').addClass('hidden');
    $('#content').empty();
  });

  // Handle click on "Single DDs"
  $('#single-dds').click(function () {
    // Hide landing title with fade and show the Single DDs content
    hideLandingTitle();
    $('#pareto-fronts-content').addClass('hidden');
    $('#ensemble-estimation-content').addClass('hidden');
    $('#single-dds-content').removeClass('hidden');
    $('#content').empty();
  });

  $('#show-button_sdd').click(function (event) {
    event.preventDefault();
    event.stopImmediatePropagation();

    const dataset = $('custom-dropdown[name="dataset"] input[type="hidden"]').val();
    const model = $('custom-dropdown[name="model"] input[type="hidden"]').val();
    const detector = $('custom-dropdown[name="dd"] input[type="hidden"]').val();
    const metrics = $('custom-dropdown[name="metrics"] input[type="hidden"]').val();

    if (isCsvBackedAccuracyRuntimeSelection(dataset, metrics)) {
      loadCsvAccuracyRuntimeExperiment({
        detector,
        dataset,
        model,
        button: this
      });
      return;
    }

    if (REAL_DATASETS.has(dataset) && metrics === 'ACCURACY-RUNTIME-REQLABELS') {
      loadCsvReqLabelsExperiment({
        detector,
        dataset,
        model,
        button: this
      });
      return;
    }

    if (SYNTHETIC_DATASETS.has(dataset) && metrics === SYNTHETIC_METRIC_VALUE) {
      loadCsvMtrRuntimeExperiment({
        detector,
        dataset,
        model,
        button: this
      });
      return;
    }

    const content = document.getElementById('content');
    content.innerHTML = `
      <div class="alert alert-warning">
        <h4>Unsupported selection</h4>
        <p>The selected detector, dataset, and metric combination is not available in the CSV-backed single-detector view.</p>
      </div>
    `;
    content.style.visibility = 'visible';
    content.classList.add('visible');
  });

  $('#show-button_par').click(function (event) {
    event.preventDefault();
    event.stopImmediatePropagation();

    const dataset = $('custom-dropdown[name="dataset_par"] input[type="hidden"]').val();
    const metrics = $('custom-dropdown[name="metrics_par"] input[type="hidden"]').val();
    const model = $('custom-dropdown[name="model_par"] input[type="hidden"]').val();

    if (model !== 'HoeffdingTreeClassifier') {
      const content = document.getElementById('content');
      content.innerHTML = `
        <div class="alert alert-warning">
          <h4>Unsupported selection</h4>
          <p>The selected model is not available in the CSV-backed Pareto view.</p>
        </div>
      `;
      content.style.visibility = 'visible';
      content.classList.add('visible');
      return;
    }

    loadCsvParetoExperiment({
      dataset,
      metrics,
      button: this
    });
  });

  // Handle click on "Pareto Fronts"
  $('#pareto-fronts').click(function () {
    hideLandingTitle();
    $('#pareto-fronts-content').removeClass('hidden');
    $('#single-dds-content').addClass('hidden');
    $('#ensemble-estimation-content').addClass('hidden');
    $('#content').empty();
  });

  // Handle click on "Ensemble Estimation"
  $('#ensemble-estimation').click(function () {
    hideLandingTitle();
    $('#ensemble-estimation-content').removeClass('hidden');
    $('#single-dds-content').addClass('hidden');
    $('#pareto-fronts-content').addClass('hidden');
    $('#content').empty();
  });

  $('#box1').click(function() {
    $('#ensemble-estimation-content .ensemble-tab-btn').removeClass('active');
    $('#box1').addClass('active');
    $('#menu1').toggleClass('hidden');
    $('#content').empty();
  });

  $('#box2').click(function(e) {
    e.preventDefault();
    $('#ensemble-estimation-content .ensemble-tab-btn').removeClass('active');
    $('#box2').addClass('active');
    $('#menu1').addClass('hidden');
    
    // Show enhanced loading state
    const $content = $('#content');
    $content.html(`
      <div class="ensemble-loading">
        <div class="loading-spinner"></div>
        <h3>Loading Single Variate DDs Visualization</h3>
        <p>Preparing multiple features visualization...</p>
      </div>
    `);
    $content.css('opacity', '1');
    
    // Create an enhanced iframe container
    const iframeContainer = document.createElement('div');
    iframeContainer.className = 'ensemble-iframe-container';
    
    // Create the iframe with enhanced styling
    const iframe = document.createElement('iframe');
    iframe.src = `data/ensemble_estimation/sv_mf/sv_mf.html`;
    iframe.className = 'ensemble-iframe';
    iframe.style.width = '100%';
    iframe.style.height = '2000px';
    iframe.style.minHeight = '1500px';
    iframe.style.border = 'none';
    iframe.style.borderRadius = '12px';
    iframe.style.boxShadow = '0 8px 32px rgba(2, 45, 63, 0.15)';
    iframe.style.background = 'white';
    iframe.style.transition = 'all 0.3s ease';
    iframe.style.opacity = '0';
    iframe.style.transform = 'translateY(20px)';
    
    // Add iframe to container
    iframeContainer.appendChild(iframe);
    
    // Add loading animation and smooth transition
    iframe.onload = function() {
      // Fade out loading, fade in iframe
      setTimeout(() => {
        $('.ensemble-loading').fadeOut(300, () => {
          iframe.style.opacity = '1';
          iframe.style.transform = 'translateY(0)';
        });
      }, 500);
    };
    
    // Clear the content and append the iframe container
    setTimeout(() => {
      $content.append(iframeContainer);
    }, 100);
    
    // Enhanced error handling
    iframe.onerror = function() {
      $content.html(`
        <div class="ensemble-error">
          <div class="error-icon">⚠️</div>
          <h3>Visualization Loading Error</h3>
          <p>Unable to load the Single Variate DDs visualization.</p>
          <p>Please try refreshing the page.</p>
          <button onclick="location.reload()" class="retry-button">Retry</button>
        </div>
      `);
    };
  });


  $('#show-button_ee_ensemble').click(function (e) {
    e.preventDefault();
    
    // Get the selected dataset
    const dataset = $('custom-dropdown[name="dataset_ee"] input[type="hidden"]').val();
    
    if (!dataset) {
      console.error('Please select a dataset');
      return;
    }
    
    // Show enhanced loading state
    const $content = $('#content');
    $content.html(`
      <div class="ensemble-loading">
        <div class="loading-spinner"></div>
        <h3>Loading Ensemble Visualization</h3>
        <p>Preparing ${dataset} dataset visualization...</p>
      </div>
    `);
    $content.css('opacity', '1');
    
    // Create an enhanced iframe container
    const iframeContainer = document.createElement('div');
    iframeContainer.className = 'ensemble-iframe-container';
    
    // Create the iframe with enhanced styling
    const iframe = document.createElement('iframe');
    iframe.src = `data/ensemble_estimation/ensemble_DDs/ensemble.html?dataset=${encodeURIComponent(dataset)}`;
    iframe.className = 'ensemble-iframe';
    iframe.style.width = '100%';
    iframe.style.height = '1200px';
    iframe.style.minHeight = '800px';
    iframe.style.border = 'none';
    iframe.style.borderRadius = '12px';
    iframe.style.boxShadow = '0 8px 32px rgba(2, 45, 63, 0.15)';
    iframe.style.background = 'white';
    iframe.style.transition = 'all 0.3s ease';
    iframe.style.opacity = '0';
    iframe.style.transform = 'translateY(20px)';
    
    // Add iframe to container
    iframeContainer.appendChild(iframe);
    
    // Add loading animation and smooth transition
    iframe.onload = function() {
      // Fade out loading, fade in iframe
      setTimeout(() => {
        $('.ensemble-loading').fadeOut(300, () => {
          iframe.style.opacity = '1';
          iframe.style.transform = 'translateY(0)';
        });
      }, 500);
    };
    
    // Clear the content and append the iframe container
    setTimeout(() => {
      $content.append(iframeContainer);
    }, 100);
    
    // Enhanced error handling
    iframe.onerror = function() {
      $content.html(`
        <div class="ensemble-error">
          <div class="error-icon">⚠️</div>
          <h3>Visualization Loading Error</h3>
          <p>Unable to load the ensemble visualization for <strong>${dataset}</strong>.</p>
          <p>Please try selecting a different dataset or refresh the page.</p>
          <button onclick="location.reload()" class="retry-button">Retry</button>
        </div>
      `);
    };
  });

});

let contentTimeout = 100;
window.activeButton = null;

function validateDatasetMetricCompatibility(datasetValue, metricsValue) {
  // Define synthetic datasets
  const syntheticDatasets = ['SineClusters', 'WaveformDrift2'];
  
  // Define synthetic metrics
  const syntheticMetrics = ['MEANTIMERATIO-RUNTIME', 'RUNTIME-MTR'];
  
  // Check if dataset is synthetic
  const isSyntheticDataset = syntheticDatasets.includes(datasetValue);
  
  // Check if metric is synthetic
  const isSyntheticMetric = syntheticMetrics.includes(metricsValue);
  
  // Validation logic: both should be synthetic or both should be real
  if (isSyntheticDataset && !isSyntheticMetric) {
    return {
      valid: false,
      message: `Incompatible Selection\n\nYou selected a Synthetic dataset "${datasetValue}" with a Real dataset metric.\n\nPlease select:\n• A Real dataset (like Electricity, GasSensor, etc.) with Real metrics\n• OR a Synthetic dataset with "Mean Time Ratio (max)|Runtime(min)" metric`
    };
  }
  
  if (!isSyntheticDataset && isSyntheticMetric) {
    return {
      valid: false,
      message: `Incompatible Selection\n\nYou selected a Real dataset "${datasetValue}" with a Synthetic dataset metric.\n\nPlease select:\n• A Real dataset with Real metrics (like "Accuracy(max)|Runtime(min)")\n• OR a Synthetic dataset (SineClusters or WaveformDrift2) with "Mean Time Ratio (max)|Runtime(min)" metric`
    };
  }
  
  return { valid: true };
}

function showValidationError(message) {
  // Create a gentle modal-style message
  const errorDiv = document.createElement('div');
  errorDiv.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.3);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10000;
    font-family: Arial, sans-serif;
  `;
  
  errorDiv.innerHTML = `
    <div style="
      background: white;
      padding: 25px 30px;
      border-radius: 8px;
      box-shadow: 0 2px 15px rgba(0, 0, 0, 0.2);
      max-width: 480px;
      text-align: left;
      border-left: 4px solid #f39c12;
    ">
      <h3 style="color: #555; margin-bottom: 15px; font-weight: normal;">Selection Notice</h3>
      <p style="white-space: pre-line; line-height: 1.5; color: #666; margin-bottom: 20px; font-size: 14px;">${message}</p>
      <div style="text-align: center;">
        <button onclick="this.parentElement.parentElement.parentElement.remove()" style="
          background: #3498db;
          color: white;
          border: none;
          padding: 8px 20px;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
        ">OK</button>
      </div>
    </div>
  `;
  
  document.body.appendChild(errorDiv);
  
  // Auto-remove after 15 seconds
  setTimeout(() => {
    if (errorDiv.parentElement) {
      errorDiv.remove();
    }
  }, 15000);
}

function handleShowButton(button, paramNames, basePath, extension = '.html', separator = '_') {
  console.log('handleShowButton called with:', { button, paramNames, basePath, extension });
  
  // Validate button parameter
  if (!button || !button.classList) {
    console.error('Invalid button parameter:', button);
    return;
  }
  
  // Start loading animation
  startLoadingAnimation(button);
  window.activeButton = button;
  
  // Extract values from custom-dropdown components using their name attributes
  const values = [];
  let datasetValue, metricsValue;
  
  if (basePath === 'paretos/') {
    // For pareto section: dataset, classifier, metrics
    datasetValue = $('custom-dropdown[name="dataset_par"] input[type="hidden"]').val();
    const classifierValue = $('custom-dropdown[name="model_par"] input[type="hidden"]').val();
    metricsValue = $('custom-dropdown[name="metrics_par"] input[type="hidden"]').val();
    
    values.push(datasetValue, classifierValue, metricsValue);
    console.log('Pareto values:', { dataset: datasetValue, classifier: classifierValue, metrics: metricsValue });
    
  } else if (basePath === 'exps/') {
    // For SDD section: detector, dataset, classifier, metrics
    const detectorValue = $('custom-dropdown[name="dd"] input[type="hidden"]').val();
    datasetValue = $('custom-dropdown[name="dataset"] input[type="hidden"]').val();
    const classifierValue = $('custom-dropdown[name="model"] input[type="hidden"]').val();
    metricsValue = $('custom-dropdown[name="metrics"] input[type="hidden"]').val();
    
    values.push(detectorValue, datasetValue, classifierValue, metricsValue);
    console.log('SDD values:', { detector: detectorValue, dataset: datasetValue, classifier: classifierValue, metrics: metricsValue });
    
  } else if (basePath === 'ensemble/') {
    // For ensemble section: detector, dataset, classifier
    const detectorValue = $('custom-dropdown[name="detector_ee"] input[type="hidden"]').val();
    datasetValue = $('custom-dropdown[name="dataset_ee"] input[type="hidden"]').val();
    const classifierValue = $('custom-dropdown[name="classifier_ee"] input[type="hidden"]').val();
    
    values.push(detectorValue, datasetValue, classifierValue);
    console.log('Ensemble values:', { detector: detectorValue, dataset: datasetValue, classifier: classifierValue });
    
  } else if (basePath === 'ensemble/single_predictions/') {
    // For single predictions: detector, dataset
    const detectorValue = $('custom-dropdown[name="detector_sv"] input[type="hidden"]').val();
    datasetValue = $('custom-dropdown[name="dataset_sv"] input[type="hidden"]').val();
    
    values.push(detectorValue, datasetValue);
    console.log('Single predictions values:', { detector: detectorValue, dataset: datasetValue });
  }
  
  // Validate dataset/metric compatibility for sections that have both
  if (datasetValue && metricsValue) {
    const validation = validateDatasetMetricCompatibility(datasetValue, metricsValue);
    if (!validation.valid) {
      console.log('Validation failed:', validation.message);
      stopLoadingAnimation(button);
      showValidationError(validation.message);
      return; // Stop execution if validation fails
    }
    console.log('✅ Dataset/metric compatibility validation passed');
  }
  
  console.log('Extracted values:', values, 'Expected params:', paramNames.length);
  
  // Validate we have enough values
  if (values.length < paramNames.length) {
    console.error('Not enough dropdown values found. Expected:', paramNames.length, 'Found:', values.length);
    stopLoadingAnimation(button);
    return;
  }

  // Check if REQLABELS is selected in any dropdown
  const useAccRunReqDir = values.some(value => value && value.includes('REQLABELS'));
  
  // Check if synthetic metrics are selected in any dropdown
  const syntheticMetrics = ['MEANTIMERATIO-RUNTIME', 'RUNTIME-MTR'];
  const useSynDir = values.some(value => value && syntheticMetrics.includes(value));
  
  // Use the appropriate directory based on the metrics and context
  let effectiveBasePath = basePath;
  if (useSynDir && basePath === 'exps/') {
    effectiveBasePath = 'exps_syn/';
  } else if (useAccRunReqDir && basePath !== 'paretos/') {
    effectiveBasePath = 'exps_acc_run_req/';
  }
  
  const path = `${effectiveBasePath}${values.join(separator)}${extension}`;
  console.log('Final path:', path);

  document.getElementById('content').style.visibility = 'hidden';
  document.getElementById('content').classList.remove('visible');

  setTimeout(() => {
    fetch(path)
      .then(response => {
        console.log('Fetch response:', response.status, response.statusText);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.text();
      })
      .then(html => {
        console.log('HTML loaded successfully, length:', html.length);
        
        // Create a temporary div to parse the HTML
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        
        // Check if this is a complete HTML document with <html> and <body> tags
        const bodyContent = tempDiv.querySelector('body');
        
        if (bodyContent) {
          // If it's a complete HTML document, extract the body content
          const newDiv = document.createElement('div');
          newDiv.innerHTML = bodyContent.innerHTML;
          handleLoadedContent.call(newDiv);
        } else {
          // If it's just a fragment, use it as is
          handleLoadedContent.call(tempDiv);
        }
      })
      .catch(error => {
        console.error('Error loading content:', error);
        
        // Stop loading animation on error
        if (window.activeButton) {
          stopLoadingAnimation(window.activeButton);
        }
        
        const content = document.getElementById('content');
        content.innerHTML = `
          <div class="alert alert-danger">
            <h4>Error loading content</h4>
            <p>Failed to load: ${path}</p>
            <p>${error.message}</p>
          </div>
        `;
        content.style.visibility = 'visible';
        content.classList.add('visible');
      });
  }, contentTimeout);
}

function startLoadingAnimation(button) {
  console.log('startLoadingAnimation called with:', button);
  
  if (!button) {
    console.error('startLoadingAnimation: button is null/undefined');
    return;
  }
  
  if (!button.classList) {
    console.error('startLoadingAnimation: button.classList is undefined', button);
    return;
  }
  
  // Safety fallback: stop animation after 10 seconds maximum
  setTimeout(() => {
    if (window.activeButton === button) {
      stopLoadingAnimation(button);
      console.log('Loading animation stopped due to safety timeout');
    }
  }, 10000);
  
  // Add loading class to button
  button.classList.add('loading');
  
  // Change button text to show loading state
  const textSpan = button.querySelector('.text');
  if (textSpan) {
    textSpan.setAttribute('data-original-text', textSpan.textContent);
    textSpan.innerHTML = '<span class="loading-spinner"></span>Loading...';
  }
  
  // Disable the button to prevent multiple clicks
  button.disabled = true;
  
  console.log('Loading animation started successfully');
}

function stopLoadingAnimation(button) {
  console.log('stopLoadingAnimation called with:', button);
  
  if (!button) {
    console.error('stopLoadingAnimation: button is null/undefined');
    return;
  }
  
  if (!button.classList) {
    console.error('stopLoadingAnimation: button.classList is undefined');
    return;
  }
  
  // Remove loading class from button
  button.classList.remove('loading');
  
  // Restore original button text
  const textSpan = button.querySelector('.text');
  if (textSpan && textSpan.hasAttribute('data-original-text')) {
    textSpan.textContent = textSpan.getAttribute('data-original-text');
    textSpan.removeAttribute('data-original-text');
  }
  
  // Re-enable the button
  button.disabled = false;
  
  console.log('Loading animation stopped successfully');
}

function handleLoadedContent() {
  const $loaded = $(this);
  // Delete &nbsp;
  $loaded.find('h1,h2').each(function () {
    const cleaned = $(this).html().replace(/^\s*&nbsp;+/g, '').trim();
    $(this).html(cleaned);
  });

  // Apply Bootstrap styles
  $loaded.find('table').addClass('table table-striped table-bordered table-sm');
  $loaded.find('button').addClass('btn btn-primary');

  // Extract JavaScript from loaded content
  const scripts = $loaded.find('script');
  const htmlContent = $loaded.html();
  
  // Substitute by loaded content first
  $('#content').html(htmlContent);

  // Execute scripts after a short delay to ensure DOM is ready
  setTimeout(() => {
    scripts.each(function() {
      const scriptContent = $(this).html();
      if (scriptContent && scriptContent.trim()) {
        try {
          // Check if this is a 3D plot script and handle it specially
          if (scriptContent.includes('create3DParetoPlot')) {
            console.log('Executing 3D Pareto plot script...');
            
            // Verify the container exists
            const container = document.getElementById('pareto-plot-container');
            if (!container) {
              console.error('pareto-plot-container not found in DOM');
              return;
            }
            
            // Verify Plotly is available
            if (typeof Plotly === 'undefined') {
              console.error('Plotly library not available');
              return;
            }
            
            // Execute the script content directly
            eval(scriptContent);
            
            // Call the function if it exists
            if (typeof create3DParetoPlot === 'function') {
              create3DParetoPlot();
              create2DParetoPlot();
              console.log('3D Pareto plot created successfully');
              
              // Stop loading animation after plots are rendered
              setTimeout(() => {
                if (window.activeButton) {
                  stopLoadingAnimation(window.activeButton);
                  console.log('Loading animation stopped after plot rendering');
                }
              }, 1500);
            } else {
              console.error('create3DParetoPlot function not found after script execution');
              setTimeout(() => {
                if (window.activeButton) {
                  stopLoadingAnimation(window.activeButton);
                }
              }, 500);
            }
          } else {
            // Execute other scripts normally
            const newScript = document.createElement('script');
            newScript.text = scriptContent;
            document.head.appendChild(newScript);
            document.head.removeChild(newScript);
            console.log('Executed script from loaded content');
            
            // Stop loading animation for non-plot content after a short delay
            setTimeout(() => {
              if (window.activeButton) {
                stopLoadingAnimation(window.activeButton);
                console.log('Loading animation stopped after script execution');
              }
            }, 300);
          }
        } catch (error) {
          console.error('Error executing script from loaded content:', error);
          setTimeout(() => {
            if (window.activeButton) {
              stopLoadingAnimation(window.activeButton);
            }
          }, 300);
        }
      }
    });
  }, 50);

  // "visible" content animation
  setTimeout(() => {
    $('#content').addClass('visible');
  }, 100);
}

// Initialize Show button event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM loaded, setting up Show button event listeners...');
  
  // Wait a bit for custom elements to be defined
  setTimeout(() => {
    setupShowButtonListeners();
  }, 500);
});

function setupShowButtonListeners() {
  console.log('Setting up Show button event listeners...');
  
  // Stream Drift Detection button
  const sddButton = document.getElementById('show-button_sdd');
  console.log('SDD button found:', !!sddButton);
  if (sddButton) {
    sddButton.addEventListener('click', function(event) {
      event.preventDefault();
      console.log('SDD button clicked, this:', this);
      handleShowButton(this, ['detector', 'dataset', 'classifier', 'metrics'], 'exps/', '.html', '_');
    });
  }
  
  // Pareto Fronts button  
  const paretoButton = document.getElementById('show-button_par');
  console.log('Pareto button found:', !!paretoButton);
  if (paretoButton) {
    paretoButton.addEventListener('click', function(event) {
      event.preventDefault();
      console.log('Pareto button clicked, this:', this);
      handleShowButton(this, ['dataset', 'classifier', 'metrics'], 'paretos/', '.html', '_');
    });
  }
  
  // Ensemble Evaluation - Ensemble button
  const ensembleButton = document.getElementById('show-button_ee_ensemble');
  console.log('Ensemble button found:', !!ensembleButton);
  if (ensembleButton) {
    ensembleButton.addEventListener('click', function(event) {
      event.preventDefault();
      console.log('Ensemble button clicked, this:', this);
      handleShowButton(this, ['detector', 'dataset', 'classifier'], 'ensemble/', '.html', '_');
    });
  }
  
  // Ensemble Evaluation - Single Predictions button
  const singleButton = document.getElementById('show-button_ee_sv');
  console.log('Single predictions button found:', !!singleButton);
  if (singleButton) {
    singleButton.addEventListener('click', function(event) {
      event.preventDefault();
      console.log('Single predictions button clicked, this:', this);
      handleShowButton(this, ['detector', 'dataset'], 'ensemble/single_predictions/', '.png', '_');
    });
  }
  
  console.log('Show button event listeners set up successfully');
}
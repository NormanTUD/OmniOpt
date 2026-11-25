<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Benchmark Concept Drift</title>
  <style>
    .no-fouc{visibility:hidden}
    .no-fouc #loading-overlay{visibility:visible}
    #loading-overlay{position:fixed;inset:0;z-index:99999;display:flex !important;align-items:center !important;justify-content:center !important;background:#fff;transition:opacity 0.4s ease-out}
    #loading-overlay.hidden{opacity:0;pointer-events:none}
    .loading-spinner{width:48px;height:48px;border:4px solid rgba(0,45,63,0.15);border-top-color:#015E80;border-radius:50%;animation:spin 0.8s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
  </style>
  <link rel="preconnect" href="https://rsms.me">
  <link rel="preload" href="https://rsms.me/inter/inter.css" as="style">
  <link rel="stylesheet" href="https://rsms.me/inter/inter.css">
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
  <link rel="stylesheet" href="expstyle.css">
  <link rel="stylesheet" href="styles.css">
  <link rel="stylesheet" href="components/dropdown/dropdown.css">
  <link rel="stylesheet" href="components/show_button/show_button.css">
  <script src="https://unpkg.com/gridjs/dist/gridjs.umd.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script src='https://cdn.jsdelivr.net/npm/plotly.js-dist@3.0.1/plotly.min.js'></script>
</head>
<body class="no-fouc">
  <div id="loading-overlay"><div class="loading-spinner"></div></div>
  <header>
    <div class="header-section" id="home">Home</div>
    <div class="header-section" id="single-dds">Single Detector Experiments</div>
    <div class="header-section" id="pareto-fronts">Multiple Detector Pareto Fronts</div>
    <div class="header-section" id="ensemble-estimation">Ensemble Estimation</div>
  </header>

  <!-- Landing Page Title (shown when no section is selected) -->
  <div id="landing-title" class="landing-title-container">
    <div class="main-title-container">
      <h1 class="main-title">Computational Performance of Unsupervised Concept Drift Detection: A Survey and Multiobjective Benchmark using Bayesian Optimization</h1>
      <p class="authors-subtitle">Elias Werner, Babak Sepehri Rad, Daniel Lukats, Norman Koch, Peter Winkler</p>
    </div>
  </div>

  <!-- Single DDs -->
  <div id="single-dds-content" class="hidden">
    <div class="container-fluid text-center">
      <div class="single-dd-control-panel">
        <div class="single-dd-field">
          <div class="single-dd-field-label">Detector:</div>
          <div class="single-dd-field-input">
            <custom-dropdown
              label="DD"
              name="dd"
              external-label
              options='[
                "BNDM", "CDBD", "CDLEEDS", "CSDDM", "D3", "DAWIDD", "DDAL",
                "EDFS", "HDDDM", "IBDD", "IKS", "NNDVI", "OCDD", "PCACD",
                "SlidShaps", "SPLL", "STUDD", "UCDD", "UDetect", "WindowKDE"
              ]'>
            </custom-dropdown>
          </div>
        </div>
        <div class="single-dd-field">
          <div class="single-dd-field-label">Dataset:</div>
          <div class="single-dd-field-input">
          <custom-dropdown
            label="Dataset"
            name="dataset"
            external-label
            options='[
              {"optgroup": "Real Datasets", "options": [
                "Electricity",
                "ForestCovertype", 
                "GasSensor",
                "NOAAWeather",
                "OutdoorObjects",
                "Ozone",
                "PokerHand",
                "RialtoBridgeTimelapse",
                "SensorStream"
              ]},
              {"optgroup": "Synthetic Datasets", "options": [
                "SineClusters",
                "WaveformDrift2"
              ]}
            ]'>
          </custom-dropdown>
          </div>
        </div>
        <div class="single-dd-field">
          <div class="single-dd-field-label">Metrics:</div>
          <div class="single-dd-field-input">
          <custom-dropdown
            label="Metrics"
            name="metrics"
            external-label
            options='[
              {
                "optgroup": "Metrics for Real Datasets",
                "options": [
                  {"label": "Accuracy(max)|Runtime(min)", "value": "ACCURACY-RUNTIME"},
                  {"label": "ReqLabels(min)|Accuracy(max)|Runtime(min)", "value": "ACCURACY-RUNTIME-REQLABELS"}
                ]
              },
              {
                "optgroup": "Metrics for Synthetic Datasets",
                "options": [
                  {"label": "Mean Time Ratio (max)|Runtime(min)", "value": "MEANTIMERATIO-RUNTIME"}
                ]
              }
            ]'
            default-value="ACCURACY-RUNTIME">
          </custom-dropdown>
          </div>
        </div>
        <div class="single-dd-field">
          <div class="single-dd-field-label">Model:</div>
          <div class="single-dd-field-input">
          <custom-dropdown
            label="Model"
            name="model"
            external-label
            value="HoeffdingTreeClassifier"
            options='[
              "HoeffdingTreeClassifier"
            ]'>
          </custom-dropdown>
        </div>
        </div>
        <div class="single-dd-show">
          <button class="show-button" id="show-button_sdd">
            <span class="text">Show</span>
          </button>
        </div>
      </div>
    </div>
    <div class="glow-line"></div>
  </div>

  <!-- Pareto Fronts -->
  <div id="pareto-fronts-content" class="hidden">
    <div class="container-fluid text-center">
      <div class="pareto-control-panel">
        <div class="pareto-field">
          <div class="pareto-field-label">Dataset:</div>
          <div class="pareto-field-input">
            <custom-dropdown
              label="Dataset"
              name="dataset_par"
              external-label
              options='[
                {"optgroup": "Real Datasets", "options": [
                  "Electricity",
                  "ForestCovertype",
                  "GasSensor",
                  "NOAAWeather",
                  "OutdoorObjects",
                  "Ozone",
                  "PokerHand",
                  "RialtoBridgeTimelapse",
                  "SensorStream"
                ]},
                {"optgroup": "Synthetic Datasets", "options": [
                  "SineClusters",
                  "WaveformDrift2"
                ]}
              ]'>
            </custom-dropdown>
          </div>
        </div>
        <div class="pareto-field">
          <div class="pareto-field-label">Metrics:</div>
          <div class="pareto-field-input">
            <custom-dropdown
              label="Metrics"
              name="metrics_par"
              external-label
              options='[
                {
                  "optgroup": "Metrics for Real Datasets",
                  "options": [
                    {"label": "Accuracy(max)|Runtime(min)", "value": "ACCURACY-RUNTIME"},
                    {"label": "ReqLabels(min)|Accuracy(max)|Runtime(min)", "value": "ACCURACY-RUNTIME-REQLABELS"}
                  ]
                },
                {
                  "optgroup": "Metrics for Synthetic Datasets",
                  "options": [
                    {"label": "Mean Time Ratio (max)|Runtime(min)", "value": "MEANTIMERATIO-RUNTIME"}
                  ]
                }
              ]'
              default-value="ACCURACY-RUNTIME">
            </custom-dropdown>
          </div>
        </div>
        <div class="pareto-field">
          <div class="pareto-field-label">Model:</div>
          <div class="pareto-field-input">
            <custom-dropdown
              label="Model"
              name="model_par"
              external-label
              value="HoeffdingTreeClassifier"
              options='[
                "HoeffdingTreeClassifier"
              ]'>
            </custom-dropdown>
          </div>
        </div>
        <div class="pareto-show">
          <button class="show-button" id="show-button_par">
            <span class="text">Show</span>
          </button>
        </div>
      </div>
    </div>
    <div class="glow-line"></div>
  </div>

  <!-- Ensemble Estimation -->
  <div id="ensemble-estimation-content" class="hidden">
    <div class="container-fluid text-center">
      <div class="row row-cols-2 justify-content-center">
        <div class="col-6">
          <div class="box ensemble-tab-btn" id="box1">Ensemble of DDs</div>
          <div class="menu hidden" id="menu1">
            <div class="ensemble-ee-control-panel">
              <div class="ensemble-ee-field">
                <div class="ensemble-ee-field-label">Dataset:</div>
                <div class="ensemble-ee-field-input">
                  <custom-dropdown
                    label="Dataset"
                    name="dataset_ee"
                    external-label
                    options='[
                      "Electricity",
                      "ForestCovertype",
                      "GasSensor",
                      "NOAAWeather",
                      "OutdoorObjects",
                      "Ozone",
                      "PokerHand",
                      "RialtoBridgeTimelapse",
                      "SensorStream"
                    ]'>
                  </custom-dropdown>
                </div>
              </div>
              <div class="ensemble-ee-field">
                <div class="ensemble-ee-field-label">Model:</div>
                <div class="ensemble-ee-field-input">
                  <custom-dropdown
                    label="Model"
                    name="model_ee"
                    external-label
                    value="HoeffdingTreeClassifier"
                    options='[
                      {"label": "HoeffdingTreeClassifier", "value": "HoeffdingTreeClassifier", "selected": true}
                    ]'>
                  </custom-dropdown>
                </div>
              </div>
              <div class="ensemble-ee-show">
                <button class="show-button" id="show-button_ee_ensemble"><span class="text">Show</span></button>
              </div>
            </div>
          </div>
        </div>
        <div class="col-6">
          <div class="box ensemble-tab-btn" id="box2">Single Variate DDs on Multiple Features</div>
        </div>
      </div>
    </div>
  </div>

  <div id="content"></div>

  <!-- Footer -->
  <footer class="site-footer">
    <div class="footer-content">
      <div class="footer-links">
        <a href="mailto:elias.werner@tu-dresden.de" class="footer-link">Contact</a>
        <a href="https://scads.ai/imprint/" target="_blank" class="footer-link">Imprint</a>
        <a href="https://scads.ai/privacy/" target="_blank" class="footer-link">Privacy</a>
        <a href="https://github.com/ScaDS/benchmark-unsupervised-concept-drift-detection" target="_blank" rel="noopener noreferrer" class="footer-link footer-github-link" aria-label="GitHub repository" title="GitHub repository">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>
        </a>
      </div>
    </div>
  </footer>

  <script>
    (function() {
      var revealed = false;
      function reveal() {
        if (revealed) return;
        revealed = true;
        document.body.classList.remove('no-fouc');
        var overlay = document.getElementById('loading-overlay');
        if (overlay) {
          overlay.classList.add('hidden');
          setTimeout(function() { overlay.remove(); }, 500);
        }
      }
      // Wait for first paint so the overlay is actually visible
      requestAnimationFrame(function() {
        setTimeout(function() {
          if (document.fonts && document.fonts.ready) {
            document.fonts.ready.then(reveal);
          } else {
            reveal();
          }
        }, 50);
      });
      setTimeout(reveal, 2000);
    })();
  </script>
  <script src="components/dropdown/dropdown.js"></script>
  <script src="components/show_button/show_button.js"></script>
  <script src="script.js"></script>
</body>
</html>

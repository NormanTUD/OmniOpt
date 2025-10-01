<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Interactive Website</title>
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
  <link rel="stylesheet" href="expstyle.css">
  <link rel="stylesheet" href="styles.css">
  <link rel="stylesheet" href="components/dropdown/dropdown.css">
  <link rel="stylesheet" href="components/scroll_button/scroll_button.css">
  <link rel="stylesheet" href="components/show_button/show_button.css">
  <link rel="preconnect" href="https://rsms.me/">
  <link rel="stylesheet" href="https://rsms.me/inter/inter.css">
  <script src="https://unpkg.com/gridjs/dist/gridjs.umd.js"></script>
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script src='https://cdn.jsdelivr.net/npm/plotly.js-dist@3.0.1/plotly.min.js'></script>
</head>
<body>
  <header>
    <div class="header-section" id="home">Home</div>
    <div class="header-section" id="single-dds">Single Detector Experiments</div>
    <div class="header-section" id="pareto-fronts">Multiple Detector Pareto Fronts</div>
    <div class="header-section" id="ensemble-estimation">Ensemble Estimation</div>
  </header>

  <!-- Landing Page Title (shown when no section is selected) -->
  <div id="landing-title" class="landing-title-container">
    <div class="main-title-container">
      <h1 class="main-title">Computational Performance of Semi- and Unsupervised Concept Drift Detection: A Survey and Multiobjective Benchmark using Bayesian Optimization</h1>
      <p class="authors-subtitle">Elias Werner, Babak Sepehri Rad, Daniel Lukats, Peter Winkler, Norman Koch</p>
    </div>
  </div>

  <!-- Single DDs -->
  <div id="single-dds-content" class="hidden">
    <div class="container-fluid text-center">
      <div class="row justify-content-center align-items-center" style="gap: 20px;">
        <div class="col-auto">
          <custom-dropdown
            label="Dataset"
            name="dataset"
            options='[
              {"optgroup": "Real", "options": [
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
              {"optgroup": "Synthetic", "options": [
                "SineClusters",
                "WaveformDrift2"
              ]}
            ]'>
          </custom-dropdown>
        </div>
        <div class="col-auto">
          <custom-dropdown
            label="Model"
            name="model"
            value="HoeffdingTreeClassifier"
            options='[
              "HoeffdingTreeClassifier"
            ]'>
          </custom-dropdown>
        </div>
        <div class="col-auto">
          <custom-dropdown
            label="Metrics"
            name="metrics"
            options='[
              {
                "optgroup": "Real Datasets",
                "options": [
                  {"label": "Accuracy(max)|Runtime(min)", "value": "ACCURACY-RUNTIME"},
                  {"label": "ReqLabels(min)|Accuracy(max)|Runtime(min)", "value": "ACCURACY-RUNTIME-REQLABELS"}
                ]
              },
              {
                "optgroup": "Synthetic Datasets",
                "options": [
                  {"label": "Mean Time Ratio (max)|Runtime(min)", "value": "MEANTIMERATIO-RUNTIME"}
                ]
              }
            ]'
            default-value="ACCURACY-RUNTIME">
          </custom-dropdown>
        </div>
        <div class="col-auto">
          <custom-dropdown
            label="DD"
            name="dd"
            options='[
              "BNDM", "CDBD", "CDLEEDS", "CSDDM", "D3", "DAWIDD", "DDAL",
              "EDFS", "HDDDM", "IBDD", "IKS", "NNDVI", "OCDD", "PCACD",
              "SlidShaps", "SPLL", "STUDD", "UCDD", "UDetect", "WindowKDE"
            ]'>
          </custom-dropdown>
        </div>
        <div class="col-auto">
          <button class="show-button" id="show-button_sdd">
            <span class="text" id="button-text">Show</span>
          </button>
        </div>

      </div>
    </div>
    <div class="glow-line"></div>
  </div>

  <!-- Pareto Fronts -->
  <div id="pareto-fronts-content" class="hidden">
    <div class="container-fluid text-center">
      <div class="row justify-content-center align-items-center" style="gap: 30px;">
        <div class="col-auto">
          <custom-dropdown
            label="Dataset"
            name="dataset_par"
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
        <div class="col-auto">
          <custom-dropdown
            label="Model"
            name="model_par"
            value="HoeffdingTreeClassifier"
            options='[
              "HoeffdingTreeClassifier"
            ]'>
          </custom-dropdown>
        </div>
        <div class="col-auto">
          <custom-dropdown
            label="Metrics"
            name="metrics_par"
            options='[
                  {"label": "Accuracy(max)|Runtime(min)", "value": "ACCURACY-RUNTIME"},
                  {"label": "ReqLabels(min)|Accuracy(max)|Runtime(min)", "value": "ACCURACY-RUNTIME-REQLABELS"}
            ]'
            default-value="ACCURACY-RUNTIME">
          </custom-dropdown>
        </div>
        <div class="col-auto">
          <button class="show-button" id="show-button_par">
            <span class="text" id="button-text">Show</span>
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
          <div class="box" id="box1">Ensemble of DDs</div>
          <div class="menu hidden" id="menu1">
            <div class="dropdown-row">
              <custom-dropdown
                label="Dataset"
                name="dataset_ee"
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
              <custom-dropdown
                label="Model"
                name="model_ee"
                value="HoeffdingTreeClassifier"
                options='[
                  {"label": "HoeffdingTreeClassifier", "value": "HoeffdingTreeClassifier", "selected": true}
                ]'>
              </custom-dropdown>
              <button class="show-button" id="show-button_ee_ensemble"><span class="text">Show</span></button>
            </div>
          </div>
        </div>
        <div class="col-6">
          <div class="box" id="box2">Single Variate DDs on Multiple Features</div>
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
      </div>
    </div>
  </footer>

  <script src="components/dropdown/dropdown.js"></script>
  <script src="components/show_button/show_button.js"></script>
  <script src="script.js"></script>
</body>
</html>

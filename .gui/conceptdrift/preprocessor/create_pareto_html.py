import os
import re
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import html
import re
import json

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Pareto Fronts Plot</title>
  <script src='https://cdn.jsdelivr.net/npm/plotly.js-dist@3.0.1/plotly.min.js'></script>
</head>
<body>
  <div id="plot" style="width:100%;height:100vh;"></div>

  <script>
    var paretoData = {pareto_data_json};

    var traces = Object.entries(paretoData).map(([methodName, methodData]) => {
  return {
    x: methodData.map(point => point.x),
    y: methodData.map(point => point.y),
    mode: 'markers',
    name: methodName, // Use the method key as the name
    type: 'scatter',
  };
})

    var layout = {
      title: 'Pareto Fronts of Different Methods',
      xaxis: { title: 'Accuracy' },
      yaxis: { title: 'Runtime' },
      hovermode: 'closest',
      showlegend: true,
    };

    Plotly.newPlot('plot', traces, layout);
  </script>
</body>
</html>
"""

def parse_box_table(raw_text):
    # Step 1: Decode HTML entities (e.g. &plusmn;)
    decoded = html.unescape(raw_text)

    # Step 2: Locate the relevant section after "Pareto Frontier Results"
    start_index = decoded.find("Pareto-Front for")
    if start_index == -1:
        raise ValueError("Could not find 'Pareto-Front for' marker.")
    table_section = decoded[start_index:]

    # Step 3: Split by real newlines (not Unicode lines)
    lines = table_section.split("\n")
    print(lines)
    # Step 4: Extract header line (the first ┃ row with text)
    header_line = None
    for line in lines:
        if line.startswith("┃"):
            header_line = line
            break

    if not header_line:
        raise ValueError("Header line with '┃' not found.")

    headers = [col.strip() for col in header_line.strip("┃").split("┃")]

    # Step 5: Parse data lines (│ rows with correct number of columns)
    data_rows = []
    for line in lines:
        if not line.startswith("│"):
            continue
        cols = [c.strip() for c in line.strip("│").split("│")]
        if len(cols) == len(headers):
            row = dict(zip(headers, cols))
            data_rows.append(row)
    print(data_rows)
    return headers, data_rows


def extract_mean(value):
    """Extract float from '0.844 ± 0.001' or similar."""
    match = re.match(r"([\d.]+)", str(value))
    return float(match.group(1)) if match else None

def extract_numeric(value):
    """Extract the first float number from a string like '0.844 ± 0.001'."""
    match = re.search(r"[\d.]+", value)
    return float(match.group(0)) if match else None

def build_js_pareto_data(dataset_model_dd_rows):
    """
    dataset_model_dd_rows: dict of form:
    {
        'method1': [row_dict1, row_dict2, ...],
        'method2': [row_dict1, row_dict2, ...],
        ...
    }
    Each row dict must contain keys like 'ACCURACY' and 'RUNTIME'.
    """
    pareto_data = {}

    for method, rows in dataset_model_dd_rows.items():
        method_points = []
        for row in rows:
            acc = row.get("ACCURACY", "")
            rt = row.get("RUNTIME", "")
            if acc is not None and rt is not None:
                method_points.append({ "x": acc, "y": rt })
        pareto_data[method] = method_points

    # Convert to pretty JS format
    js_string = json.dumps(pareto_data, indent=2)
    return js_string



target_model = "HoeffdingTreeClassifier"
target_metric = "ACCURACY-RUNTIME"
html_dir = "../exps"  # Change to your actual path



# File name pattern: dataset_model_dd.html
for target_dataset in ["Electricity", "SensorStream", "GasSensor", "Ozone",
                       "ForestCovertype", "RialtoBridgeTimelapse", "PokerHand",
                       "OutdoorObjects", "Electricity", "NOAAWeather"]:
    # Iterate over HTML files
    pattern = re.compile(
        rf"(.+)_{re.escape(target_dataset)}_{re.escape(target_model)}_{re.escape(target_metric)}.html")

    # Data structure to hold results
    parsed_data_by_dd = {}
    for filename in os.listdir(html_dir):

        match = pattern.match(filename)
        if not match:
            continue

        dd_name = match.group(1)
        file_path = os.path.join(html_dir, filename)

        # Parse the HTML content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html = f.read()
            soup = BeautifulSoup(html, 'html.parser')
            scripts = soup.find_all('script')

            # Find the script containing the variable
            for script in scripts:
                if script.string and 'var pareto_front_data' in script.string:
                    match = re.search(r'var\s+pareto_front_data\s*=\s*(\{.*?\});',
                                      script.string, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                        # Parse the JSON
                        pareto_front_data = json.loads(json_str)
                        break

            try:
                # Extract the relevant data
                accuracy_values = pareto_front_data["ACCURACY"]["RUNTIME"]["means"][
                    "ACCURACY"]
                runtime_values = pareto_front_data["ACCURACY"]["RUNTIME"]["means"][
                    "RUNTIME"]
                param_dicts = pareto_front_data["ACCURACY"]["RUNTIME"]["param_dicts"]

                # Transform into a three-dimensional list
                three_dimensional_list = [
                    [accuracy, runtime, param]
                    for accuracy, runtime, param in
                    zip(accuracy_values, runtime_values, param_dicts)
                ]

                parsed_data_by_dd[dd_name] = three_dimensional_list
            except:
                continue


    pareto_data = {}
    for dd, df in parsed_data_by_dd.items():
        points = []
        for data in df:
            points.append({"ACCURACY": data[0], "RUNTIME": data[1]})
        pareto_data[dd] = points


    js_code = build_js_pareto_data(pareto_data)
    #print(js_code)
    # Serialize the pareto data for JavaScript
    pareto_data_json = json.dumps(pareto_data, indent=2)
    #print(pareto_data_json)
    # Final HTML
    final_html = html_template.replace("{pareto_data_json}", js_code)

    # Save to a file
    with open(f"../paretos/{target_dataset}_{target_model}_{target_metric}.html", "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"HTML file created: {target_dataset}_{target_model}_{target_metric}.html")







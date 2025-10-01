import os
from bs4 import BeautifulSoup
import re
import json
# Define the target string and the replacement
search_text = 'var data = pareto_front_data;'
replacement_line = 'var data = '
html_dir = "../exps"

# Define the file path
for file_path in os.listdir(html_dir):

    with open(html_dir + "/" + file_path, 'r', encoding='utf-8',
              errors='ignore') as file:
        html = file.read()
    # Read the file
    with open(html_dir + "/" + file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        scripts = soup.find_all('script')

        headers = []
        data = []

        # Extract variables from script
        for script in scripts:
            content = script.string
            if content:
                # Find and parse headers
                header_match = re.search(r'var\s+tab_results_headers_json\s*=\s*(\[[^\]]*\])',
                                         content)
                if header_match:
                    headers = json.loads(header_match.group(1))

                # Find and parse data
                data_match = re.search(r'var\s+tab_results_csv_json\s*=\s*(\[\s*\[.*?\]\s*\])',
                                       content, re.DOTALL)
                if data_match:
                    data = json.loads(data_match.group(1))

        # Combine headers and data
        combined = [dict(zip(headers, row)) for row in data]
        accuracies = []
        runtimes = []
        for row in combined:
            if "ACCURACY" in row.keys() and "RUNTIME" in row.keys():
                accuracies.append(row['ACCURACY'])
                runtimes.append(row['RUNTIME'])

        scatter_data = {
            "ACCURACY": {
                "RUNTIME": {
                    "ACCURACY": accuracies,
                    "RUNTIME": runtimes
                }
            }
        }

        js_variable = f"{json.dumps(scatter_data, indent=4)};\n"

    # Replace the line
    new_lines = []
    for line in lines:
        if search_text in line:
            new_lines.append(replacement_line + js_variable + '\n')  # add newline character
        else:
            new_lines.append(line)


    # Write back to the file (or to a new file if you want to preserve the original)
    with open(html_dir + "/" + file_path, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)



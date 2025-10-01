from bs4 import BeautifulSoup, Tag, NavigableString
import os
import re

html_dir = "../exps"

def clear_tables():
    # Define the file path
    for file_path in os.listdir(html_dir):
        # Load HTML
        with open(html_dir + "/" + file_path, "r", encoding="utf-8") as file:
            html = file.read()

        soup = BeautifulSoup(html, "html.parser")

        # --- Remove <h1> tags with specific content
        keywords = ["Errors", "Args Overview", "Worker-Usage", "CPU/RAM-Usage (main)"]
        for h1 in soup.find_all("h1"):
            if any(keyword in h1.get_text() for keyword in keywords):
                h1.decompose()

        # --- Remove <div id="workerUsagePlot">
        worker_div = soup.find("div", id="workerUsagePlot")
        if worker_div:
            worker_div.decompose()

        worker_div = soup.find("div", id="mainWorkerCPURAM")
        if worker_div:
            worker_div.decompose()


        start_div = soup.find("div", id="pareto_front_graphs_container")
        end_h1 = soup.find("h1", string="\xa0Parallel Plot")

        if start_div and end_h1:
            current = start_div.next_sibling
            while current and current != end_h1:
                next_node = current.next_sibling
                if isinstance(current, Tag):
                    if current.name not in ["div", "h1", "script"]:
                        current.decompose()
                elif isinstance(current, NavigableString):
                    current.extract()
                current = next_node

        # Find the parallel plot marker div
        marker_div = soup.find("div", id="parallel-plot",
                               class_="invert_in_dark_mode")

        if marker_div:
            current = marker_div.next_sibling
            while current:
                next_node = current.next_sibling
                current.extract()
                current = next_node

        # --- Step 2: Remove <h2>Git-Version</h2> and its following <tt>
        git_header = soup.find("h2", string="Git-Version")
        if git_header:
            tt_tag = git_header.find_next_sibling("tt")
            git_header.decompose()
            if tt_tag:
                tt_tag.decompose()

        # --- Target function names to remove (escaped for regex)
        function_names = [
            "plotScatter2d",
            "plotScatter3d",
            "plotJobStatusDistribution",
            "plotBoxplot",
            "plotViolin",
            "plotHistogram",
            "plotHeatmap",
            "plotResultPairs",
            "plotResultEvolution",
            "plotExitCodesPieChart",
            "plotWorkerUsage",
            "plotCPUAndRAMUsage",
        ]

        # Regex pattern to match full JS function calls (even if commented or with extra semicolons)
        call_pattern = re.compile(
            r"^\s*(//\s*)?(" +
            "|".join(re.escape(name) for name in function_names) +
            r")\s*\(\)\s*;*\s*$",
            re.MULTILINE
        )

        # Process each <script> block
        for script in soup.find_all("script"):
            if script.string and "$(document).ready" in script.string:
                original_code = script.string
                # Remove lines that call the target functions
                cleaned_code = re.sub(call_pattern, '', original_code)
                script.string.replace_with(cleaned_code)

        # Save the cleaned HTML
        with open(html_dir + "/" + file_path, "w", encoding="utf-8") as file:
            file.write(str(soup))


def remove_css():
    #remove css, use a single css file instead
    # Define the file path
    for file_path in os.listdir(html_dir):
        # Load HTML
        if file_path.endswith(".css"):
            continue
        with open(html_dir + "/" + file_path, "r", encoding="utf-8", errors='ignore') as file:
            html = file.read()

        soup = BeautifulSoup(html, "html.parser")

        # Find and remove all <style> tags
        for style_tag in soup.find_all("style"):
            style_tag.decompose()

        # Create a new <link> tag for the external stylesheet
        link_tag = soup.new_tag("link", rel="stylesheet", href="expstyle.css")

        # Insert it into the <head>
        if soup.head:
            soup.head.append(link_tag)
        else:
            # If no <head> tag exists, insert one
            head_tag = soup.new_tag("head")
            head_tag.append(link_tag)
            soup.insert(0, head_tag)

        # Save the cleaned HTML
        with open(html_dir + "/" + file_path, "w", encoding="utf-8") as file:
            file.write(str(soup))


def update_gridjs_config():

    for file_path in os.listdir(html_dir):
        # Load HTML
        if file_path.endswith(".css"):
            continue
        with open(html_dir + "/" + file_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        # Regular expression to find all gridjs.Grid({ ... })
        pattern = r'gridjs\.Grid\(\{([\s\S]*?)\}\)'

        def replace_function(match):
            content = match.group(1)

            # Add pagination: true if not present
            if 'pagination: true' not in content:
                content = '\n\t\t\t\t\tpagination: true,' + content

            # Set search: false
            if 'search:' in content:
                content = re.sub(r'search:\s*true', 'search: false', content)
            else:
                content = re.sub(r'(\{)', r'\1\n  search: false,', content)

            return f'gridjs.Grid({{{content}}})'

        # Replace all occurrences in the HTML content
        updated_html = re.sub(pattern, replace_function, html_content)
        with open(html_dir + "/" + file_path, "w", encoding="utf-8") as file:
            file.write(updated_html)

remove_css()
clear_tables()
update_gridjs_config()
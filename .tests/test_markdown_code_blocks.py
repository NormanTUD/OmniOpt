import os
import subprocess
import json
import argparse

def validate_bash(code):
    try:
        subprocess.run(['bash', '-n', '-c', code], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def validate_json(code):
    try:
        json.loads(code)
        return True
    except json.JSONDecodeError:
        return False

def validate_python(code):
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

def process_md_files(directory, debug=False):
    if not os.path.isdir(directory):
        print(f"Error: Directory {directory} does not exist.")
        return

    for md_file in os.listdir(directory):
        if md_file.endswith('.md'):
            file_path = os.path.join(directory, md_file)
            print(f"Processing file: {file_path}")

            with open(file_path, 'r') as file:
                code_block = None
                lang = None
                for line in file:
                    if debug:
                        print(f"Processing line: {line.strip()}")

                    if line.startswith('```'):
                        if lang is None:  # First code block opening
                            lang = line.strip('```').strip()
                            code_block = ''
                            if debug:
                                print(f"Detected language: {lang}")
                        elif lang:  # Code block end
                            if line.strip('```').strip() == lang:
                                if debug:
                                    print(f"End of {lang} block detected.")

                                if lang == 'bash' and validate_bash(code_block):
                                    print(f"Valid Bash code block found in {md_file}")
                                elif lang == 'json' and validate_json(code_block):
                                    print(f"Valid JSON block found in {md_file}")
                                elif lang == 'python' and validate_python(code_block):
                                    print(f"Valid Python block found in {md_file}")
                                else:
                                    print(f"Invalid {lang.capitalize()} code block in {md_file}")
                                lang = None
                                code_block = None
                        continue
                    
                    if lang:
                        code_block += line
                        if debug:
                            print(f"Current code block content:\n{code_block}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate code blocks in Markdown files.")
    parser.add_argument('directory', help="Directory containing .md files to process")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode to show detailed output")
    args = parser.parse_args()

    process_md_files(args.directory, debug=args.debug)

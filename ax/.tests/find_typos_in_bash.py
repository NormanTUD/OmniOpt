from pprint import pprint
import argparse
import os
import re
import sys
from spellchecker import SpellChecker
from rich.progress import Progress
from rich.console import Console
from rich.table import Table

# Initialize the spellchecker
spell = SpellChecker(language='en')

def read_file_to_array(file_path):
    if not os.path.exists(file_path):
        print(f"Cannot find file {file_path}")
        sys.exit(9)
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

# Read the whitelist from the file
whitelisted = read_file_to_array(".tests/whitelisted_words")

console = Console()

# Function to extract strings from Bash files
def extract_strings_from_bash(bash_script_path):
    with open(bash_script_path, 'r') as file:
        bash_content = file.read()

    # Regex for strings in single and double quotes
    string_pattern = r"(\".*?\"|'.*?')"
    strings = re.findall(string_pattern, bash_content)

    # Remove the quotes
    strings = [s[1:-1] for s in strings]

    return strings

def filter_and_spellcheck_words(words):
    misspelled_words = []
    for word in words:
        # Check if the word consists of letters and is not all uppercase
        if word.isalpha() and word not in whitelisted:
            # Check if the word is misspelled
            if word not in spell:  # Check if the word is not recognized
                misspelled_words.append(word)
    return misspelled_words

# Main function
def main():
    # Handle arguments
    parser = argparse.ArgumentParser(description='Extract strings from Bash files and display filtered misspelled words.')
    parser.add_argument('bash_files', nargs='+', help='Path(s) to the Bash files')
    args = parser.parse_args()

    file_count_with_errors = 0

    # Table for results
    table = Table(title="Misspelled Words in bash files:", show_lines=True)
    table.add_column("File", justify="center", style="cyan", no_wrap=True)
    table.add_column("Words", justify="left", style="magenta")

    # Progress bar with rich
    with Progress(transient=True) as progress:
        task = progress.add_task("[green]Processing files...", total=len(args.bash_files))

        # Process each file
        for bash_file in args.bash_files:
            progress.advance(task)

            if not os.path.exists(bash_file):
                console.print(f"[bold red]Warning: File '{bash_file}' does not exist![/bold red]")
                continue

            # Extract strings
            strings = extract_strings_from_bash(bash_file)

            # Split strings by space, comma, semicolon, etc.
            words = []
            for string in strings:
                words.extend(re.split(r'[ ,;]', string))

            # Filter and spellcheck the words
            misspelled_words = filter_and_spellcheck_words(words)
            misspelled_words = list(set(misspelled_words))

            # If there are misspelled words, add them to the table and increase the error count
            if misspelled_words:
                file_count_with_errors += 1
                table.add_row(bash_file, ", ".join(misspelled_words))

    if file_count_with_errors > 0:
        # Display the table before the progress bar ends
        console.print(table)

    # Set the exit code based on the number of files with errors
    sys.exit(file_count_with_errors)

if __name__ == "__main__":
    main()

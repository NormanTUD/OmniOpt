import os
import sys
import re
import argparse
from spellchecker import SpellChecker
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

import argparse

parser = argparse.ArgumentParser(description='Analyze PHP files and check the spelling of string literals.')
parser.add_argument(
    "--lang", default="en", help="Specify the language (default is 'en')"
)
parser.add_argument('files', metavar='FILE', nargs='+', help='The PHP files to analyze.')
args = parser.parse_args()

# Initialize spellchecker with English dictionary
spell = None
try:
    spell = SpellChecker(language=args.lang)
except KeyboardInterrupt:
        console.print("[red]Cancelled script by using CTRL + C[/red]")

console = Console()

def read_file_to_array(file_path):
    if not os.path.exists(file_path):
        console.print(f"[red]Cannot find file {file_path}[/red]")
        sys.exit(9)
    with open(file_path, mode='r', encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

# Read the whitelist from the file
IGNORE_PATTERNS = read_file_to_array(".tests/whitelisted_words")

def is_ignored(word):
    for pattern in IGNORE_PATTERNS:
        if word.lower() == pattern.lower():
            return True
    return False

def is_valid_word(word):
    return re.match(r'^[a-zA-Z]{1,}$', word) is not None

def extract_php_strings(content):
    """Extract all single and double quoted strings from PHP content within <?php ... ?> tags."""
    # Pattern to match PHP tags and capture quoted strings within
    pattern = r'<\?php(.*?)\?>'
    php_code_blocks = re.findall(pattern, content, re.DOTALL)

    strings = []
    for block in php_code_blocks:
        # Extract single-quoted strings
        single_quoted_pattern = r"'((?:\\.|[^'\\])*)'"
        strings.extend(re.findall(single_quoted_pattern, block))

        # Extract double-quoted strings
        double_quoted_pattern = r'"((?:\\.|[^"\\])*)"'
        strings.extend(re.findall(double_quoted_pattern, block))

    return strings

def clean_word(word):
    after = re.sub(r'[^.()[]\'a-zA-Z0-9_/-]', '', word)
    return after

def analyze_php_file(filepath, progress):
    with open(filepath, mode='r', encoding='utf-8') as file:
        content = file.read()

    strings = extract_php_strings(content)
    possibly_incorrect_words = []

    total_words = sum(len(entry.split()) for entry in strings)
    current_word_count = 0

    # Create a progress task for each file
    task_id = progress.add_task(f"[bold]Analyzing {filepath}[/bold]", total=total_words)

    for entry in strings:
        words = entry.split()
        for word in words:
            word = clean_word(word)
            current_word_count += 1

            progress.update(task_id, advance=1, description=f"[bold]{filepath}: Checking word {current_word_count}/{total_words}...[/bold]")

            if is_valid_word(word):
                if not is_ignored(word):
                    if spell.correction(word) != word:
                        if word not in possibly_incorrect_words:
                            possibly_incorrect_words.append(word)

    progress.update(task_id, completed=True)
    return possibly_incorrect_words

def main():
    typo_files = 0
    results = {}

    with Progress(transient=True) as progress:
        for filepath in args.files:
            if os.path.splitext(filepath)[1] == '.php':
                possibly_incorrect_words = analyze_php_file(filepath, progress)
                results[filepath] = possibly_incorrect_words

                if possibly_incorrect_words:
                    typo_files += 1
                    console.print(f"\n[red]Unknown or misspelled words in {filepath}:[/red]")
                    console.print("\n[red]" + "\n".join(possibly_incorrect_words) + "[/red]")

    # Summary Table
    if results:
        table = Table(title="Summary of Misspelled Words")

        table.add_column("File", justify="left")
        table.add_column("Misspelled Words:", justify="left")

        files_with_errors = 0
        for filepath, words in results.items():
            if len(words):
                table.add_row(filepath, ', '.join(words) if words else "None")
                files_with_errors += 1

        if files_with_errors:
            console.print(table)

    sys.exit(typo_files)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("[red]Cancelled script by using CTRL + C[/red]")

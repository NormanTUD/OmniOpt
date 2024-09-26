import os
import sys
import re
import argparse
from spellchecker import SpellChecker
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

# Initialize spellchecker with English dictionary
spell = SpellChecker(language='en')
console = Console()

def read_file_to_array(file_path):
    if not os.path.exists(file_path):
        console.print(f"[red]Cannot find file {file_path}[/red]")
        sys.exit(9)
    with open(file_path, 'r') as file:
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

def extract_single_quoted_strings(content):
    """Extract all single-quoted strings from JavaScript content."""
    pattern = r"'([^'\\]*(?:\\.[^'\\]*)*)'"
    return re.findall(pattern, content)

def extract_double_quoted_strings(content):
    """Extract all double-quoted strings from JavaScript content."""
    pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
    return re.findall(pattern, content)

def extract_template_strings(content):
    """Extract all template strings from JavaScript content."""
    pattern = r'`([^`\\]*(?:\\.[^`\\]*)*)`'
    return re.findall(pattern, content)

def extract_strings_from_js(filepath):
    """Extract all string literals from a JavaScript file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    strings = []
    strings.extend(extract_single_quoted_strings(content))
    strings.extend(extract_double_quoted_strings(content))
    strings.extend(extract_template_strings(content))

    return strings

def clean_word(word):
    after = re.sub(r'[^\'a-zA-Z0-9_/-]', '', word)
    return after

def analyze_js_file(filepath, progress, task_id):
    strings = extract_strings_from_js(filepath)
    possibly_incorrect_words = []
    
    total_words = sum(len(string.split()) for string in strings)
    current_word_count = 0
    
    progress.update(task_id, description=f"[bold]Analyzing {filepath}...[/bold]")

    for string in strings:
        words = string.split()
        for word in words:
            word = clean_word(word)
            current_word_count += 1
            
            progress.update(task_id, description=f"[bold]{filepath}: Checking word {current_word_count}/{total_words}...[/bold]")

            if is_valid_word(word):
                if not is_ignored(word):
                    if spell.correction(word) != word:
                        if word not in possibly_incorrect_words:
                            possibly_incorrect_words.append(word)

    progress.update(task_id, description=f"[dim]Checked {total_words} words in {filepath}.[/dim]", completed=True)
    return possibly_incorrect_words

def main():
    parser = argparse.ArgumentParser(description='Analyze JavaScript files and check the spelling of string literals.')
    parser.add_argument('files', metavar='FILE', nargs='+', help='The JavaScript files to analyze.')
    args = parser.parse_args()

    typo_files = 0
    results = {}

    with Progress() as progress:
        task_id = progress.add_task("Analyzing files...", total=len(args.files))
        
        for filepath in args.files:
            if os.path.splitext(filepath)[1] == '.js':
                possibly_incorrect_words = analyze_js_file(filepath, progress, task_id)
                results[filepath] = possibly_incorrect_words
                
                if possibly_incorrect_words:
                    typo_files += 1
                    console.print(f"\n[red]Unknown or misspelled words in {filepath}:[/red]")
                    console.print("\n[red]'\n'" + "\n".join(possibly_incorrect_words) + "[/red]")

            progress.update(task_id, advance=1)

    # Summary Table
    if results:
        table = Table(title="Summary of Misspelled Words")

        table.add_column("File", justify="left")
        table.add_column("Misspelled Words", justify="left")

        for filepath, words in results.items():
            table.add_row(filepath, ', '.join(words) if words else "None")

        console.print(table)

    sys.exit(typo_files)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("[red]Cancelled script by using CTRL + C[/red]")


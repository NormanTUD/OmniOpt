import os
import sys
import re
import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.autodeps import ensure_imports_or_exit
from _framework.helpers import find_files

ensure_imports_or_exit((("spellchecker", "pyspellchecker"), ("rich", "rich")))

from spellchecker import SpellChecker

REPO_ROOT = THIS_DIR.parent
GUI_DIR = REPO_ROOT / ".gui"

MAX_LINE_LENGTH = 1000


def is_vendor_or_minified(path):
    """Systematically skip vendor/minified JS (jquery, gridjs, plotly, ...).

    Minified bundles (e.g. ``*.min.js`` or ``gridjs.umd.js``) either carry
    the ``.min.js`` suffix or are written on extremely long lines that no
    hand-written source would have.
    """
    if path.name.endswith(".min.js"):
        return True
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if len(line) > MAX_LINE_LENGTH:
                    return True
    except OSError:
        return False
    return False

def read_file_to_array(console, file_path):
    if not os.path.exists(file_path):
        console.print(f"[red]Cannot find file {file_path}[/red]")
        sys.exit(9)
    with open(file_path, mode='r', encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


def is_ignored(word, ignore_set):
    return word.lower() in ignore_set

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

def extract_single_line_comments(content):
    """Extract all single-line comments from JavaScript content."""
    pattern = r'//(.*)'
    return re.findall(pattern, content)

def extract_multi_line_comments(content):
    """Extract all multi-line comments from JavaScript content."""
    pattern = r'/\*([\s\S]*?)\*/'
    return re.findall(pattern, content)

def extract_strings_and_comments_from_js(filepath):
    """Extract all string literals and comments from a JavaScript file."""
    with open(filepath, mode='r', encoding='utf-8') as file:
        content = file.read()

    strings_and_comments = []
    # Extract all string literals
    strings_and_comments.extend(extract_single_quoted_strings(content))
    strings_and_comments.extend(extract_double_quoted_strings(content))
    strings_and_comments.extend(extract_template_strings(content))
    # Extract all comments
    strings_and_comments.extend(extract_single_line_comments(content))
    strings_and_comments.extend(extract_multi_line_comments(content))

    return strings_and_comments

def clean_word(word):
    after = re.sub(r'[^.()[]\'a-zA-Z0-9_/-]', '', word)
    return after

def analyze_js_file(spell, ignore_set, filepath, progress):
    strings_and_comments = extract_strings_and_comments_from_js(filepath)
    possibly_incorrect_words = []

    total_words = sum(len(entry.split()) for entry in strings_and_comments)
    current_word_count = 0

    # Create a progress task for each file
    task_id = progress.add_task(f"[bold]Analyzing {filepath}[/bold]", total=total_words)

    for entry in strings_and_comments:
        words = entry.split()
        for word in words:
            word = clean_word(word)
            current_word_count += 1

            progress.update(task_id, advance=1, description=f"[bold]{filepath}: Checking word {current_word_count}/{total_words}...[/bold]")

            if is_valid_word(word):
                if not is_ignored(word, ignore_set):
                    if spell.correction(word) != word:
                        if word not in possibly_incorrect_words:
                            possibly_incorrect_words.append(word)

    progress.update(task_id, completed=True)
    progress.remove_task(task_id)
    return possibly_incorrect_words

def main():
    parser = argparse.ArgumentParser(description='Analyze JavaScript files and check the spelling of string literals and comments.')
    parser.add_argument('files', metavar='FILE', nargs='*', help='The JavaScript files to analyze.')
    args = parser.parse_args()

    files = args.files
    if not files:
        files = [
            str(p) for p in find_files(GUI_DIR, (".js",))
            if not is_vendor_or_minified(p)
        ]

    console = Console()

    # Initialize spellchecker with English dictionary (deferred so a
    # KeyboardInterrupt here can still be caught by the if __name__ guard).
    spell = SpellChecker(language='en')

    # Read the whitelist from the file
    IGNORE_PATTERNS = read_file_to_array(console, ".tests/whitelisted_words")
    IGNORE_SET = {p.lower() for p in IGNORE_PATTERNS if p}

    typo_files = 0
    results = {}

    with Progress(transient=True) as progress:
        for filepath in files:
            if os.path.splitext(filepath)[1] == '.js':
                possibly_incorrect_words = analyze_js_file(spell, IGNORE_SET, filepath, progress)
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
                files_with_errors = files_with_errors + 1

        if files_with_errors:
            console.print(table)

    sys.exit(typo_files)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Cancelled script by using CTRL + C", file=sys.stderr)
        sys.exit(0)

import os
import sys
import ast
import argparse
import re
from pprint import pprint
from pathlib import Path
from rich.progress import Progress
from rich.console import Console

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _framework.autodeps import ensure_imports_or_exit
from _framework.helpers import erase_progress_trail, find_files

ensure_imports_or_exit((("spellchecker", "pyspellchecker"), ("rich", "rich")))

from spellchecker import SpellChecker

REPO_ROOT = THIS_DIR.parent

def dier(msg):
    pprint(msg)
    sys.exit(10)


def read_file_to_array(file_path):
    if not os.path.exists(file_path):
        print(f"Cannot find file {file_path}")
        sys.exit(9)
    with open(file_path, mode='r', encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


# Read the whitelist from the file
IGNORE_PATTERNS = read_file_to_array(".tests/whitelisted_words")
IGNORE_SET = {p.lower() for p in IGNORE_PATTERNS if p}


def is_ignored(word):
    """Check if the word should be ignored (O(1) lookup)."""
    return word.lower() in IGNORE_SET

def is_valid_word(word):
    """Check if the word contains only alphanumeric characters (ignores anything with special characters)."""
    return re.match(r'^[a-zA-Z]{1,}$', word) is not None

def extract_strings_from_ast(node):
    """Extract all string literals from the AST."""
    # Check for string constants
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]

    # Handle lists and tuples of string literals
    if isinstance(node, (ast.List, ast.Tuple)):
        strings = []
        for element in node.elts:
            strings.extend(extract_strings_from_ast(element))
        return strings

    # Handle binary operations with string concatenation
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return extract_strings_from_ast(node.left) + extract_strings_from_ast(node.right)

    return []  # Return empty list if no string literals are found

def clean_word(word):
    after = re.sub(r'[^\'a-zA-Z0-9_/-]', '', word)

    return after

def analyze_file(spell, filepath, progress, task_id):
    """Analyze a Python file and check the spelling of string literals."""
    with open(filepath, mode='r', encoding='utf-8') as file:
        content = file.read()

    tree = ast.parse(content)
    strings = []

    # Traverse the AST to extract strings
    for node in ast.walk(tree):
        strings.extend(extract_strings_from_ast(node))

    # Update the total number of string literals in the progress bar
    if progress is not None:
        progress.update(task_id, total=len(strings))

    # Process the strings
    possibly_incorrect_words = []
    strings = list(set(strings))
    for _, string in enumerate(strings):
        words = string.split()
        for word in words:
            word = clean_word(word)
            if is_valid_word(word):
                if not is_ignored(word):
                    if spell.correction(word) != word:
                        if word not in possibly_incorrect_words:
                            print(word)
                            possibly_incorrect_words.append(word)
        if progress is not None:
            progress.advance(task_id)

    return possibly_incorrect_words

def main():
    # Initialize spellchecker (deferred so KeyboardInterrupt can be
    # caught cleanly by the if __name__ guard).
    spell = SpellChecker(language='en')

    parser = argparse.ArgumentParser(description='Analyze Python scripts and check the spelling of string literals.')
    parser.add_argument('files', metavar='FILE', nargs='*', help='The Python files to analyze.')
    args = parser.parse_args()

    files = args.files
    if not files:
        files = [str(p) for p in find_files(REPO_ROOT, (".py",))]

    console = Console()
    typo_files = 0

    # Use rich's progress bar only when running in an interactive
    # terminal; otherwise the auto-refresh becomes pure overhead and
    # dominates the runtime (many add_task calls = many subprocess
    # waitpid's per file).
    use_progress = console.is_terminal
    if use_progress:
        with Progress(console=console, transient=True, auto_refresh=True) as progress:
            for filepath in files:
                task_id = progress.add_task(f"[cyan]Analyzing {filepath}", total=1)
                try:
                    possibly_incorrect_words = analyze_file(spell, filepath, progress, task_id)
                    if possibly_incorrect_words:
                        typo_files += 1
                        console.print(f"[red]Unknown or misspelled words in {filepath}: {possibly_incorrect_words}")
                except SyntaxError:
                    print(f"File {filepath} is not valid python. Cannot continue.")
                    sys.exit(1)
                finally:
                    progress.remove_task(task_id)
        if typo_files == 0:
            erase_progress_trail(console)
    else:
        for i, filepath in enumerate(files, start=1):
            if i % 20 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] analyzing {filepath}", file=sys.stderr)
            try:
                possibly_incorrect_words = analyze_file(spell, filepath, None, None)
                if possibly_incorrect_words:
                    typo_files += 1
                    console.print(f"[red]Unknown or misspelled words in {filepath}: {possibly_incorrect_words}")
            except SyntaxError:
                print(f"File {filepath} is not valid python. Cannot continue.")
                sys.exit(1)

    sys.exit(typo_files)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Cancelled script by using CTRL c", file=sys.stderr)
        sys.exit(0)

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
from _framework.helpers import erase_progress_trail, find_files

ensure_imports_or_exit((("spellchecker", "pyspellchecker"), ("rich", "rich")))

from spellchecker import SpellChecker

REPO_ROOT = THIS_DIR.parent


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
    # Only letters (no digits or punctuation), length >=1
    return re.match(r'^[a-zA-Z]+$', word) is not None

def clean_word(word):
    # Remove unwanted characters from words
    cleaned = re.sub(r'[^a-zA-Z\-]', '', word)
    return cleaned

def extract_words_from_markdown(content):
    """
    Extract words from Markdown content, ignoring Markdown syntax elements.
    We'll:
    - Remove code blocks (```...```)
    - Remove inline code (`...`)
    - Remove Markdown links [text](url)
    - Remove images ![alt](url)
    - Remove HTML tags if any
    - Then split by whitespace and punctuation to get words.
    """
    # Remove fenced code blocks
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    # Remove inline code blocks
    content = re.sub(r'`[^`]*`', '', content)
    # Remove images ![alt](url)
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    # Remove links [text](url)
    content = re.sub(r'\[.*?\]\(.*?\)', '', content)
    # Remove HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    # Remove Markdown headers, blockquotes, lists marks, etc.
    content = re.sub(r'^[>\-\*\#\d\.\s]+', '', content, flags=re.MULTILINE)

    # Now split into words by non-letter characters
    words = re.findall(r'\b[a-zA-Z\-]+\b', content)

    return words

def analyze_markdown_file(spell, ignore_set, filepath, progress):
    with open(filepath, mode='r', encoding='utf-8') as file:
        content = file.read()

    words = extract_words_from_markdown(content)
    possibly_incorrect_words = []

    total_words = len(words)
    if total_words == 0:
        return possibly_incorrect_words

    # Create a progress task for each file (only when progress is enabled)
    task_id = None
    if progress is not None:
        task_id = progress.add_task(f"[bold]Analyzing {filepath}[/bold]", total=total_words)

    current_word_count = 0
    seen_words = set()

    for word in words:
        cleaned_word = clean_word(word)
        current_word_count += 1

        if task_id is not None:
            progress.update(
                task_id, advance=1,
                description=f"[bold]{filepath}: Checking word {current_word_count}/{total_words}...[/bold]",
            )

        if is_valid_word(cleaned_word):
            if not is_ignored(cleaned_word, ignore_set):
                corrected = spell.correction(cleaned_word)
                if corrected != cleaned_word:
                    # Avoid duplicates
                    lowered = cleaned_word.lower()
                    if lowered not in seen_words:
                        possibly_incorrect_words.append(cleaned_word)
                        seen_words.add(lowered)

    if task_id is not None:
        progress.update(task_id, completed=True)
        progress.remove_task(task_id)
    return possibly_incorrect_words

def main():
    parser = argparse.ArgumentParser(description='Analyze Markdown files and check the spelling of words.')
    parser.add_argument(
        "--lang", default="en", help="Specify the language (default is 'en')"
    )
    parser.add_argument('files', metavar='FILE', nargs='*', help='The Markdown files to analyze.')
    args = parser.parse_args()

    if not args.files:
        args.files = [str(p) for p in find_files(REPO_ROOT, (".md",))]

    console = Console()

    # Initialize spellchecker with chosen language dictionary
    # (deferred so a KeyboardInterrupt here can still be caught cleanly).
    try:
        spell = SpellChecker(language=args.lang)
    except Exception as e:
        console.print(f"[red]Failed to initialize SpellChecker with language '{args.lang}': {e}[/red]")
        sys.exit(1)

    # Read the whitelist from the file (words to ignore).
    # Convert to a set of lower-cased patterns for O(1) lookup instead of
    # scanning the full list per word (which dominates the runtime for
    # large docs).
    IGNORE_PATTERNS = read_file_to_array(console, ".tests/whitelisted_words")
    IGNORE_SET = {p.lower() for p in IGNORE_PATTERNS if p}

    typo_files = 0
    results = {}

    use_progress = console.is_terminal
    if use_progress:
        with Progress(transient=True) as progress:
            for filepath in args.files:
                if os.path.splitext(filepath)[1].lower() == '.md':
                    possibly_incorrect_words = analyze_markdown_file(spell, IGNORE_SET, filepath, progress)
                    results[filepath] = possibly_incorrect_words

                    if possibly_incorrect_words:
                        typo_files += 1
                        console.print(f"\n[red]Unknown or misspelled words in {filepath}:[/red]")
                        console.print("\n[red]" + "\n".join(possibly_incorrect_words) + "[/red]")
    else:
        for i, filepath in enumerate(args.files, start=1):
            if os.path.splitext(filepath)[1].lower() == '.md':
                if i % 5 == 0 or i == 1:
                    print(f"  [{i}] analyzing {filepath}", file=sys.stderr)
                possibly_incorrect_words = analyze_markdown_file(spell, IGNORE_SET, filepath, None)
                results[filepath] = possibly_incorrect_words

                if possibly_incorrect_words:
                    typo_files += 1
                    print(f"\nUnknown or misspelled words in {filepath}:", file=sys.stderr)
                    print("\n".join(possibly_incorrect_words), file=sys.stderr)

    # Summary Table
    if results:
        table = Table(title="Summary of Misspelled Words")

        table.add_column("File", justify="left")
        table.add_column("Misspelled Words", justify="left")

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
        sys.exit(main())
    except KeyboardInterrupt:
        print("Cancelled script by using CTRL + C", file=sys.stderr)
        sys.exit(0)

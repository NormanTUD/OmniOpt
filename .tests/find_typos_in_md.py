import os
import sys
import re
import argparse
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

try:
    from spellchecker import SpellChecker
except ModuleNotFoundError:
    print("spellchecker could not be loaded")
    sys.exit(0)

parser = argparse.ArgumentParser(description='Analyze Markdown files and check the spelling of words.')
parser.add_argument(
    "--lang", default="en", help="Specify the language (default is 'en')"
)
parser.add_argument('files', metavar='FILE', nargs='+', help='The Markdown files to analyze.')
args = parser.parse_args()

console = Console()

# Initialize spellchecker with chosen language dictionary
try:
    spell = SpellChecker(language=args.lang)
except Exception as e:
    console.print(f"[red]Failed to initialize SpellChecker with language '{args.lang}': {e}[/red]")
    sys.exit(1)

def read_file_to_array(file_path):
    if not os.path.exists(file_path):
        console.print(f"[red]Cannot find file {file_path}[/red]")
        sys.exit(9)
    with open(file_path, mode='r', encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

# Read the whitelist from the file (words to ignore)
IGNORE_PATTERNS = read_file_to_array(".tests/whitelisted_words")

def is_ignored(word):
    for pattern in IGNORE_PATTERNS:
        if word.lower() == pattern.lower():
            return True
    return False

def is_valid_word(word):
    # Only letters (no digits or punctuation), length >=1
    return re.match(r'^[a-zA-Z]+$', word) is not None

def clean_word(word):
    # Remove unwanted characters from words
    cleaned = re.sub(r'[^a-zA-Z\-]', '', word)
    return cleaned

def extract_words_from_markdown(content):
    """
    Extract words from markdown content, ignoring markdown syntax elements.
    We'll:
    - Remove code blocks (```...```)
    - Remove inline code (`...`)
    - Remove markdown links [text](url)
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
    # Remove markdown headers, blockquotes, lists marks, etc.
    content = re.sub(r'^[>\-\*\#\d\.\s]+', '', content, flags=re.MULTILINE)

    # Now split into words by non-letter characters
    words = re.findall(r'\b[a-zA-Z\-]+\b', content)

    return words

def analyze_markdown_file(filepath, progress):
    with open(filepath, mode='r', encoding='utf-8') as file:
        content = file.read()

    words = extract_words_from_markdown(content)
    possibly_incorrect_words = []

    total_words = len(words)
    if total_words == 0:
        return possibly_incorrect_words

    # Create a progress task for each file
    task_id = progress.add_task(f"[bold]Analyzing {filepath}[/bold]", total=total_words)

    current_word_count = 0
    seen_words = set()

    for word in words:
        cleaned_word = clean_word(word)
        current_word_count += 1

        progress.update(task_id, advance=1, description=f"[bold]{filepath}: Checking word {current_word_count}/{total_words}...[/bold]")

        if is_valid_word(cleaned_word):
            if not is_ignored(cleaned_word):
                corrected = spell.correction(cleaned_word)
                if corrected != cleaned_word:
                    # Avoid duplicates
                    lowered = cleaned_word.lower()
                    if lowered not in seen_words:
                        possibly_incorrect_words.append(cleaned_word)
                        seen_words.add(lowered)

    progress.update(task_id, completed=True)
    return possibly_incorrect_words

def main():
    typo_files = 0
    results = {}

    with Progress(transient=True) as progress:
        for filepath in args.files:
            if os.path.splitext(filepath)[1].lower() == '.md':
                possibly_incorrect_words = analyze_markdown_file(filepath, progress)
                results[filepath] = possibly_incorrect_words

                if possibly_incorrect_words:
                    typo_files += 1
                    console.print(f"\n[red]Unknown or misspelled words in {filepath}:[/red]")
                    console.print("\n[red]" + "\n".join(possibly_incorrect_words) + "[/red]")

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
        main()
    except KeyboardInterrupt:
        console.print(f"[red]Cancelled script for {', '.join(args.files)} by using CTRL + C[/red]")

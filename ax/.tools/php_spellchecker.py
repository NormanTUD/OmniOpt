import os
import sys
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from spellchecker import SpellChecker
import emoji
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table

spell = SpellChecker(language='en')

console = Console()

def read_file_to_array(file_path):
    if not os.path.exists(file_path):
        print(f"Cannot find file {file_path}")
        sys.exit(9)
    with open(file_path, mode='r', encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


# Liste von Regex-Mustern, die ignoriert werden sollen (z.B. technische Begriffe, Abkürzungen, usw.)
whitelisted = read_file_to_array(".tests/whitelisted_words")

def extract_visible_text_from_html(html_content):
    try:
        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()

        # Extract the visible text
        visible_text = soup.get_text(separator='\n')

        # Clean up unnecessary whitespace and empty lines
        clean_text = "\n".join([line.strip() for line in visible_text.splitlines() if line.strip()])
        return clean_text
    except Exception as e:
        console.print(f"[red]Error processing HTML content: {e}[/red]")
        return None

def clean_word(word):
    # Remove punctuation and split hyphenated words
    word = re.sub(r'[^\w\s/\'-_]', '', word)  # Remove punctuation except hyphen
    return word.split('-')  # Split on hyphens to check each part separately

def is_valid_word(word):
    if "/" in word:
        return False
    return word.isalpha()

def filter_emojis(text):
    # Remove emojis and other non-alphanumeric characters
    return ''.join(char for char in text if not emoji.is_emoji(char))

def check_spelling(text):
    try:

        # Split the text into words
        words = text.split()

        # Filter out words that match any of the ignored patterns or contain emojis
        filtered_words = []
        for word in words:
            cleaned_word_parts = clean_word(word)
            for part in cleaned_word_parts:
                word = filter_emojis(part)

                word = word.strip()
                word = word.rstrip()

                if word and is_valid_word(word):
                    if not word.lower() in whitelisted:
                        filtered_words.append(word)

        # Find words that are misspelled
        misspelled = spell.unknown(filtered_words)

        return sorted(misspelled)  # Sort the misspelled words alphabetically
    except Exception as e:
        console.print(f"[red]Error checking spelling: {e}[/red]")
        return None

def process_php_file(file_path):
    try:
        # Execute the PHP file and capture the output
        _result = subprocess.run(['php', file_path], capture_output=True, text=True, check=False)
        html_content = _result.stdout

        # Extract the visible text from HTML content
        extracted_text = extract_visible_text_from_html(html_content)

        if extracted_text:
            # Perform spell check on the extracted text
            misspelled_words = check_spelling(extracted_text)

            if misspelled_words:
                return (file_path, misspelled_words)

        return None
    except Exception as e:
        console.print(f"[red]Error processing {file_path}: {e}[/red]")
        return None

def process_directory(directory_path):
    total_errors = 0
    php_files = [os.path.join(root, file) for root, _, files in os.walk(directory_path) for file in files if file.endswith(".php")]

    # Use ThreadPoolExecutor to parallelize file processing
    _errors_found = []
    with ThreadPoolExecutor() as executor, Progress(
        TextColumn("[bold blue]Processing:[/bold blue] {task.fields[filename]}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=True
    ) as progress:
        task = progress.add_task("Processing files", total=len(php_files), filename="")

        futures = {executor.submit(process_php_file, file_path): file_path for file_path in php_files}
        for future in as_completed(futures):
            file_path = futures[future]
            progress.update(task, advance=1, filename=file_path)

            _result = future.result()
            if _result:
                _errors_found.append(_result)

    return _errors_found

def show_summary_table(_errors_found):
    if _errors_found:
        console.print("\n[bold red]Summary of Misspelled Words:[/bold red]")
        summary_table = Table(title="Misspelled Words Summary", title_style="red bold", box=None)
        summary_table.add_column("File", justify="left", style="bold yellow")
        summary_table.add_column("Misspelled Words", justify="left", style="red")

        for file_path, misspelled_words in _errors_found:
            summary_table.add_row(file_path, ", ".join(misspelled_words))

        console.print(summary_table)

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            console.print("[red]Usage: python php_spellcheck.py <file_or_directory>[/red]")
            sys.exit(1)

        input_path = sys.argv[1]

        if os.path.isfile(input_path):
            errors_found = []
            _result = process_php_file(input_path)
            if _result:
                errors_found.append(_result)
            show_summary_table(errors_found)

            sys.exit(len(errors_found))

        elif os.path.isdir(input_path):
            errors_found = process_directory(input_path)
            show_summary_table(errors_found)

            sys.exit(len(errors_found))
        else:
            console.print(f"[red]{input_path} is not a valid file or directory.[/red]")
            sys.exit(254)

    except KeyboardInterrupt:
        console.print("[yellow]You pressed CTRL+C. Script will be cancelled.[/yellow]")
        sys.exit(255)

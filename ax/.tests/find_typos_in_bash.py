import argparse
import os
import re
from spellchecker import SpellChecker
from rich.progress import Progress
from rich.console import Console
from rich.table import Table

# Initialisiere den Spellchecker
spell = SpellChecker(language='en')

def read_file_to_array(file_path):
    if not os.path.exists(file_path):
        print(f"Cannot find file {file_path}")
        sys.exit(9)
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

# Whitelist aus Datei einlesen
whitelisted = read_file_to_array(".tests/whitelisted_words")

console = Console()

# Funktion zum Extrahieren von Strings aus Bash-Dateien
def extract_strings_from_bash(bash_script_path):
    with open(bash_script_path, 'r') as file:
        bash_content = file.read()

    # Regex für Strings in einfachen und doppelten Anführungszeichen
    string_pattern = r"(\".*?\"|'.*?')"
    strings = re.findall(string_pattern, bash_content)

    # Entferne die Anführungszeichen
    strings = [s[1:-1] for s in strings]

    return strings

# Funktion zum Filtern und Spellchecken der Wörter
def filter_and_spellcheck_words(words):
    misspelled_words = []
    for word in words:
        # Prüfe, ob das Wort aus Buchstaben besteht und nicht komplett groß ist
        if word.isalpha() and not word.isupper() and word not in whitelisted:
            # Prüfe, ob das Wort falsch geschrieben ist
            if word in spell.unknown([word]):
                misspelled_words.append(word)
    return misspelled_words

# Hauptfunktion
def main():
    # Argumente verarbeiten
    parser = argparse.ArgumentParser(description='Extrahiere Strings aus Bash-Dateien und zeige gefilterte Wörter an.')
    parser.add_argument('bash_files', nargs='+', help='Pfad(e) zu den Bash-Dateien')
    args = parser.parse_args()

    # Fortschrittsanzeige mit rich
    with Progress() as progress:
        task = progress.add_task("[green]Verarbeite Dateien...", total=len(args.bash_files))

        # Tabelle für die Ergebnisse
        table = Table(title="Falsch geschriebene Wörter", show_lines=True)
        table.add_column("Datei", justify="center", style="cyan", no_wrap=True)
        table.add_column("Wörter", justify="left", style="magenta")

        # Verarbeite jede Datei
        for bash_file in args.bash_files:
            progress.advance(task)

            if not os.path.exists(bash_file):
                console.print(f"[bold red]Warnung: Datei '{bash_file}' existiert nicht![/bold red]")
                continue

            # Strings extrahieren
            strings = extract_strings_from_bash(bash_file)

            # Splitte die Strings nach Leerzeichen, Komma, Semikolon usw.
            words = []
            for string in strings:
                words.extend(re.split(r'[ ,;]', string))

            # Filtere und checke die Wörter auf Rechtschreibung
            misspelled_words = filter_and_spellcheck_words(words)
            misspelled_words = list(set(misspelled_words))

            # Falls es falsch geschriebene Wörter gibt, füge sie zur Tabelle hinzu
            if misspelled_words:
                table.add_row(bash_file, ", ".join(misspelled_words))

        # Zeige Tabelle an
        console.print(table)

if __name__ == "__main__":
    main()


import ast
import difflib
import argparse
from rich.console import Console
from rich.progress import Progress
import signal
import sys

class FunctionCollector(ast.NodeVisitor):
    def __init__(self, min_lines):
        self.functions = {}
        self.min_lines = min_lines

    def visit_FunctionDef(self, node):
        # Funktion überspringen, wenn sie weniger als die Mindestanzahl an Zeilen hat
        func_code = ast.unparse(node)
        if len(func_code.splitlines()) >= self.min_lines:
            func_code_structure = ast.dump(node, annotate_fields=False)
            self.functions[node.name] = func_code_structure
        self.generic_visit(node)

def find_similar_functions(filename, threshold, min_lines):
    # Erstellen der Konsole und der Progress-Bar
    console = Console()
    with Progress(transient=True) as progress:
        task = progress.add_task("[cyan]Verarbeite Datei...", total=1)
        
        # Laden des Quellcodes
        with open(filename, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Fortschrittsbalken aktualisieren
        progress.update(task, description="Parsen des Quellcodes...")
        tree = ast.parse(source_code)
        collector = FunctionCollector(min_lines)
        collector.visit(tree)

        funcs = list(collector.functions.items())
        total_comparisons = len(funcs) * (len(funcs) - 1) // 2  # Kombinierte Vergleiche
        current_comparisons = 0

        progress.update(task, description="Vergleiche Funktionen...", total=total_comparisons)
        
        # Vergleichen der Funktionen
        for i in range(len(funcs)):
            name1, code1 = funcs[i]
            for j in range(i + 1, len(funcs)):
                name2, code2 = funcs[j]
                
                # Zeige die aktuelle Funktion im Fortschrittsbalken an
                progress.update(task, description=f"[yellow]Vergleiche:[/yellow] {name1} <-> {name2}")
                
                similarity = difflib.SequenceMatcher(None, code1, code2).ratio()
                if similarity >= threshold:
                    console.print(f"[green]Ähnliche Funktionen:[/green] {name1} <-> {name2} ({similarity*100:.2f}%)")

                current_comparisons += 1
                progress.update(task, completed=current_comparisons)

        progress.update(task, completed=total_comparisons)

# Funktion zum sicheren Beenden des Programms bei Ctrl+C
def handle_interrupt(signal, frame):
    console = Console()
    console.print("\n[red]Programm wurde durch den Benutzer gestoppt.[/red]", end="")
    sys.exit(0)

if __name__ == "__main__":
    # Signal für Ctrl+C (KeyboardInterrupt) abfangen
    signal.signal(signal.SIGINT, handle_interrupt)

    parser = argparse.ArgumentParser(description="Findet ähnliche Funktionen in einer Python-Datei.")
    parser.add_argument("file", help="Pfad zur Python-Datei")
    parser.add_argument("--threshold", type=float, default=0.8, help="Ähnlichkeitsschwelle (0.0 - 1.0)")
    parser.add_argument("--min-lines", type=int, default=3, help="Mindestanzahl an Zeilen für Funktionen, die untersucht werden (Standard: 3)")

    args = parser.parse_args()
    find_similar_functions(args.file, args.threshold, args.min_lines)

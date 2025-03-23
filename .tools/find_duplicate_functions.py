import signal
import sys
import ast
import difflib
import argparse
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

max_compare_these_length = 0

class FunctionCollector(ast.NodeVisitor):
    def __init__(self, min_lines):
        self.functions = {}
        self.min_lines = min_lines

    def visit_FunctionDef(self, node):
        # Skip the function if it has fewer than the minimum number of lines
        func_code = ast.unparse(node)
        if len(func_code.splitlines()) >= self.min_lines:
            func_code_structure = ast.dump(node, annotate_fields=False)
            self.functions[node.name] = (func_code_structure, len(func_code.splitlines()))
        self.generic_visit(node)

def find_similar_functions(filename, threshold, min_lines):
    # Create the console and progress bar
    console = Console()
    with Progress(transient=True) as progress:
        task = progress.add_task("[cyan]Processing file...", total=1)

        # Load the source code
        with open(filename, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Update progress bar
        progress.update(task, description="Parsing the source code...")
        tree = ast.parse(source_code)
        collector = FunctionCollector(min_lines)
        collector.visit(tree)

        funcs = list(collector.functions.items())
        total_comparisons = len(funcs) * (len(funcs) - 1) // 2  # Total comparisons
        current_comparisons = 0

        progress.update(task, description="Comparing functions...", total=total_comparisons)

        similar_functions = []  # List to hold similar function pairs

        # Compare the functions
        for i in range(len(funcs)):
            name1, (code1, _) = funcs[i]
            for j in range(i + 1, len(funcs)):
                name2, (code2, _) = funcs[j]

                # Update progress bar for current comparison
                global max_compare_these_length

                compare_these = f"{name1} <-> {name2}"
                if len(similar_functions):
                    if len(similar_functions) == 1:
                        compare_these = f"{compare_these} ({len(similar_functions)} similar function)"
                    else:
                        compare_these = f"{compare_these} ({len(similar_functions)} similar functions)"
                compare_these_length = len(compare_these)
                max_compare_these_length = max(max_compare_these_length, compare_these_length)

                compare_these = compare_these.ljust(max_compare_these_length)

                progress.update(task, description=f"[yellow]Comparing:[/yellow] {compare_these}")

                similarity = difflib.SequenceMatcher(None, code1, code2).ratio()
                if similarity >= threshold:
                    similar_functions.append([name1, name2, f"{similarity*100:.2f}%", len(code1.splitlines()), len(code2.splitlines())])

                current_comparisons += 1
                progress.update(task, completed=current_comparisons)

        # Update progress bar completion
        progress.update(task, completed=total_comparisons)

        # Display the results as a rich table
        if similar_functions:
            console.print("\n[green]Similar functions found:[/green]")
            table = Table(title="Function Similarities")
            table.add_column("Function 1", justify="left")
            table.add_column("Lines 1", justify="right")
            table.add_column("Function 2", justify="left")
            table.add_column("Lines 2", justify="right")
            table.add_column("Similarity", justify="right")

            for row in similar_functions:
                table.add_row(row[0], str(row[3]), row[1], str(row[4]), row[2])

            console.print(table)
        else:
            console.print("[yellow]No similar functions found above the threshold.[/yellow]")

# Function to handle safe program termination on Ctrl+C
def handle_interrupt(_signal, _frame):
    console = Console()
    console.print("\n[red]Program was interrupted by the user.[/red]", end="")
    sys.exit(0)

if __name__ == "__main__":
    # Handle Ctrl+C signal (KeyboardInterrupt)
    signal.signal(signal.SIGINT, handle_interrupt)

    parser = argparse.ArgumentParser(description="Finds similar functions in a Python file.")
    parser.add_argument("file", help="Path to the Python file")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold (0.0 - 1.0)")
    parser.add_argument("--min-lines", type=int, default=3, help="Minimum number of lines for functions to be analyzed (default: 3)")

    args = parser.parse_args()
    find_similar_functions(args.file, args.threshold, args.min_lines)

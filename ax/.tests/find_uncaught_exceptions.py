import ast
import os
import sys
import traceback
import importlib
import inspect
from typing import List
from rich.console import Console
from rich.table import Table
import argparse

console = Console()

class ExceptionTracker:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.imports = {}
        self.exceptions = []
        self.parsed_code = self._parse_file()

    def _parse_file(self):
        """ Parse the given Python file and return the AST. """
        with open(self.file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return ast.parse(file_content)

    def check_try_except(self, node):
        """ Check for try-except blocks and log any unhandled exceptions. """
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                # Capture the exceptions in the except block
                if handler.type:
                    exception_type = handler.type.id if isinstance(handler.type, ast.Name) else str(handler.type)
                    self.imports[exception_type] = True

    def find_raise_statements(self, node):
        """ Find and track any raise statements in the code. """
        if isinstance(node, ast.Raise):
            exc_type = node.exc
            if exc_type:
                exception_type = str(exc_type)
                self.exceptions.append((node.lineno, exception_type))

    def track_imports_and_functions(self, node):
        """ Track imported modules and functions used in the script. """
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.imports[alias.name] = False
        elif isinstance(node, ast.ImportFrom):
            self.imports[node.module] = False

    def check_code(self):
        """ Traverse the AST to check for unhandled exceptions. """
        for node in ast.walk(self.parsed_code):
            self.check_try_except(node)
            self.find_raise_statements(node)
            self.track_imports_and_functions(node)

    def resolve_imports(self):
        """ Resolve import modules and check their functions for raise statements. """
        for module_name in self.imports.keys():
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module):
                    if callable(obj):
                        self.check_for_raises_in_function(obj)
            except ModuleNotFoundError:
                continue

    def check_for_raises_in_function(self, func):
        """ Check for raise statements within the function from imported modules. """
        try:
            func_code = inspect.getsource(func)
            parsed_func_code = ast.parse(func_code)
            for node in ast.walk(parsed_func_code):
                self.find_raise_statements(node)
        except Exception as e:
            console.print(f"Error inspecting function {func.__name__}: {str(e)}")

    def summarize_results(self):
        """ Summarize and output the results. """
        table = Table(title="Unhandled Exceptions Summary")
        table.add_column("Line", style="cyan")
        table.add_column("Exception Type", style="magenta")
        table.add_column("Description", style="green")

        # Add all raised exceptions in main code
        for line, exception_type in self.exceptions:
            table.add_row(str(line), exception_type, f"Exception '{exception_type}' not caught")

        # Add all imports and check for exceptions they might raise
        self.resolve_imports()

        # Print the table with the results
        console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Analyze Python script for unhandled exceptions.")
    parser.add_argument("file", type=str, help="The path to the Python script to analyze.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File {args.file} does not exist!")
        sys.exit(1)

    tracker = ExceptionTracker(args.file)
    tracker.check_code()
    tracker.summarize_results()

if __name__ == "__main__":
    main()


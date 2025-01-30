import ast
import argparse
from rich.console import Console
from rich.table import Table

class ExceptionAnalyzer(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.try_blocks = []
        self.imports = []
        self.console = Console()
        self.unhandled_exceptions = []

    def analyze(self):
        with open(self.filename, 'r') as f:
            tree = ast.parse(f.read())
            self.visit(tree)
        return self.unhandled_exceptions

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_Try(self, node):
        for handler in node.handlers:
            exc_types = handler.type
            if isinstance(exc_types, ast.Name):
                print(f"Found exception: {exc_types.id} at line {node.lineno}")
                self.try_blocks.append((exc_types.id, node.lineno))
            elif isinstance(exc_types, ast.Tuple):
                for exc_type in exc_types.elts:
                    if isinstance(exc_type, ast.Name):
                        print(f"Found exception: {exc_type.id} at line {node.lineno}")
                        self.try_blocks.append((exc_type.id, node.lineno))
            elif isinstance(exc_types, ast.Attribute):
                module_name = exc_types.value.id if isinstance(exc_types.value, ast.Name) else None
                attr_name = exc_types.attr
                if module_name:
                    exc_type_name = f"{module_name}.{attr_name}"
                else:
                    exc_type_name = f"{attr_name}"
                print(f"Found exception: {exc_type_name} at line {node.lineno}")
                self.try_blocks.append((exc_type_name, node.lineno))
            else:
                print(f"Unexpected exception type found: {exc_types} at line {node.lineno}")
        self.generic_visit(node)

    def report_unhandled_exceptions(self):
        table = Table(title="Unhandled Exceptions")
        table.add_column("Line", justify="right", style="cyan", no_wrap=True)
        table.add_column("Exception Type", style="magenta")
        table.add_column("Description", justify="center", style="dim")
        
        # Debugging-Ausgabe der erfassten Ausnahmen
        print(f"Total uncaught exceptions found: {len(self.unhandled_exceptions)}")
        
        for exc in self.unhandled_exceptions:
            table.add_row(str(exc[0]), exc[1], exc[2])
        
        self.console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Analyze Python script for uncaught exceptions")
    parser.add_argument("filename", type=str, help="Path to the Python file to analyze")
    args = parser.parse_args()

    analyzer = ExceptionAnalyzer(args.filename)
    uncaught_exceptions = analyzer.analyze()
    analyzer.report_unhandled_exceptions()

if __name__ == "__main__":
    main()


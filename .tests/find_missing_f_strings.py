#!/usr/bin/env python3

import argparse
import ast
import os
import re
import sys

def collect_python_files(path):
    files = []
    if os.path.isfile(path) and path.endswith(".py"):
        files.append(os.path.abspath(path))
    elif os.path.isdir(path):
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".py"):
                    files.append(os.path.join(root, filename))
    return files

def extract_defined_names_with_lines(tree):
    """
    Extrahiert alle Variablennamen mit der Zeilennummer, in der sie definiert wurden.
    """
    names = []  # Liste von (name, lineno)

    class NameCollector(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append((target.id, node.lineno))
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.append((elt.id, node.lineno))
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            names.append((node.name, node.lineno))
            for arg in node.args.args:
                names.append((arg.arg, node.lineno))
            if node.args.vararg:
                names.append((node.args.vararg.arg, node.lineno))
            if node.args.kwarg:
                names.append((node.args.kwarg.arg, node.lineno))
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            names.append((node.name, node.lineno))
            self.generic_visit(node)

    NameCollector().visit(tree)
    return names  # Liste aller (name, lineno)

def find_suspicious_strings(tree, filename, source_lines):
    suspicious = []
    string_pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

    defined_names_with_lines = extract_defined_names_with_lines(tree)

    class StringChecker(ast.NodeVisitor):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                matches = string_pattern.findall(node.value)
                # Filtere nur Namen, die vor dieser Zeile definiert sind
                defined_before = {name for name, lineno in defined_names_with_lines if lineno < getattr(node, "lineno", 0)}
                for match in matches:
                    if match in defined_before:
                        lineno = getattr(node, "lineno", None)
                        line = source_lines[lineno - 1].rstrip("\n") if lineno else node.value
                        suspicious.append((filename, lineno, match, line))
            self.generic_visit(node)

    StringChecker().visit(tree)
    return suspicious

def process_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError) as e:
        sys.stderr.write(f"Fehler beim Lesen von {filename}: {e}\n")
        return []

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        sys.stderr.write(f"Syntaxfehler in {filename}: {e}\n")
        return []

    source_lines = source.splitlines()
    return find_suspicious_strings(tree, filename, source_lines)

def main():
    parser = argparse.ArgumentParser(
        description="Finds string literals like 'bla {var}' without f-Strings."
    )
    parser.add_argument("paths", nargs="*", help="Path to one or many .py-files or a folder")
    args = parser.parse_args()

    input_paths = args.paths if args.paths else ["."]

    python_files = []
    for path in input_paths:
        python_files.extend(collect_python_files(path))

    if not python_files:
        sys.stderr.write("No python files found.\n")
        sys.exit(1)

    all_suspicious = []
    for file in python_files:
        all_suspicious.extend(process_file(file))

    if all_suspicious:
        print("Possible forgotten f's:\n")
        for filename, lineno, var, line in all_suspicious:
            print(f"{filename}:{lineno}: Variable '{var}' detected in file -> {line}")

        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()

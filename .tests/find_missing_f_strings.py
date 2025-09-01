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
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".py"):
                    files.append(os.path.join(root, filename))
    return files

def extract_defined_names(tree):
    names = set()

    class NameCollector(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            names.add(node.name)
            for arg in node.args.args:
                names.add(arg.arg)
            if node.args.vararg:
                names.add(node.args.vararg.arg)
            if node.args.kwarg:
                names.add(node.args.kwarg.arg)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            names.add(node.name)
            self.generic_visit(node)

    NameCollector().visit(tree)
    return names

def find_suspicious_strings(tree, defined_names, filename, source_lines):
    suspicious = []
    string_pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

    class StringChecker(ast.NodeVisitor):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                matches = string_pattern.findall(node.value)
                for match in matches:
                    if match in defined_names:
                        lineno = getattr(node, "lineno", None)
                        if lineno is not None and 1 <= lineno <= len(source_lines):
                            line = source_lines[lineno - 1].rstrip("\n")
                        else:
                            line = node.value
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

    defined_names = extract_defined_names(tree)
    source_lines = source.splitlines()
    return find_suspicious_strings(tree, defined_names, filename, source_lines)

def main():
    parser = argparse.ArgumentParser(
        description="Findet String-Literale wie 'bla {var}' ohne f-String-Markierung."
    )
    parser.add_argument("path", help="Pfad zu einer .py-Datei oder einem Ordner")
    args = parser.parse_args()

    python_files = collect_python_files(args.path)
    if not python_files:
        sys.stderr.write("Keine Python-Dateien gefunden.\n")
        sys.exit(1)

    all_suspicious = []
    for file in python_files:
        all_suspicious.extend(process_file(file))

    if all_suspicious:
        print("Mögliche vergessene f-Strings gefunden:\n")
        for filename, lineno, var, line in all_suspicious:
            print(f"{filename}:{lineno}: Variable '{var}' in String entdeckt -> {line}")
    else:
        print("Keine verdächtigen Strings gefunden.")

if __name__ == "__main__":
    main()

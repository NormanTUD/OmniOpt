import ast
import os
import importlib.util
import sys
from collections import defaultdict

def find_raised_exceptions(node):
    """Durchsucht den AST nach `raise`-Statements und extrahiert die Exception-Typen."""
    exceptions = set()
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Raise) and stmt.exc:
            if isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name):
                exceptions.add(stmt.exc.func.id)
            elif isinstance(stmt.exc, ast.Name):
                exceptions.add(stmt.exc.id)
    return exceptions

def find_caught_exceptions(node):
    """Sammelt alle Exception-Typen, die in `except`-Blöcken abgefangen werden."""
    caught_exceptions = set()
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.ExceptHandler) and stmt.type:
            if isinstance(stmt.type, ast.Name):
                caught_exceptions.add(stmt.type.id)
            elif isinstance(stmt.type, ast.Tuple):
                for exc in stmt.type.elts:
                    if isinstance(exc, ast.Name):
                        caught_exceptions.add(exc.id)
    return caught_exceptions

def analyze_script(filename):
    """Analysiert eine Python-Datei und gibt die nicht abgefangenen Exceptions aus."""
    with open(filename, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename)
    
    raised_exceptions = find_raised_exceptions(tree)
    caught_exceptions = find_caught_exceptions(tree)
    
    uncaught_exceptions = raised_exceptions - caught_exceptions
    return raised_exceptions, caught_exceptions, uncaught_exceptions

def find_imported_modules(node):
    """Findet alle importierten Module in einem Python-Skript."""
    modules = set()
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                modules.add(alias.name)
        elif isinstance(stmt, ast.ImportFrom) and stmt.module:
            modules.add(stmt.module)
    return modules

def locate_module(module_name):
    """Versucht, den Speicherort eines Moduls zu finden."""
    spec = importlib.util.find_spec(module_name)
    if spec and spec.origin and spec.origin.endswith(".py"):
        return spec.origin
    return None

def analyze_script_and_modules(filename):
    """Analysiert eine Python-Datei und ihre importierten Module auf Exception-Handling."""
    with open(filename, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename)
    
    imported_modules = find_imported_modules(tree)
    all_exceptions = defaultdict(set)
    
    # Analysiere Hauptskript
    raised, caught, uncaught = analyze_script(filename)
    all_exceptions[filename] = uncaught
    
    # Analysiere importierte Module
    for module in imported_modules:
        module_path = locate_module(module)
        if module_path and os.path.isfile(module_path):
            mod_raised, mod_caught, mod_uncaught = analyze_script(module_path)
            all_exceptions[module_path] = mod_uncaught
    
    return all_exceptions

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_exceptions.py <script.py>")
        sys.exit(1)
    
    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    
    results = analyze_script_and_modules(filename)
    
    for script, uncaught_exceptions in results.items():
        print(f"In Datei: {script}")
        if uncaught_exceptions:
            print("  ⚠️ Nicht abgefangene Exceptions:")
            for exc in uncaught_exceptions:
                print(f"    - {exc}")
        else:
            print("  ✅ Alle Exceptions werden abgefangen.")

if __name__ == "__main__":
    main()

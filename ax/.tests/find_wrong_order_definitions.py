#!/usr/bin/env python3

import sys
import os
import ast
from pprint import pprint

def dier(msg):
    pprint(msg)
    sys.exit(10)

class UndefinedVariableChecker(ast.NodeVisitor):
    def __init__(self):
        self.defined_vars = set()
        self.errors = []
        self.built_in_vars = set(dir(__builtins__))
        self.special_vars = {'__file__', '__name__', '__doc__', '__builtins__'}

    def visit_Import(self, node):
        for alias in node.names:
            self.defined_vars.add(alias.asname if alias.asname else alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.defined_vars.add(alias.asname if alias.asname else alias.name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Tuple):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.defined_vars.add(target.id)
            else:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.defined_vars.add(target.id)
        elif isinstance(node.targets[0], ast.Subscript):
            if isinstance(node.targets[0].value, ast.Name):
                self.defined_vars.add(node.targets[0].value.id)
        self.generic_visit(node)

    # Hier wird die neue Methode hinzugefügt
    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.defined_vars.add(node.target.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Normale Argumente hinzufügen
        for arg in node.args.args:
            self.defined_vars.add(arg.arg)
        # *args hinzufügen
        if node.args.vararg:
            self.defined_vars.add(node.args.vararg.arg)
        # **kwargs hinzufügen
        if node.args.kwarg:
            self.defined_vars.add(node.args.kwarg.arg)

        # Auch den Funktionsnamen als definiert markieren
        self.defined_vars.add(node.name)
        current_defined_vars = self.defined_vars.copy()
        self.generic_visit(node)
        self.defined_vars = current_defined_vars

    def visit_ClassDef(self, node):
        self.defined_vars.add(node.name)
        current_defined_vars = self.defined_vars.copy()
        self.generic_visit(node)
        self.defined_vars = current_defined_vars

    def visit_ExceptHandler(self, node):
        if node.name:
            self.defined_vars.add(node.name)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if isinstance(item.optional_vars, ast.Name):
                self.defined_vars.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_For(self, node):
        if isinstance(node.target, ast.Name):
            self.defined_vars.add(node.target.id)
        current_defined_vars = self.defined_vars.copy()
        self.generic_visit(node)
        self.defined_vars = current_defined_vars

    def visit_ListComp(self, node):
        self.process_comprehension_vars(node.generators, node.elt)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        # Verarbeite die Variablen in den Generatoren (z. B. k, v in for k, v in ...)
        for gen in node.generators:
            if isinstance(gen.target, (ast.Name, ast.Tuple)):
                if isinstance(gen.target, ast.Name):
                    self.defined_vars.add(gen.target.id)
                elif isinstance(gen.target, ast.Tuple):
                    for elt in gen.target.elts:
                        if isinstance(elt, ast.Name):
                            self.defined_vars.add(elt.id)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.process_comprehension_vars(node.generators, node.elt)
        self.generic_visit(node)

    def process_comprehension_vars(self, generators, *nodes):
        for gen in generators:
            if isinstance(gen.target, ast.Name):
                self.defined_vars.add(gen.target.id)
            if isinstance(gen.iter, ast.Name):
                self.defined_vars.add(gen.iter.id)
            self.generic_visit(gen.iter)

        for node in nodes:
            if isinstance(node, ast.Name):
                self.defined_vars.add(node.id)
            elif isinstance(node, (ast.Call, ast.Attribute)):
                self.generic_visit(node)

    def visit_Lambda(self, node):
        for arg in node.args.args:
            self.defined_vars.add(arg.arg)
        current_defined_vars = self.defined_vars.copy()
        self.generic_visit(node)
        self.defined_vars = current_defined_vars

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self.defined_vars:
            if node.id not in self.built_in_vars and node.id not in self.special_vars:
                self.errors.append(f"Variable '{node.id}' used before assignment at line {node.lineno}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.special_vars:
            self.defined_vars.add(node.value.id)
        self.generic_visit(node)

    def visit_Tuple(self, node):
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                self.defined_vars.add(elt.id)
        self.generic_visit(node)

    def visit_List(self, node):
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                self.defined_vars.add(elt.id)
        self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                self.defined_vars.add(node.value.func.id)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name):
            self.defined_vars.add(node.value.id)
        self.generic_visit(node)

if len(sys.argv) > 1:
    if os.path.exists(sys.argv[1]):
        with open(sys.argv[1], mode='r', encoding="utf-8") as file:
            code = file.read()
    else:
        print(f"File '{sys.argv[1]}' does not exist")
        sys.exit(1)
else:
    print("Not enough parameters")
    sys.exit(1)

tree = ast.parse(code)
checker = UndefinedVariableChecker()
checker.visit(tree)

for error in checker.errors:
    print(error)

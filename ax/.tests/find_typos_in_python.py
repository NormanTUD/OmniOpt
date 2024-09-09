import sys
import ast
import argparse
import re
from spellchecker import SpellChecker
from pprint import pprint

def dier (msg):
    pprint(msg)
    sys.exit(10)


# Initialize spellchecker with English dictionary
spell = SpellChecker(language='en')

# Regex patterns to ignore specific cases (you can modify these if needed)
IGNORE_PATTERNS = [
    r'^\d+$',
    r'^runtime$',
    r'^Hostname$',
    r'^coolwarm$',
    r'^DataFrame$',
    r'^darkred$',
    r'^lightcoral$',
    r'^palegreen$',
    r'^darkgreen$',
    r'^min$',
    r'^max$',
    r'^Gridsize$',
    r'^Params$',
    r'^Min$',
    r'^Max$',
    r'^darktheme$',
    r'^TkAgg$',
    r'^dataset$',
    r'^subscriptable$',
    r'^csv$',
    r'^CSV$',
    r'^Timestamp$',
    r'^timestamp$',
    r'^pstate$',
    r'^Num$',
    r'^num$',
    r'^dir$',
    r'^botorch$',
    r'^venv$',
    r'^seaborn$',
    r'^psutil$',
    r'^numpy$',
    r'^matplotlib$',
    r'^tqdm$',
    r'^submitit$',
    r'^hostname$',
    r'^[A-Z]{2,}$',
    r'^[a-z]{1,2}$'
]

def is_ignored(word):
    """Check if the word should be ignored based on defined regex patterns."""
    for pattern in IGNORE_PATTERNS:
        if re.match(pattern, word):
            return True
    return False

def is_valid_word(word):
    """Check if the word contains only alphanumeric characters (ignores anything with special characters)."""
    return re.match(r'^[a-zA-Z]{3,}$', word) is not None

def extract_strings_from_ast(node):
    """Extract all string literals from the AST."""
    if isinstance(node, ast.Str):
        return [node.s]
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):  # For Python 3.8+
        return [node.value]
    elif isinstance(node, (ast.List, ast.Tuple)):
        strings = []
        for element in node.elts:
            strings.extend(extract_strings_from_ast(element))
        return strings
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # For concatenated strings like "Hello " + "World"
        return extract_strings_from_ast(node.left) + extract_strings_from_ast(node.right)
    return []

def analyze_file(filepath):
    """Analyze a Python file and check the spelling of string literals."""
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    tree = ast.parse(content)
    strings = []

    # Traverse the AST to extract strings
    for node in ast.walk(tree):
        strings.extend(extract_strings_from_ast(node))

    # Process the strings
    incorrect_words = []
    for string in strings:
        # Split the string by spaces
        words = string.split()
        for word in words:
            # Only consider words that contain alphanumeric characters and match the regex
            if is_valid_word(word) and not is_ignored(word):
                # Check the spelling
                if spell.correction(word) != word:
                    incorrect_words.append(word)

    return incorrect_words

def main():
    parser = argparse.ArgumentParser(description='Analyze Python scripts and check the spelling of string literals.')
    parser.add_argument('files', metavar='FILE', nargs='+', help='The Python files to analyze.')
    args = parser.parse_args()

    # Process each file
    for filepath in args.files:
        print(f'Analyzing file: {filepath}')
        incorrect_words = analyze_file(filepath)
        if incorrect_words:
            print(f'Unknown or misspelled words in {filepath}: {incorrect_words}')
        else:
            print(f'No misspelled words found in {filepath}.')

if __name__ == '__main__':
    main()


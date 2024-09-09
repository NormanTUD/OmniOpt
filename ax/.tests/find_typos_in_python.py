import sys
import ast
import argparse
import re
from pprint import pprint
from spellchecker import SpellChecker
from rich.progress import Progress
from rich.console import Console

def dier(msg):
    pprint(msg)
    sys.exit(10)

# Initialize spellchecker with English dictionary
spell = SpellChecker(language='en')

# Regex patterns to ignore specific cases
IGNORE_PATTERNS = [
    r'^\d+$',
    r'^runtime$',
    r'^Hostname$',
    r'^coolwarm$',
    r'^DataFrame$',
    r'^darkred$',
    r'^php$',
    r'^Norman$',
    r'^Koch$',
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
    r'^[a-z]{1,2}$',
    r'^anonymized$',
    r'^argparse$',
    r'^AssertionError$',
    r'^AttributeError$',
    r'^bla$',
    r'^chmod$',
    r'^ChoiceParameter$',
    r'^comparision$',
    r'^CPUs$',
    r'^Cuda$',
    r'^def$',
    r'^dict$',
    r'^diff$',
    r'^dpi$',
    r'^Ein$',
    r'^EOFError$',
    r'^ExcludeNode$',
    r'^ExcludeNodeAndRestartAll$',
    r'^filename$',
    r'^ghostbusters$',
    r'^gpus$',
    r'^GPUs$',
    r'^gridsearch$',
    r'^Gridsearch$',
    r'^Hitchcock$',
    r'^Hyperparam$',
    r'^hyperparameter$',
    r'^Hyperparameter$',
    r'^hyperparameters$',
    r'^Hyperparameters$',
    r'^ImportError$',
    r'^IndentationError$',
    r'^IndexError$',
    r'^int$',
    r'^intendation$',
    r'^ist$',
    r'^KeyboardInterrupt$',
    r'^KeyError$',
    r'^Logfile$',
    r'^MemoryError$',
    r'^ModuleNotFoundError$',
    r'^Montana$',
    r'^multiline$',
    r'^NameError$',
    r'^noir$',
    r'^NotImplementedError$',
    r'^ntasks$',
    r'^omniopt$',
    r'^OmniOpt$',
    r'^OSError$',
    r'^outfile$',
    r'^OverflowError$',
    r'^ParameterType$',
    r'^params$',
    r'^prev$',
    r'^QOSMinGRES$',
    r'^quickfix$',
    r'^RangeParameter$',
    r'^RecursionError$',
    r'^ReferenceError$',
    r'^res$',
    r'^RestartOnDifferentNode$',
    r'^RuntimeError$',
    r'^sbatch$',
    r'^sixel$',
    r'^slurm$',
    r'^Slurm$',
    r'^slurmbased$',
    r'^Sobol$',
    r'^srun$',
    r'^stderr$',
    r'^stdout$',
    r'^stimpy$',
    r'^subjobs$',
    r'^SyntaxError$',
    r'^SystemError$',
    r'^TabError$',
    r'^Taurus$',
    r'^trex$',
    r'^TypeError$',
    r'^unicode$',
    r'^UnicodeError$',
    r'^Unspported$',
    r'^ValueError$',
    r'^xxx$',
    r'^ZeroDivisionError$',
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
    if isinstance(node, ast.Constant) and isinstance(node.value, str):  # For Python 3.8+
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple)):
        strings = []
        for element in node.elts:
            strings.extend(extract_strings_from_ast(element))
        return strings
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return extract_strings_from_ast(node.left) + extract_strings_from_ast(node.right)
    return []

def analyze_file(filepath, progress, task_id):
    """Analyze a Python file and check the spelling of string literals."""
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    tree = ast.parse(content)
    strings = []

    # Traverse the AST to extract strings
    for node in ast.walk(tree):
        strings.extend(extract_strings_from_ast(node))

    # Update the total number of string literals in the progress bar
    progress.update(task_id, total=len(strings))

    # Process the strings
    incorrect_words = []
    strings = list(set(strings))
    for i, string in enumerate(strings):
        words = string.split()
        for word in words:
            if is_valid_word(word) and not is_ignored(word):
                if spell.correction(word) != word:
                    incorrect_words.append(word)

        # Update the progress bar as each string is processed
        progress.advance(task_id)

    return incorrect_words

def main():
    parser = argparse.ArgumentParser(description='Analyze Python scripts and check the spelling of string literals.')
    parser.add_argument('files', metavar='FILE', nargs='+', help='The Python files to analyze.')
    args = parser.parse_args()

    console = Console()
    typo_files = 0

    # Progress bar setup with Rich
    with Progress(console=console, transient=True, auto_refresh=True) as progress:
        for filepath in args.files:
            # Each progress bar disappears once 100% complete
            task_id = progress.add_task(f"[cyan]Analyzing {filepath}", total=1)

            # Analyze the file and show real-time progress
            incorrect_words = analyze_file(filepath, progress, task_id)
            if incorrect_words:
                typo_files += 1
                console.print(f"[red]Unknown or misspelled words in {filepath}: {incorrect_words}")
            else:
                console.print(f"[green]No misspelled words found in {filepath}")

    sys.exit(typo_files)

if __name__ == '__main__':
    main()

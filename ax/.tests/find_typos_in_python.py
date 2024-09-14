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
    'anonymized',
    'argparse',
    'AssertionError',
    'AttributeError',
    'binbash',
    'binsh',
    'bla',
    'botorch',
    'BUBBLESIZEINPX',
    'chmod',
    'ChoiceParameter',
    'CI',
    'CLI',
    'CONT',
    'coolwarm',
    'CPU',
    'CPUs',
    'CSV',
    'CTRL',
    'CTRLC',
    'CUDA',
    'darkgreen',
    'darkred',
    'darktheme',
    'DataFrame',
    'dataset',
    'def',
    'dict',
    'didnt',
    'Diff',
    'dir',
    'DJ',
    'dont',
    'dpi',
    'memoryfree',
    'HMSf',
    'pcielinkgencurrent',
    'memorytotal',
    'memoryused',
    'utilizationgpu',
    'utilizationmemory',
    'env',
    'resultscsv',
    'pcielinkgenmax',
    'temperaturegpu',
    'subprocess',
    'stat',
    'axservice',
    'omnioptpy',
    'axserviceutils',
    'botorchoptimfit',
    'torchautograd',
    'axcoreexperiment',
    'axmodelbridgetransforms',
    'headerscsv',
    'EOFError',
    'axserviceutilsinstantiation',
    'png',
    'df',
    'shutil',
    'plotpy',
    'botorchmodelsutilsassorted',
    'os',
    'axcoredata',
    'axmodelbridgeaxmodelbridgebase',
    'botorchoptimoptimize',
    'axcoreparameter',
    'richtable',
    'yaml',
    'axmodelbridgetorch',
    'parameterscsv',
    'ie',
    'difflib',
    'pformat',
    'VALUEVALUEVALUE',
    'PIL',
    'richpretty',
    'itertools',
    'json',
    'etcpasswd',
    'evaluatex',
    'ExcludeNode',
    'ExcludeNodeAndRestartAll',
    'expectedsclass',
    'filename',
    'FULLYBAYESIAN',
    'Gandalf',
    'GB',
    'ghostbusters',
    'GPEI',
    'GPU',
    'gpus',
    'GPUs',
    'greenLoading',
    'Gridsearch',
    'Gridsize',
    'Hitchcock',
    'HMS',
    'hostname',
    'Hostname',
    'HPC',
    'Hyperparam',
    'Hyperparameter',
    'hyperparameters',
    'IDONOTEXIST',
    'ImportError',
    'IndentationError',
    'IndexError',
    'INFOS',
    'int',
    'intendation',
    'isnt',
    'Jedi',
    'KDE',
    'KeyboardInterrupt',
    'KeyError',
    'lastavg',
    'len',
    'levelnames',
    'lightcoral',
    'Logfile',
    'matplotlib',
    'Max',
    'MemoryError',
    'MiB',
    'min',
    'ModuleNotFoundError',
    'Montana',
    'multiline',
    'NameError',
    'Neo',
    'noheader',
    'noir',
    'NotImplementedError',
    'nr',
    'num',
    'Num',
    'numbervariable',
    'numpy',
    'OK',
    'omniopt',
    'OOM',
    'OSError',
    'outfile',
    'OverflowError',
    'palegreen',
    'param',
    'ParameterType',
    'Params',
    'prev',
    'pstate',
    'psutil',
    'pwd',
    'QOSMinGRES',
    'quickfix',
    'RangeParameter',
    'RecursionError',
    'ReferenceError',
    'res',
    'RestartOnDifferentNode',
    'rg',
    'runtime',
    'RuntimeError',
    'SAASBO',
    'sbatch',
    'seaborn',
    'SIGABRT',
    'SIGALRM',
    'SIGBUS',
    'SIGCHLD',
    'SIGCONT',
    'SIGFPE',
    'SIGHUP',
    'SIGILL',
    'SIGIO',
    'SIGKILL',
    'SIGPIPE',
    'SIGPROF',
    'SIGPWR',
    'SIGQUIT',
    'SIGSEGV',
    'SIGSTKFLT',
    'SIGSTOP',
    'SIGSYS',
    'SIGTERM',
    'SIGTRAP',
    'SIGTSTP',
    'SIGTTIN',
    'SIGTTOU',
    'SIGURG',
    'SIGUSR',
    'SIGVTALRM',
    'SIGWINCH',
    'SIGXCPU',
    'SIGXFSZ',
    'sixel',
    'Slurm',
    'SLURM',
    'slurmbased',
    'Sobol',
    'srun',
    'STDERR',
    'stdout',
    'stimpy',
    'subjobs',
    'submitit',
    'subscriptable',
    'SyntaxError',
    'SystemError',
    'TabError',
    'Taurus',
    'taurusitaurusi',
    'tb',
    'THOMPSON',
    'timestamp',
    'Timestamp',
    'TkAgg',
    'tmp',
    'tqdm',
    'Traceback',
    'TraceBreakpoint',
    'treeslowly',
    'trex',
    'TypeError',
    'UID',
    'unicode',
    'UnicodeError',
    'Unspported',
    'USR',
    'UTF',
    'ValueError',
    'venv',
    'xxx',
    'yellowContinuation',
    'Ymd',
    'Youve',
    'ZeroDivisionError'
]

def is_ignored(word):
    """Check if the word should be ignored based on defined regex patterns."""
    for pattern in IGNORE_PATTERNS:
        if word.lower() == pattern.lower():
            return True
    return False

def is_valid_word(word):
    """Check if the word contains only alphanumeric characters (ignores anything with special characters)."""
    return re.match(r'^[a-zA-Z]{1,}$', word) is not None

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

def clean_word(word):
    # Entfernt alle nicht-alphabetischen Zeichen und behält nur "a-zA-Z"
    after = re.sub(r'[^a-zA-Z0-9_/\-]', '', word)

    return after

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
    possibly_incorrect_words = []
    strings = list(set(strings))
    for i, string in enumerate(strings):
        words = string.split()
        for word in words:
            word = clean_word(word)
            if is_valid_word(word):
                if not is_ignored(word):
                    if spell.correction(word) != word:
                        if word not in possibly_incorrect_words:
                            print(f"'{word}',")
                            possibly_incorrect_words.append(word)
            #    else:
            #        print(f"Ignored word: {word}")
            #else:
            #    print(f"Invalid word: {word}")
        # Update the progress bar as each string is processed
        progress.advance(task_id)

    return possibly_incorrect_words

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
            possibly_incorrect_words = analyze_file(filepath, progress, task_id)
            if possibly_incorrect_words:
                typo_files += 1
                console.print(f"[red]Unknown or misspelled words in {filepath}: {possibly_incorrect_words}")

    sys.exit(typo_files)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelled script by using CTRL c")

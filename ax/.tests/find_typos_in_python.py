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
    r'^anonymized$',
    r'^argparse$',
    r'^ASCIIplots$',
    r'^AssertionError$',
    r'^AttributeError$',
    r'^axcoredata$',
    r'^axcoreexperiment$',
    r'^axcoreparameter$',
    r'^axmodelbridgeaxmodelbridgebase$',
    r'^axmodelbridgetorch$',
    r'^axmodelbridgetransforms$',
    r'^axservice$',
    r'^axserviceutils$',
    r'^axserviceutilsinstantiation$',
    r'^[a-z]{1,2}$',
    r'^[A-Z]{2,}$',
    r'^behaviorentry$',
    r'^binbash$',
    r'^binsh$',
    r'^bla$',
    r'^botorch$',
    r'^botorchmodelsutilsassorted$',
    r'^botorchoptimfit$',
    r'^botorchoptimoptimize$',
    r'^bubblesize$',
    r'^checkpointjson$',
    r'^chmod$',
    r'^ChoiceParameter$',
    r'^Commaseparated$',
    r'^comparision$',
    r'^constraintstxt$',
    r'^ControlC$',
    r'^CONTsignal$',
    r'^CONTSignal$',
    r'^coolwarm$',
    r'^CPUs$',
    r'^csv$',
    r'^CSV$',
    r'^CtrlC$',
    r'^Cuda$',
    r'^\d+$',
    r'^darkgreen$',
    r'^darkmode$',
    r'^darkred$',
    r'^darktheme$',
    r'^dataframe$',
    r'^DataFrame$',
    r'^dataset$',
    r'^DebugInfos$',
    r'^def$',
    r'^dev$',
    r'^devgabe$',
    r'^dict$',
    r'^didnt$',
    r'^diff$',
    r'^Diff$',
    r'^difflib$',
    r'^dir$',
    r'^dont$',
    r'^dpi$',
    r'^Ein$',
    r'^endalgorithm$',
    r'^env$',
    r'^EOFError$',
    r'^etcpasswd$',
    r'^evaluatex$',
    r'^ExcludeNode$',
    r'^ExcludeNodeAndRestartAll$',
    r'^exitcode$',
    r'^ExitCode$',
    r'^expectedsclass$',
    r'^filename$',
    r'^finetuning$',
    r'^Finetuning$',
    r'^formatcsv$',
    r'^Gandalf$',
    r'^ghostbusters$',
    r'^gpu$',
    r'^gpus$',
    r'^GPUs$',
    r'^gpussetting$',
    r'^GPUUsage$',
    r'^greenLoading$',
    r'^gridsearch$',
    r'^Gridsearch$',
    r'^gridsize$',
    r'^Gridsize$',
    r'^headerscsv$',
    r'^helperspy$',
    r'^HexScatter$',
    r'^Hitchcock$',
    r'^HMSf$',
    r'^hostname$',
    r'^Hostname$',
    r'^HPCsystems$',
    r'^Hyperparam$',
    r'^hyperparameter$',
    r'^Hyperparameter$',
    r'^hyperparameters$',
    r'^Hyperparameters$',
    r'^Hypersanity$',
    r'^ImportError$',
    r'^Inceptionlevel$',
    r'^IndentationError$',
    r'^IndexError$',
    r'^InputString$',
    r'^int$',
    r'^intendation$',
    r'^INTsignal$',
    r'^INTSignal$',
    r'^isnt$',
    r'^ist$',
    r'^itertools$',
    r'^Jedi$',
    r'^json$',
    r'^KeyboardInterrupt$',
    r'^KeyError$',
    r'^Koch$',
    r'^laserguided$',
    r'^lastavg$',
    r'^len$',
    r'^levelnames$',
    r'^lightcoral$',
    r'^Logfile$',
    r'^mainscript$',
    r'^matplotlib$',
    r'^max$',
    r'^Max$',
    r'^MemoryError$',
    r'^memoryfree$',
    r'^memorytotal$',
    r'^memoryused$',
    r'^MiB$',
    r'^min$',
    r'^Min$',
    r'^ModuleNotFoundError$',
    r'^Montana$',
    r'^multiline$',
    r'^nameentry$',
    r'^NameError$',
    r'^Neo$',
    r'^noheader$',
    r'^noir$',
    r'^nonexisting$',
    r'^nonsbatchsystems$',
    r'^Norman$',
    r'^NotImplementedError$',
    r'^Nr$',
    r'^ntasks$',
    r'^num$',
    r'^Num$',
    r'^numbervariable$',
    r'^numpy$',
    r'^nvidiasmi$',
    r'^omniopt$',
    r'^OmniOpt$',
    r'^omnioptpy$',
    r'^OOrun$',
    r'^OSError$',
    r'^outfile$',
    r'^OverflowError$',
    r'^palegreen$',
    r'^pandasdataframecsv$',
    r'^param$',
    r'^parameterscsv$',
    r'^parameterstxt$',
    r'^ParameterType$',
    r'^params$',
    r'^Params$',
    r'^partitionalpha$',
    r'^pcielinkgencurrent$',
    r'^pcielinkgenmax$',
    r'^pformat$',
    r'^php$',
    r'^plotpy$',
    r'^png$',
    r'^prev$',
    r'^ProgramCode$',
    r'^pstate$',
    r'^psutil$',
    r'^pwd$',
    r'^QOSMinGRES$',
    r'^quickfix$',
    r'^RangeParameter$',
    r'^RecursionError$',
    r'^ReferenceError$',
    r'^res$',
    r'^RestartOnDifferentNode$',
    r'^resultscsv$',
    r'^RESULTstring$',
    r'^richpretty$',
    r'^richtable$',
    r'^rundir$',
    r'^RunDirs$',
    r'^RunProgram$',
    r'^runsh$',
    r'^runtime$',
    r'^RuntimeError$',
    r'^sbatch$',
    r'^seaborn$',
    r'^shutil$',
    r'^SignalCode$',
    r'^sixel$',
    r'^slurm$',
    r'^Slurm$',
    r'^slurmbased$',
    r'^SlurmEnvironment$',
    r'^Sobol$',
    r'^sOOInfoswss$',
    r'^sRESULTsdd$',
    r'^srun$',
    r'^sshconnection$',
    r'^stat$',
    r'^StatusChangeTime$',
    r'^stderr$',
    r'^stdout$',
    r'^stimpy$',
    r'^subjobs$',
    r'^submitit$',
    r'^subprocess$',
    r'^subscriptable$',
    r'^SyntaxError$',
    r'^SystemError$',
    r'^TabError$',
    r'^Taurus$',
    r'^taurusitaurusi$',
    r'^temperaturegpu$',
    r'^textfile$',
    r'^Timesetting$',
    r'^timestamp$',
    r'^Timestamp$',
    r'^TkAgg$',
    r'^tmp$',
    r'^torchautograd$',
    r'^tqdm$',
    r'^Traceback$',
    r'^TraceBreakpoint$',
    r'^treeslowly$',
    r'^trex$',
    r'^TypeError$',
    r'^unicode$',
    r'^UnicodeError$',
    r'^Unspported$',
    r'^USRsignal$',
    r'^USRSignal$',
    r'^utf$',
    r'^utilizationgpu$',
    r'^utilizationmemory$',
    r'^uuid$',
    r'^ValueError$',
    r'^venv$',
    r'^xxx$',
    r'^yaml$',
    r'^yellowContinuation$',
    r'^yellowRunfolderyellow$',
    r'^Ymd$',
    r'^Youve$',
    r'^ZeroDivisionError$',
    r'^zerojobs$',
]

def is_ignored(word):
    """Check if the word should be ignored based on defined regex patterns."""
    for pattern in IGNORE_PATTERNS:
        if re.match(pattern, word):
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
    return re.sub(r'[^a-zA-Z_]', '', word)

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
                        print(f"r'^{word}$',")
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
    main()

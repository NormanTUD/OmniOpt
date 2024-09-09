import os
import sys
import re
import subprocess
from bs4 import BeautifulSoup
from spellchecker import SpellChecker
import emoji

# Liste von Regex-Mustern, die ignoriert werden sollen (z.B. technische Begriffe, Abkürzungen, usw.)
IGNORED_PATTERNS = [
    r'^linux$',
    r'^cd$',
    r'^send_anonymized_usage_stats$',
    r'^elif$',
    r'^eq$',
    r'^fi$',
    r'^learning_rate$',
    r'^learning_ratelearning_rate$',
    r'^num_random_steps20$',
    r'^pathtomy_experimentrunsh$',
    r'^time60$',
    r'^show_sixel_graphics$',
    r'^base64$',
    r'^binbash$',
    r'^epochsepochs$',
    r'^exit_code$',
    r'^max_eval500$',
    r'^num_parallel_jobs20$',
    r'^partitionalpha$',
    r'^run_programecho$',
    r'^runsmy_experiment0$',
    r'^worker_timeout30$',
    r'^cpus_per_task1$',
    r'^experiment_namemy_experiment$',
    r'^gpus0$',
    r'^layerslayers$',
    r'^mem_gb1$',
    r'^modelbotorch_modular$'
    r'^sigkill$',
    r'^xr$',
    r'^youd$',
    r'^norman$',
    r'^oom$',
    r'^float_param$',
    r'^pwd$',
    r'^submitit$',
    r'^tracebreakpoint$',
    r'^io$',
    r'^sigxfsz$',
    r'^infos$',
    r'^infos$',
    r'^sigpoll$',
    r'^builtins$',
    r'^sigquit$',
    r'^sigsegv$',
    r'^sigterm$',
    r'^sigtrap$',
    r'^sigsys$',
    r'^sigstop$',
    r'^sigpwr$',
    r'^sigvtalrm$',
    r'^choice_param1$',
    r'^dont$',
    r'^homenormanreposomnioptax$',
    r'^hostnamethinkpad44020211128$',
    r'^jobenvironmentjob_id2387026$',
    r'^runsmy_experiment0single_runs$',
    r'^pathlike$',
    r'^testsoptimization_example$',
    r'^choice_param$',
    r'^global_rank01$',
    r'^int_param$',
    r'^int_param_two$',
    r'^rwxr$',
    r'^local_rank01$',
    r'^sigfpe$',
    r'^sigprof$',
    r'^sigurg$',
    r'^sigwinch$',
    r'^sigfpe$',
    r'^sighup$',
    r'^sigill$',
    r'^sigint$',
    r'^sigkill$',
    r'^sigpipe$',
    r'^node01$',
    r'^sigabrt$',
    r'^sigalrm$',
    r'^sigbus$',
    r'^thinkpad44020211128$',
    r'^boxplot$',
    r'^stderr$',
    r'^stdout$',
    r'^orchestratoryaml$',
    r'^assertionerror$',
    r'^2019b$',
    r'^add_argument$',
    r'^args$',
    r'^allow_axes$',
    r'^2d$',
    r'^3d$',
    r'^ax_clientget_next_trials$',
    r'^barnard$',
    r'^ie$',
    r'^gpus$',
    r'^gridsize$'
    r'^checkout_to_latest_tested_version$',
    r'^test_wronggoing_stuff$',
    r'^squeue$',
    r'^empirical_bayes_thompson$',
    r'^dier$',
    r'^py$',
    r'^int$',
    r'^experiment_name$',
    r'^gpu$',
    r'^gotrequested$',
    r'^plot_typescatter_generation_method$',
    r'^plot_typetime_and_exit_code$',
    r'^plot_typeget_next_trials$',
    r'^hostname$',
    r'^excludenodeandrestartall$',
    r'^match_strings$',
    r'^gui$',
    r'^csv$',
    r'^exclude_params$',
    r'^cpu$',
    r'^darkmode$',
    r'^dresdende$',
    r'^plot_typetrial_index_result$',
    r'^png$',
    r'^run_dir$',
    r'^runtimes$',
    r'^taurus$',
    r'^subgraphs$',
    r'^sobol$',
    r'^indexresult$',
    r'^bubblesize$',
    r'^plot_typescatter_hex$',
    r'^plot_typeworker$',
    r'^merge_with_previous_runs$',
    r'^get_next_trials$',
    r'^filenamesvg$',
    r'^kde$',
    r'^plot_type$',
    r'^plot_type3d$',
    r'^plot_typegeneral$',
    r'^plot_typecpu_ram_usage$',
    r'^plot_typescatter$',
    r'^plot_typekde$',
    r'^plot_typegpu_usage$',
    r'^omniopt_plot$',
    r'^mem_gb$',
    r'^tutorialshelp$',
    r'^homes\d+/repos/omnioptaxgui/usage_stats.php\d*$',
    r'^errorexception$',
    r'^regex$',
    r'^errorno$',
    r'^max$',
    r'^min$',
    r'^runtime$',
    r'^s\d+$',
    r'^hyperparameter$',
    r'^dark-mode$',
    r'^python3$',
    r'^botorch-modular$',
    r'^v2\d*$',
    r'^omniopt2$',
    r'^slurm$',
    r'^botorch$',
    r'^omniopt$',
    r'^\d+$',
    r'^https?://\S+$',
    r'^\b[A-Z]{2,}\b$'
    r'^\b[A-Za-z0-9_-]+@[A-Za-z0-9._-]+\.[A-Za-z]{2,}\b$'
]

def extract_visible_text_from_html(html_content):
    try:
        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()

        # Extract the visible text
        visible_text = soup.get_text(separator='\n')

        # Clean up unnecessary whitespace and empty lines
        clean_text = "\n".join([line.strip() for line in visible_text.splitlines() if line.strip()])
        return clean_text
    except Exception as e:
        print(f"Error processing HTML content: {e}")
        return None

def clean_word(word):
    # Remove punctuation and split hyphenated words
    word = re.sub(r'[^\w\s-]', '', word)  # Remove punctuation except hyphen
    return word.split('-')  # Split on hyphens to check each part separately

def filter_emojis(text):
    # Remove emojis and other non-alphanumeric characters
    return ''.join(char for char in text if not emoji.is_emoji(char))

def check_spelling(text):
    try:
        # Initialize the spell checker with the American English dictionary
        spell = SpellChecker(language='en')

        # Split the text into words
        words = text.split()

        # Filter out words that match any of the ignored patterns or contain emojis
        filtered_words = []
        for word in words:
            cleaned_word_parts = clean_word(word)
            for part in cleaned_word_parts:
                part_no_emoji = filter_emojis(part)
                if part_no_emoji and not any(re.fullmatch(pattern, part_no_emoji, flags=re.IGNORECASE) for pattern in IGNORED_PATTERNS):
                    filtered_words.append(part_no_emoji)

        # Find words that are misspelled
        misspelled = spell.unknown(filtered_words)

        return sorted(misspelled)  # Sort the misspelled words alphabetically
    except Exception as e:
        print(f"Error checking spelling: {e}")
        return None

def process_php_file(file_path):
    try:
        # Execute the PHP file and capture the output
        result = subprocess.run(['php', file_path], capture_output=True, text=True)
        html_content = result.stdout

        # Extract the visible text from HTML content
        extracted_text = extract_visible_text_from_html(html_content)

        if extracted_text:
            # Perform spell check on the extracted text
            misspelled_words = check_spelling(extracted_text)

            if misspelled_words:
                print(f"Misspelled words in {file_path}:")
                print(", ".join(misspelled_words))
                return len(misspelled_words)
            else:
                print(f"\nNo misspelled words found in {file_path}.")
                return 0
        else:
            print(f"No text was extracted from {file_path}.")
            return 1
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 1

def process_directory(directory_path):
    total_errors = 0
    # Walk through the directory and process each .php file
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".php"):
                file_path = os.path.join(root, file)
                print(f"========================\nProcessing: {file_path}")
                errors = process_php_file(file_path)
                total_errors += errors
    return total_errors

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python php_spellcheck.py <file_or_directory>")
        sys.exit(1)

    input_path = sys.argv[1]

    if os.path.isfile(input_path):
        # If it's a single file, process it directly
        total_errors = process_php_file(input_path)
    elif os.path.isdir(input_path):
        # If it's a directory, process all PHP files recursively
        total_errors = process_directory(input_path)
    else:
        print(f"{input_path} is not a valid file or directory.")
        sys.exit(1)

    # Exit with the total number of errors found
    sys.exit(total_errors)


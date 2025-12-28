# <img class='emoji_nav' src='emojis/herb.svg' /> Environment Variables

<!-- List of all environment variables that change how OmniOpt2 works -->

<!-- Category: Advanced Usage -->

<div id="toc"></div>

## What are environment variables?

Every program on Linux has an environment, like a bash- or zsh-shell, that it runs in. These shells can contain variables that can change how OmniOpt2 works. Here is a list of all shell variables that change how OmniOpt2 works.

It is important that you run these commands before you run OmniOpt2, and also that you write <samp>export</samp> in front of them. Unexported variables are not passed to programs started by the shell.

## Table of environment variables

```run_php
$env_variables = [
    "OmniOpt2" => [
        "HIDE_PARETO_FRONT_TABLE_DATA" => "Hide pareto front table",
        "DO_NOT_SEARCH_FOLDERS_FOR_RESULTS_CSV" => "Do not search for folders with results.csv",
        "RUN_WITH_PYSPY" => "Run the OmniOpt2 main script with py-spy to create flame graphs",
        "ENABLE_BEARTYPE" => "Enables beartype for type checking, makes everything slower but less prone to unnoticed bugs",
        "DIE_AFTER_THIS_NR_OF_DONE_JOBS" => "Dies after DIE_AFTER_THIS_NR_OF_DONE_JOBS jobs, only useful for debugging",
        "install_tests" => "Install test modules",
        "RUN_UUID" => "Sets the UUID for the run. Default is a new one via uuidgen",
        "OO_NO_LOGO" => "Disables showing the logo",
        "OO_MAIN_TESTS" => "Sets the user-id to affed00faffed00faffed00faffed00f, so the statistics can determine whether you are a real user or a test-user",
        "ITWORKSONMYMACHINE" => "Sets the user-id to affeaffeaffeaffeaffeaffeaffeaffe, so the statistics can determine whether you are a real user or a the main developer (only I should set this variable)",
        "root_venv_dir" => "Path to where virtualenv should be installed. Default is \$HOME",
        "DISABLE_SIXEL_GRAPHICS" => "Disables sixel-graphics, no matter what other parameters are set",
        "DONT_INSTALL_MODULES" => "Disables installing modules",
        "DONT_SHOW_DONT_INSTALL_MESSAGE" => "Don't show messages regarding the installation of modules",
        "DONTSTARTFIREFOX" => "Don't start firefox when RUN_WITH_COVERAGE is defined",
        "RUN_WITH_COVERAGE" => "Runs omniopt- and plot-script with coverage to find out test code coverage",
        "CI" => "Disables certain tests in a CI environment",
        "PRINT_SEPARATOR" => "Prints a seperator line after OmniOpt2 runs (useful for automated tests)",
        "SKIP_SEARCH" => "Skip the actual search, very useful for debugging",
        "SKIP_SEARCH_EXIT_CODE" => "The exit code that should be used for this job",
        "DEBUG_PARAM_EVAL" => "Tests if the argparse clone for bash works and creates the required arguments. Must be non-empty exported string in your shell to run a test. Please use it mainly over .tests/test_bash_argparse_clone"
    ],
    "Plot-Script" => [
        "BUBBLESIZEINPX" => "Size of bubbles in plot scripts in px"
    ],
    "Test-Scripts" => [
        "CREATE_PLOT_HELPS" => "Creates the help files for the tutorials page for each omniopt_plot",
        "DONT_SHOW_STARTUP_COMMAND" => "Don't show the command to start from start_simple_optimization_run. Set to 1 to hide.",
        "NO_RUNTIME" => "Don't show omniopt_plot runtime at the end",
        "NO_TESTS" => "Disable tests for .tests/pre-commit-hook",
        "NO_NO_RESULT_ERROR" => "Disable errors to stdout for plots when no results are found",
        "SHOW_COMMAND_BEFORE_EXECUTION" => "Show the command before execution",
        "SHOW_SUCCESS" => ".tests/php_unit_tests then also shows successes then, set it to 1",
        "IS_RUNNING_UNIT_TESTS" => ".gui/tests.php: If set, it doesn't print out some warnings that are expected to reduce clutter"
    ]
];

function generate_env_table($env_dict) {
    $html = '<table>';
    $html .= '<tr class="invert_in_dark_mode"><th>Name</th><th>What does it do?</th></tr>';

    foreach ($env_dict as $section => $variables) {
        $html .= '<tr><th class="section-header invert_in_dark_mode" colspan="2">' . htmlspecialchars($section) . '</th></tr>';
        foreach ($variables as $name => $desc) {
            $html .= '<tr>';
            $html .= '<td><pre class="invert_in_dark_mode"><code class="language-bash">' . htmlspecialchars($name) . '</code></pre></td>';
            $html .= '<td>' . htmlspecialchars($desc) . '</td>';
            $html .= '</tr>';
        }
    }

    $html .= '</table>';
    return $html;
}
print(generate_env_table($env_variables));
```

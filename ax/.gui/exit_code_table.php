    <h2>Exit Code Information</h2>
        <table>
            <tr class="invert_in_dark_mode">
                <th>Exit Code</th>
                <th>Error Group Description</th>
            </tr>
            <?php
                $exit_code_info = [];

                array_unshift(
                    $exit_code_info,
                    [
			    0 => "Seems to have worked properly",
			    1 => "Could not create log dir",
			    2 => "Loading of Environment failed",
			    3 => "Invalid exit code detected",
			    4 => "Failed loading modules",
			    5 => "Errors regarding toml or yaml config files",
			    6 => "Error creating .logs dir",
			    7 => "Probably versioning error. Try removing virtualenv and try again.",
			    8 => "Probably something went wrong trying to plot sixel graphics",
			    9 => "Probably something went wrong trying to use or define the ax_client or executor",
			    10 => "Usually only returned by dier (for debugging).",
			    11 => "Required program not found (check logs)",
			    12 => "Error with pip, check logs.",
			    13 => "Run folder already exists",
			    15 => "Unimplemented error.",
			    16 => "Wrongly called .py files: Probably you tried to call them directly instead of over the bash file",
			    18 => "test_wronggoing_stuff program not found (only --tests).",
			    19 => "Something was wrong with your parameters. See output for details.",
			    31 => "Basic modules could not be loaded or you cancelled loading them.",
			    44 => "Continuation of previous job failed.",
			    47 => "Missing checkpoint or defective file or state files (check output).",
			    49 => "Something went wrong while creating the experiment.",
			    87 => "Search space exhausted or search cancelled.",
			    88 => "Search was done according to ax",
			    90 => "Error creating the experiment_args",
			    91 => "Error creating the experiment_args",
			    99 => "It seems like the run folder was deleted during the run.",
			    100 => "--mem_gb or --gpus, which must be int, has received a value that is not int.",
			    101 => "Error using ax_client: it was not defined where it should have been",
			    103 => "--time is not in minutes or HH:MM format.",
			    104 => "One of the parameters --mem_gb, --time, or --experiment_name is missing.",
			    105 => "Continued job error: previous job has missing state files.",
			    130 => "Interrupt-Signal detected",
			    133 => "Error loading --config_toml, --config_json or --config_yaml",
			    138 => "USR-Signal detected.",
			    142 => "Error in Models like THOMPSON or EMPIRICAL_BAYES_THOMPSON. Not sure why.",
			    146 => "CONT-Signal detected.",
			    181 => "Error parsing --parameter. Check output for more details.",
			    191 => "Could not create workdir ",
			    192 => "Unknown data type (--tests).",
			    193 => "Error in printing logs. You may be on a read only file system.",
			    199 => "This happens on unstable file systems when trying to write a file.",
			    203 => "Unsupported --model.",
			    206 => "Invalid orchestrator file.",
			    210 => "Unknown orchestrator mode",
			    211 => "Git checkout failed (--checkout_to_latest_tested_version)",
			    233 => "No random steps set.",
			    242 => "Error at fetching new trials.",
			    243 => "Job was not found in squeue anymore, it may got cancelled before it ran.",
			    244 => "get_executor() failed. See logs for more details."
                    ]
                );

                if (!array_key_exists("HIDE_SUBZERO", $GLOBALS)) {
                    array_unshift(
                        $exit_code_info,
                        [
                        "-1" => "No proper Exit code found",
                        ]
                    );
                }

                foreach ($exit_code_info as $code_block_id => $code_block_content) {
                    foreach ($code_block_content as $code => $description) {
                        echo "<tr>";
                        echo "<td>$code</td>";
                        echo "<td>$description</td>";
                        echo "</tr>";
                    }
                }
                ?>
        </table>

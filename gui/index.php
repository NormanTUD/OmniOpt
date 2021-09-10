<?php
	function get_get ($name, $default = NULL) {
		if(array_key_exists($name, $_GET)) {
			return $_GET[$name];
		} else {
			return $default;
		}
	}
?>
<!doctype html>
<html lang="us">
	<head>
		<meta charset="utf-8">
		<title>OmniOpt-Generator</title>
		<link href="jquery-ui.css" rel="stylesheet">
		<link href="main.css" rel="stylesheet" />
		<link href="prism.css" rel="stylesheet" />
	</head>
	<body>
		<script>
			function getNewURL(url, param, paramVal){
				var newAdditionalURL = "";
				var tempArray = url.split("?");
				var baseURL = tempArray[0];
				var additionalURL = tempArray[1];
				var temp = "";
				if (additionalURL) {
					tempArray = additionalURL.split("&");
					for (var i=0; i<tempArray.length; i++){
						if(tempArray[i].split('=')[0] != param){
							newAdditionalURL += temp + tempArray[i];
							temp = "&";
						}
					}
				}

				var rows_txt = temp + "" + param + "=" + encodeURIComponent(paramVal);
				return baseURL + "?" + newAdditionalURL + rows_txt;
			}

			function update_url_param(param, val) {
				window.history.replaceState('', '', getNewURL(window.location.href, param, val));
			}

			var urlParams = new URLSearchParams(window.location.search);
			// Add new partitions here and they'll automatically be added as option
			var partition_data = {
				"ml": {
					"number_of_workers": 180,
					"computation_time": 168,
					"max_number_of_gpus": 6,
					"max_mem_per_core": 63500,
					"mem_per_cpu": 63500,
					"name": "Machine Learning (ppc64le)",
					"warning": "It is not recommended to use the /scratch or /lustre-filesystem on the ML partition"
				},
				"gpu2": {
					"number_of_workers": 63,
					"computation_time": 168,
					"max_number_of_gpus": 4,
					"max_mem_per_core": 2583,
					"mem_per_cpu": 3875,
					"name": "GPU2 (amd64)",
					"warning": ""
				},
				"alpha": {
					"number_of_workers": 160,
					"computation_time": 168,
					"max_number_of_gpus": 8,
					"max_mem_per_core": 49500,
					"mem_per_cpu": 49500,
					"name": "Alpha Centauri (amd64)",
					"warning": ""
				}, 
				"hpdlf": {
					"number_of_workers": 42,
					"computation_time": 168,
					"max_number_of_gpus": 3,
					"max_mem_per_core": 7916,
					"mem_per_cpu": 7916,
					"name": "Hardware for Deep Learning (amd64)",
					"warning": ""
				}, 
				"haswell64": {
					"number_of_workers": 1056,
					"computation_time": 168,
					"max_number_of_gpus": 0,
					"max_mem_per_core": 2541,
					"mem_per_cpu": 2541,
					"name": "haswell64 (amd64)",
					"warning": ""
				},
				"haswell128": {
					"number_of_workers": 80,
					"computation_time": 168,
					"max_number_of_gpus": 0,
					"max_mem_per_core": 5250,
					"mem_per_cpu": 5250,
					"name": "haswell128 (amd64)",
					"warning": ""
				},
				"haswell256": {
					"number_of_workers": 40,
					"computation_time": 168,
					"max_number_of_gpus": 0,
					"max_mem_per_core": 10583,
					"mem_per_cpu": 10583,
					"name": "haswell256 (amd64)",
					"warning": ""
				}
			};
		</script>

		<h1>OmniOpt-Generator</h1>
		<i>Based on <a target="_blank" href="https://github.com/hyperopt/hyperopt">HyperOpt</a></i> &mdash;
		<i>Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. TProc. of the 30th International Conference on Machine Learning (ICML 2013), June 2013, pp. I-115 to I-23.</i><br><br>

		<h2>What your program needs to be like: </h2>
		<ol>
			<li>It needs to be able to run on Taurus on your account</li>
			<li>It needs to accept it's hyperparameters by command line parameters (for example via <tt>sys.argv</tt>)</li>
			<li>A lower <tt>RESULT</tt> must mean that it's somehow better, i.e. the area where lower results are are researched more</li>
			<li>The result needs to be printed in Stdout in a single line like this: <br><pre class="language-bash">RESULT: 0.123456</pre></li>
			<li>If you want to <i>maximize</i> a value, just prepend <tt>-</tt> to the result string to &raquo;negate&laquo; it and turn a maximization-problem to a minimization-problem, like this: <pre class="language-bash">RESULT: -0.123456</pre></li>
			<li>Only the last <tt>RESULT</tt>-line counts, all others will be disregarded!</li>
			<li>Make sure your programs can be run from any Working Directory, as it's problable that the CWD will not be the same as the directory your program runs in</li>
			<li>Make sure your program runs on the architecture of the partition you chose</li>
		</ol>

		<h2>Additional information:</h2>
		<ol>
			<li>Once the job ran, go to the <tt>omniopt</tt>-folder it ran in and run <pre class="language-bash">bash evaluate-run.sh</pre> to gain easy access to the results</li>
			<li>Anything in the <tt>STRING: FLOAT</tt>-Format will be saved in the DB and can be output to a CSV file via <tt>bash evaluate-run.sh</tt></li>
			<li>If your program does not seem to run properly, do <pre class="language-bash">bash evaluate-run.sh</pre>, go to your project and run <tt>Check this project for errors</tt>. It will check for the most common errors in your project.</li>
		</ol>

		<form id="all" oninput="update_config()">
			<table id="maintable">
				<tr>
					<td>
						<table id="configtable">
							<tr>
								<th colspan="2">Slurm-Options</th>
							</tr>
							<tr>
								<td><span class="helpicon" data-help="The name of the project. Gets used as project folder and in outputting the results.">Optimization Run name:</span></td>
								<td>
									<input class="parameter_input" type="text" id="projectname" placeholder="Optimization Run name" value="<?php print htmlentities(get_get("projectname")); ?>" onkeyup="update_url_param('projectname', this.value)" />
									<div class="errors" id="noprojectnameerror"></div>
									<div class="errors" id="invalidprojectnameerror"></div>
								</td>
							</tr>
							<tr>
								<td><span class="helpicon" data-help="The partition on which the job should run. Different partitions may have different CPU-architectures and different use-cases (e.g. the ML-partition uses ppc64le instead of amd64 and has powerful GPUs for training neural networks)">Partition:</span></td>
								<td>
									<select onchange="update_url_param('partition', this.value)" name="partition" id="partition"></select>
									<script>
										var url_partition = "None";
										if(urlParams.has("partition")) {
											url_partition = urlParams.get("partition");
										}
										var partition_select = document.getElementById('partition');
										for (var this_partition in partition_data) {
											var opt = document.createElement('option');
											opt.value = this_partition;
											if(this_partition == url_partition) {
												opt.setAttribute("selected", true);
											}
											opt.innerHTML = partition_data[this_partition]["name"];
											partition_select.appendChild(opt);
										}
									</script>
									<div id="partitionwarning"></div>
								</td>
							</tr>
							<tr>
								<td><span class="helpicon" data-help="On GPUs, many complex programs run way faster. But you have to manually make your program work with GPU; this option only assigns GPUs to the workers so that you can use them (your program will then have 1 GPU available).">Enable GPU?</span></td>
								<td>
									<input class="parameter_input" type="checkbox" value="1" checked id="enable_gpus" />
									<div id="gputext"></div>
								</td>
							</tr>
							<tr>
								<td><span class="helpicon" data-help="If you use a reservation, the job will start immediately.">Reservation:</span></td>
								<td><input class="parameter_input" type="text" id="reservation" placeholder="Just leave this empty if you have no reservation" value="<?php print htmlentities(get_get("reservation")); ?>" onkeyup="update_url_param('reservation', this.value)" /></td>
							</tr>
							<tr>
								<td><span class="helpicon" data-help="This option allows you to associate that job to a specific account with higher priorities, so it starts faster.">Account:</span></td>
								<td><input class="parameter_input" type="text" id="account" placeholder="Name of the HPC-Project, not your username. Can be left empty." value="<?php print htmlentities(get_get("account")); ?>" onkeyup="update_url_param('account', this.value)" /></td>
							</td>
							<tr>
								<td><span class="helpicon" data-help="Every hyperparameter-constellation gets tested on real hardware, and the more workers you assign, the faster many jobs get done. But it might also take longer for slurm to allocate the ressources for you.">Number of workers:</span></td>
								<td>
									<input class="parameter_input" type="number" value="<?php print htmlentities(get_get("number_of_workers", 5)); ?>" onkeyup="update_url_param('number_of_workers', this.value)" min="1" id="number_of_workers" placeholder="Number of parallel workers" />
									<div id="maxworkerwarning"></div>
									<div id="workerevalswarning"></div>
									<div class="errors" id="emptyworkerwarning"></div>
								</td>
							</tr>
							<tr>
								<td><span class="helpicon" data-help="The maximum amount of memory per worker. This depends largely on the amount of memory your program needs, plus a small overhead from OmniOpt itself (&#x2248; 100MB)" data-help="">Mem. (MB)/Worker:</span></td>
								<td>
									<input class="parameter_input" type="number" value="<?php print htmlentities(get_get("mem_per_worker", 2000)); ?>" onkeyup="update_url_param('mem_per_worker', this.value)" min="1" id="mem_per_cpu" placeholder="Memory per worker" /><div id="maxmemperworkertext"></div>
									<div class="errors" id="memworkererror"></div>
								</td>
							</tr>
							<tr>
								<td><span class="helpicon" data-help="The maximum amount of time this optimization will run (hard-limit)">Runtime:</span></td>
								<td>
									<input class="parameter_input" type="number" value="<?php print htmlentities(get_get("runtime", 20)); ?>" onkeyup="update_url_param('runtime', this.value)" min="1" id="computing_time" placeholder="Runtime (h)" /><div id="timewarning"></div>
									<div class="errors" id="computingtimeerror"></div>
								</td>
							</tr>
							<tr>
								<th colspan="2">OmniOpt-Options</th>
							</tr>
							<tr>
								<td><span class="helpicon" data-help="tpe.suggest searches intelligently and gives good results faster, hyperopt.rand.suggest searches the search space randomly">Search type:</span></td>
								<td>
									<select onchange="update_url_param('searchtype', this.value)" onchange="update_config()" class="parameter_input" id="algo_name">
										<option value="tpe.suggest" <?php if (get_get("searchtype", "") == "tpe.suggest") { print ' selected="true" '; } ?>>Suggestion Search (tpe.suggest)</option>
										<option value="hyperopt.rand.suggest" <?php if (get_get("searchtype", "") == "hyperopt.rand.suggest") { print ' selected="true" '; } ?>>Random Search (hyperopt.rand.suggest)</option>
									</select>
								</td>
							</tr>
							<tr>
								<td><span class="helpicon" data-help="The number of hyperparameter-constellations that should be tested. If this takes longer than the defined computing time, it might not be reached. If doesn't take longer, then the Optimization will end once reached.">Max. number set evaluations:</span></td>
								<td>
									<input class="parameter_input" type="number" value="<?php print htmlentities(get_get("max_evals", 1000)); ?>" min="1" id="max_evals" onkeyup="update_url_param('max_evals', this.value)" placeholder="Max number of evaluations" />
									<div class="errors" id="maxevalserror"></div>
								</td>
							</tr>
							<tr>
								<td><span class="helpicon" data-help="The program you want to optimize that prints 'RESULT: ...' somewhere in STDOUT and accepts command-line-parameters">Objective program:</span></td>
								<td>
									<i>($x_0) for zero'th command line parameter, ($x_1) for first, ...</i><br>
									<textarea class="parameter_input" type="text" id="objective_program" placeholder="bash /scratch/placeholder.sh --layers=($x_0) --neurons=($x_1) ($x_2)" onchange="update_url_param('objective_program', this.value)"><?php print htmlentities(get_get("objective_program", '')); ?></textarea>
									<div class="errors" id="objectiveprogramerror"></div>
									<ul>
										<li>Make sure your program-path has execution rights or run it with the command it's supposed to run as (<tt>bash</tt>, <tt>python</tt>, ...)</li>
										<li>The program path can be anywhere, also in different directories from OmniOpt's folder</li>
										<li>It is recommended to use a <tt>run.sh</tt>-file, check <a href="run_sh.php" target="_blank">this tutorial</a> for creating a <tt>run.sh</tt>-file for your script</li>
										<li>Example: <pre class=language-bash">bash /scratch/run.sh --layers=($x_0) --neurons=($x_1) ($x_2)</pre></li>
									</ul>
								</td>
							</tr>
							<tr>
								<td><span class="helpicon" data-help="This is how many parameters the program has that need to be optimized.">Number of hyperparameters:</span></td>
								<td>
									<input class="parameter_input" type="number" min="1" value="<?php print htmlentities(get_get("number_of_parameters", 1)); ?>" oninput="change_number_of_parameters();" id="number_of_parameters" placeholder="Number of parameters" onkeyup="update_url_param('number_of_parameters', this.value)" />
									<div class="errors" id="toofewparameterserror"></div>
								</td>
							</tr>
						</table>

						<button type="button" onclick="$('#noneedtochange').toggle();" id="toggle_noneedtochange">Show/hide stuff that you probably don't need to change.</button>
						<div id="noneedtochange" style="display: none;">
							<table style="background-color: yellow;">
								<tr>
									<th colspan="2">Installer options</th>
								</tr>
								<tr>
									<td>Enable Installer-Debug-Mode</td>
									<td>
										<input type="checkbox" value="1" id="enable_curl_debug" />
									</td>
								</tr>
								<tr>
									<td>Don't ask to start a new job and don't start new job</td>
									<td>
										<input type="checkbox" value="1" id="dont_ask_to_start" />
									</td>
								</tr>
								<tr>
									<td>Don't add command to shell history</td>
									<td>
										<input type="checkbox" value="1" id="dont_add_to_shell_history" />
									</td>
								</tr>
								<tr>
									<th colspan="2"><tt>sbatch</tt>-Options</th>
								</tr>
								<tr>
									<td>Number of CPUs per worker:</td>
									<td>
										<input type="number" value="4" min="1" id="number_of_cpus_per_worker" placeholder="Number of CPUs per worker" />
									</td>
								</tr>
								<tr>
									<th colspan="2"><tt>config.ini</tt>-Options</th>
								</tr>
								<tr>
									<td>Precision:</td>
									<td>
										<input type="number" value="8" min="0" id="precision" placeholder="Precision of floats" />
									</td>
								</tr>
								<tr>
									<td>Worker last Job timeout:</td>
									<td>
										<input type="number" value="500" min="0" id="worker_last_job_timeout" placeholder="Number of seconds after which a worker times out" />
									</td>
								</tr>
								<tr>
									<td>Worker poll interval:</td>
									<td>
										<input type="number" step="any" value="10" min="0" id="poll_interval" placeholder="Number of seconds between the worker looking for new jobs" />
									</td>
								</tr>
								<tr>
									<td>Kill worker after n polls without new jobs:</td>
									<td>
										<input type="number" step="1" value="100000" min="0" id="kill_after_n_no_results" placeholder="Kills worker after this number of polls that consecutively got no results" />
									</td>
								</tr>
								<tr>
									<td>Enable OmniOpt debug Options?</td>
									<td>
										<input type="checkbox" value="1" id="enable_debug" />
									</td>
								</tr>
								<tr>
									<td>Show live output?</td>
									<td>
										<input type="checkbox" value="1" id="show_live_output" />
									</td>
								</tr>
							</table>
						</div>

						<div id="parameters"></div>
					</td>
					<td>
						<div class="errors" id="errors"></div>
						<div id="hidewhenmissingdata">
							<div id="downloadbashlink"></div>
							<div id="downloadbashlink2"></div>
							<h3>Run OmniOpt</h3>
							<div>Execute this command on your Taurus-Shell to run this optimization:</div>
							<div id="bashcommand"></div>
							<div id="copytoclipboard" style="display: none"><button type="button" id="copytoclipboardbutton">Copy to clipboard</button></div>
							<div id="copied" style="display: none">&#128203; <b>Copied bash command to the clipboard</b></div>

						</div>
						<br>
						Autohide config and sbatch when it is not needed? <input type="checkbox" value="1" checked="checked" id="autohide_config_and_sbatch" /><br>
						<button type="button" onclick="$('#hidewhendone').toggle();" id="toggle_hidewhendone">Show/hide config and sbatch.</button>
						<div id="hidewhendone">
							<h3>Config file</h3>
							<h3>Sbatch command</h3>
							<pre id="sbatch"></pre>
							<div id="downloadlink"></div>
							<pre id="config"></pre>
						</div>

						<br>
						<button type="button" onclick="$('#configiniparser').toggle();" id="toggle_configiniparser">Parse from <tt>config.ini</tt>.</button>
						<div id="configiniparser">
							<textarea id="configini" placeholder="Paste your config here and press the button at the end of this dialog"></textarea><br>
							<button type="button" onclick="parse_from_config_ini()">Parse from <tt>config.ini</tt> now</button>
						</div>
					</td>
				</tr>
			</table>
		</form>

		<script src="external/jquery/jquery.js"></script>
		<script src="jquery-ui.js"></script>
		<script src="prism.js"></script>
		<script src="main.js"></script>
	</body>
</html>

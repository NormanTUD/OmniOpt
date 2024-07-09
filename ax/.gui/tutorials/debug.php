<?php
	include("../_header_base.php");
?>
<h1>Debugging</h1>

<div id="toc"></div>

<h2 id="debug">How to find errors in my runs?</h2>

<p>OmniOpt2 saves many logs in the <a target="_blank" href="folder_structure.php"><tt>run</tt>-folder</a>. First, check the <tt>.stdout</tt>-file of
the run. OmniOpt2 will tell you many things when it detects errors there.</p>

<p>If you don't find anything useful in the <tt>.stdout</tt>-file, look into the 
<tt>runs/my_experiment/0/single_runs/</tt>-folder. It contains the outputs of each
worker in seperate directories. It looks something like this:</p>

<pre>
submitit INFO (2024-07-08 17:34:57,444) - Starting with JobEnvironment(job_id=2387026, hostname=thinkpad44020211128, local_rank=0(1), node=0(1), global_rank=0(1))
submitit INFO (2024-07-08 17:34:57,445) - Loading pickle: /home/norman/repos/OmniOpt/ax/runs/__main__tests__/1/single_runs/2387026/2387026_submitted.pkl
parameters: {'int_param': -100, 'float_param': -100.0, 'choice_param': 1, 'int_param_two': -5}
Debug-Infos: 
========
DEBUG INFOS START:
Program-Code: ./.tests/optimization_example --int_param='-100' --float_param='-100.0' --choice_param='1'  --int_param_two='-5'
pwd: /home/norman/repos/OmniOpt/ax
File: ./.tests/optimization_example
Size: 4065 Bytes
Permissions: -rwxr-xr-x
Owner: norman
Last access: 1720413747.240997
Last modification: 1718802483.4208295
Hostname: thinkpad44020211128
========
DEBUG INFOS END

./.tests/optimization_example --int_param='-100' --float_param='-100.0' --choice_param='1'  --int_param_two='-5'
stdout:
RESULT: -222001.32

Result: -222001.32
EXIT_CODE: 0
submitit INFO (2024-07-08 17:34:57,477) - Job completed successfully
submitit INFO (2024-07-08 17:34:57,477) - Exiting after successful completion
</pre>

<p>The output (stdout and stderr) of your job is after the <tt>stdout:</tt> and before the <tt>EXIT_CODE: 0</tt>. Check
this output for errors. Here, you'd see Slurm Errors for your job.</p>

<p>Also check the exit-code. Some exit codes have special meanings, like 137, have special meaning. See this table for special exit codes:</p>

<p>The code between <tt>DEBUG INFOS START:</tt> and <tt>DEBUG INFOS END</tt> contains info about the string of the command that is about to be executed. It is searched for file paths and the permissions, owner and so on of the file is displayed. This is useful to check for seeing if scripts you call really have the <tt>x</tt>-flag, or are readable and so on. All pathlike structures will be searched and only printed here if they link to a valid file.</p>


<p>If, for example, you have error code 137, that means you likely ran out of RAM and need to increase the amount of RAM for your workers.</p>

<table>
	<thead>
		<tr>
			<th>Exit Code</th>
			<th>Description</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>0</td>
			<td>Success</td>
		</tr>
		<tr>
			<td>1</td>
			<td>General error</td>
		</tr>
		<tr>
			<td>2</td>
			<td>Misuse of shell builtins</td>
		</tr>
		<tr>
			<td>126</td>
			<td>Command invoked cannot execute</td>
		</tr>
		<tr>
			<td>127</td>
			<td>Command not found</td>
		</tr>
		<tr>
			<td>128</td>
			<td>Invalid argument to exit</td>
		</tr>
		<tr>
			<td>129</td>
			<td>Hangup (SIGHUP)</td>
		</tr>
		<tr>
			<td>130</td>
			<td>Interrupt (SIGINT)</td>
		</tr>
		<tr>
			<td>131</td>
			<td>Quit (SIGQUIT)</td>
		</tr>
		<tr>
			<td>132</td>
			<td>Illegal instruction (SIGILL)</td>
		</tr>
		<tr>
			<td>133</td>
			<td>Trace/breakpoint trap (SIGTRAP)</td>
		</tr>
		<tr>
			<td>134</td>
			<td>Abort (SIGABRT)</td>
		</tr>
		<tr>
			<td>135</td>
			<td>Bus error (SIGBUS)</td>
		</tr>
		<tr>
			<td>136</td>
			<td>Floating-point exception (SIGFPE)</td>
		</tr>
		<tr>
			<td>137</td>
			<td>Killed (SIGKILL) - maybe caused by OOM killer</td>
		</tr>
		<tr>
			<td>138</td>
			<td>Segmentation fault (SIGSEGV)</td>
		</tr>
		<tr>
			<td>139</td>
			<td>Broken pipe (SIGPIPE)</td>
		</tr>
		<tr>
			<td>140</td>
			<td>Alarm clock (SIGALRM)</td>
		</tr>
		<tr>
			<td>141</td>
			<td>Termination (SIGTERM)</td>
		</tr>
		<tr>
			<td>142</td>
			<td>Urgent condition on socket (SIGURG)</td>
		</tr>
		<tr>
			<td>143</td>
			<td>Socket has been shut down (SIGSTOP)</td>
		</tr>
		<tr>
			<td>145</td>
			<td>File size limit exceeded (SIGXFSZ)</td>
		</tr>
		<tr>
			<td>146</td>
			<td>Virtual timer expired (SIGVTALRM)</td>
		</tr>
		<tr>
			<td>147</td>
			<td>Profiling timer expired (SIGPROF)</td>
		</tr>
		<tr>
			<td>148</td>
			<td>Window size change (SIGWINCH)</td>
		</tr>
		<tr>
			<td>149</td>
			<td>I/O now possible (SIGPOLL)</td>
		</tr>
		<tr>
			<td>150</td>
			<td>Power failure (SIGPWR)</td>
		</tr>
		<tr>
			<td>151</td>
			<td>Bad system call (SIGSYS)</td>
		</tr>
	</tbody>
</table>

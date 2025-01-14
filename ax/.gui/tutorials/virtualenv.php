<h1>VirtualEnv</h1>

<div id="toc"></div>

<h2 id="what_are_virtual_environments">What is a virtual environment?</h2>

<p>Most software depends on other software to work. This software usually needs to be there in a very specific version, and may conflict with other programs that need other versions of the same program. To solve this, <i>virtual environments</i> have been invented. They are a single folder containing all the dependencies your program needs.</p>

<h2 id="where_is_omniopts_venv">Where will OmniOpt2 install it's dependencies to?</h2>

<p>By default, OmniOpt2 tries to install the dependencies in a folder in your home folder, so that you only need to install OmniOpt2 once, and all other installations can access this virtualenv.</p>

<p>The default folder will be created like this: <samp>.omniax_$(uname -m)_$(python3 --version | sed -e 's# #_#g')$_cluster</samp>, while <samp>uname -m</samp> yields to your CPU architecture (usually <samp>x86_64</samp>,
<samp>python --version | sed -e 's# #_#g'</samp> will yield to your python version (e.g. <samp>3.11.2</samp>), and <samp>$_cluster</samp> contains the cluster name, taken from the environment variable <samp>CLUSTERHOST</samp>.</p>

<h2 id="change_venv_dir">Change the directory the virtual environment will be installed to</h2>

<p>If you cannot write in your home, for example, because it is too full, you can set the path where the virtual environment will be installed and used to. Simply set the variable:</p>

<pre><code class="language-bash">export root_venv_dir=/path/where/it/should/be/installed/to</code></pre>

<p>You need to do this every time before you start OmniOpt2 as well, since without that shell variable (that has to be exported!) it will try to do it from your home again.</p>

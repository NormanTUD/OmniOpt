<?php
	include("_header_base.php");
?>
    <link href="jquery-ui.css" rel="stylesheet">
    <style>
        body {
            font-family: Verdana, sans-serif;
        }
        .toc {
            margin-bottom: 20px;
        }
        .toc ul {
            list-style-type: none;
            padding: 0;
        }
        .toc li {
            margin-bottom: 5px;
        }
        .toc a {
            text-decoration: none;
            color: #007bff;
        }
        .toc a:hover {
            text-decoration: underline;
        }
    </style>
    <link href="prism.css" rel="stylesheet" />

    <h1>Continue jobs</h1>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#script-example">Script Example</a></li>
        </ul>
    </div>

    <h2 id="script-example">Script Example</h2>
    <p>To make your script robust enough for the environment of OmniOpt on HPC-Systems,
    it is recommended that you do not run your script directly in the objective program
    string. Rather, it is recommended that you create a <tt>run.sh</tt>-file from which
    your program gets run.</p>

    <script src="prism.js"></script>
    <script>
        Prism.highlightAll();
    </script>
</body>
</html>


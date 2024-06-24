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

    <h1>Folder structure of OmniOpt runs</h1>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#script-example">Script Example</a></li>
        </ul>
    </div>

    <h2 id="script-example">Script Example</h2>

    <pre><code class="language-bash">#!/bin/bash -l
This is currently in work
</code></pre>
    <script src="prism.js"></script>
    <script>
        Prism.highlightAll();
    </script>
</body>
</html>


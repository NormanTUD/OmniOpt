<?php
// Die Datenstruktur für die Tabs und deren Inhalte
$tabs = [
    'General Info' => [
        'id' => 'tab_general_info',
        'content' => '<pre>General Info</pre>',
    ],
    'Next-Trials' => [
        'id' => 'tab_next_trials',
        'content' => '<p>Next Trials</p>',
    ],
    '2D-Scatter' => [
        'id' => 'tab_scatter_2d',
        'content' => '<div id="scatter2d"></div>',
    ],
    '3D-Scatter' => [
        'id' => 'tab_scatter_3d',
        'content' => '<div id="scatter3d"></div>',
    ],
    'Parallel Plot' => [
        'id' => 'tab_parallel',
        'content' => '<div id="parallel"></div>',
    ],
    'Results-Table' => [
        'id' => 'tab_table',
        'content' => '<div id="table"></div>',
    ],
    'Single Logs' => [
        'id' => 'tab_logs',
        'content' => generate_log_tabs(50),  // Beispiel: 50 Logs dynamisch generiert
    ],
];

// Funktion zur Generierung der Log-Tab-Inhalte
function generate_log_tabs($nr_files) {
    $output = '<section class="tabs" style="width: 100%"><menu role="tablist" aria-label="Single-Runs">';
    for ($i = 0; $i < $nr_files; $i++) {
        $output .= '<button role="tab" ' . ($i == 0 ? 'aria-selected="true"' : '') . ' aria-controls="single_run_' . $i . '">Single-Run-' . $i . '</button>';
    }
    $output .= '</menu>';
    for ($i = 0; $i < $nr_files; $i++) {
        $output .= '<article role="tabpanel" id="single_run_' . $i . '"><pre>C:\WINDOWS\SYSTEM32> Single-Run ' . $i . '</pre></article>';
    }
    $output .= '</section>';
    return $output;
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniOpt2-Share</title>
    <script src="../plotly-latest.min.js"></script>
    <script src="gridjs.umd.js"></script>
    <link href="mermaid.min.css" rel="stylesheet" />
    <link href="tabler.min.css" rel="stylesheet">
    <?php include("css.php"); ?>
</head>
<body>
    <div class="page window" style='font-family: sans-serif'>
        <div class="title-bar">
            <div class="title-bar-text">OmniOpt2-Share</div>
        </div>
        <div id="spinner" class="spinner"></div>

        <div id="main_window" style="display: none" class="container py-4 window-body has-space">
            <section class="tabs" style="width: 100%">
                <menu role="tablist" aria-label="OmniOpt2-Run">
                    <?php
                    // Tabs generieren
                    $first_tab = true;
                    foreach ($tabs as $tab_name => $tab_data) {
                        echo '<button role="tab" aria-controls="' . $tab_data['id'] . '" ' . ($first_tab ? 'aria-selected="true"' : '') . '>' . $tab_name . '</button>';
                        $first_tab = false;
                    }
                    ?>
                </menu>

                <?php
                // Inhalte der Tabs generieren
                foreach ($tabs as $tab_name => $tab_data) {
                    echo '<article role="tabpanel" id="' . $tab_data['id'] . '" ' . ($tab_name === 'General Info' ? '' : 'hidden') . '>';
                    echo $tab_data['content'];
                    echo '</article>';
                }
                ?>
            </section>
        </div>
    </div>

    <script src="functions.js"></script>
    <script src="main.js"></script>
</body>
</html>

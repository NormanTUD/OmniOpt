<?php
header('Content-Type: application/json');

// Anzahl der Dummy-Datenpunkte
$num_points = 1000;

// Zufallsdaten generieren
$data = [];
for ($i = 0; $i < $num_points; $i++) {
    $data[] = [
        'learning_rate' => round(mt_rand(1, 100) / 1000, 4),
        'batch_size' => mt_rand(8, 256),
        'accuracy' => round(mt_rand(70, 99) + mt_rand(0, 9999) / 10000, 4),
        'num_layers' => mt_rand(1, 10),
        'neurons_per_layer' => mt_rand(32, 512)
    ];
}

// JSON ausgeben
echo json_encode($data, JSON_PRETTY_PRINT);
?>

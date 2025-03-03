<?php
header('Content-Type: application/json');

// Anzahl der Datenpunkte
$num_data_points = 100;

// Generiere Fake-Daten
$data = [];
for ($i = 0; $i < $num_data_points; $i++) {
    $timestamp = time() - ($num_data_points - $i) * 3600;  // Erstelle Zeitstempel in Stundenabständen
    $data[] = [
        'timestamp' => date('Y-m-d H:i:s', $timestamp),
        'learning_rate' => mt_rand(1, 100) / 1000,  // Random Learning Rate zwischen 0.001 und 0.1
        'accuracy' => mt_rand(70, 99) + mt_rand() / mt_getrandmax(),  // Random Accuracy zwischen 70 und 100
        'batch_size' => mt_rand(32, 256),  // Random Batch Size zwischen 32 und 256
    ];
}

// Rückgabe der Daten als JSON
echo json_encode($data);


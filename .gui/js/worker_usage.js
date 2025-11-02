function plotWorkerUsage() {
    var $plot = $("#workerUsagePlot");
    if ($plot.data("loaded") == "true") return;

    var data = tab_worker_usage_csv_json;
    if (!Array.isArray(data) || data.length === 0) {
        console.error("Invalid or empty data provided.");
        return;
    }

    var n = data.length;
    var timestamps = new Array(n);
    var desiredWorkers = new Array(n);
    var realWorkers = new Array(n);
    var used = 0;

    for (var i = 0; i < n; i++) {
        var entry = data[i];
        if (!Array.isArray(entry) || entry.length < 3) {
            console.warn("Skipping invalid entry:", entry);
            continue;
        }

        var unixTime = Number(entry[0]);
        var desired = Number(entry[1]);
        var real = Number(entry[2]);

        if (!isFinite(unixTime) || !isFinite(desired) || !isFinite(real)) {
            console.warn("Skipping invalid numerical values:", entry);
            continue;
        }

        timestamps[used] = unixTime * 1000; // <-- epoch ms number (faster)
        desiredWorkers[used] = desired;
        realWorkers[used] = real;
        used++;
    }

    if (used !== n) {
        timestamps.length = used;
        desiredWorkers.length = used;
        realWorkers.length = used;
    }

    var trace1 = {
        x: timestamps,
        y: desiredWorkers,
        mode: 'lines+markers',
        name: 'Desired Workers',
        line: { color: 'blue' }
    };

    var trace2 = {
        x: timestamps,
        y: realWorkers,
        mode: 'lines+markers',
        name: 'Real Workers',
        line: { color: 'red' }
    };

    var layout = {
        title: "Worker Usage Over Time",
        xaxis: { title: get_axis_title_data("Time", "date") },
        yaxis: { title: get_axis_title_data("Number of Workers") },
        legend: { x: 0, y: 1 }
    };

    Plotly.newPlot('workerUsagePlot', [trace1, trace2], add_default_layout_data(layout));
    $plot.data("loaded", "true");
}

(function (root, factory) {
	if (typeof module === "object" && module.exports) {
		module.exports = factory(root);
	} else {
		root.OmniOptLandscape = factory(root);
	}
})(typeof globalThis !== "undefined" ? globalThis : this, function (root) {
	"use strict";

	var DEFAULT_GRID_SIZE = 84;
	var MAX_PATH_POINTS = 4000;
	var MAX_MARKER_POINTS = 12000;
	var LOSS_COLORSCALE = [
		[0.0, "#0f766e"],
		[0.22, "#16a34a"],
		[0.46, "#facc15"],
		[0.72, "#f97316"],
		[1.0, "#dc2626"]
	];

	function trimString(value) {
		return value === null || value === undefined ? "" : String(value).trim();
	}

	function parseNumeric(value) {
		if (value === null || value === undefined) {
			return null;
		}

		if (typeof value === "number") {
			return isFinite(value) ? value : null;
		}

		var text = trimString(value);
		if (text === "") {
			return null;
		}

		var number = Number(text);
		return isFinite(number) ? number : null;
	}

	function findHeaderIndex(headers, name) {
		if (!name || !Array.isArray(headers)) {
			return -1;
		}

		var wanted = String(name).toLowerCase();
		for (var i = 0; i < headers.length; i++) {
			if (String(headers[i]).toLowerCase() === wanted) {
				return i;
			}
		}

		return -1;
	}

	function isIgnoredParameterColumn(name, resultNames, specialColumns) {
		var lower = String(name).toLowerCase();

		if (lower.indexOf("oo_info") === 0) {
			return true;
		}

		for (var i = 0; i < resultNames.length; i++) {
			if (String(resultNames[i]).toLowerCase() === lower) {
				return true;
			}
		}

		for (var j = 0; j < specialColumns.length; j++) {
			if (String(specialColumns[j]).toLowerCase() === lower) {
				return true;
			}
		}

		return false;
	}

	function getNumericParameterColumns(headers, rows, resultNames, specialColumns) {
		var resultNamesList = Array.isArray(resultNames) ? resultNames : [];
		var specialList = Array.isArray(specialColumns) ? specialColumns : [];
		var columns = [];

		if (!Array.isArray(headers) || !Array.isArray(rows)) {
			return columns;
		}

		for (var c = 0; c < headers.length; c++) {
			var name = headers[c];

			if (isIgnoredParameterColumn(name, resultNamesList, specialList)) {
				continue;
			}

			var numericCount = 0;
			var firstValue = null;
			var hasVariation = false;
			var isNumeric = true;

			for (var r = 0; r < rows.length; r++) {
				var raw = rows[r] ? rows[r][c] : null;

				if (trimString(raw) === "") {
					continue;
				}

				var numericValue = parseNumeric(raw);

				if (numericValue === null) {
					isNumeric = false;
					break;
				}

				if (firstValue === null) {
					firstValue = numericValue;
				} else if (numericValue !== firstValue) {
					hasVariation = true;
				}

				numericCount++;
			}

			if (isNumeric && numericCount >= 2 && hasVariation) {
				columns.push(name);
			}
		}

		return columns;
	}

	function getResultGoal(resultName, resultNames, resultMinMax) {
		var index = -1;

		for (var i = 0; i < resultNames.length; i++) {
			if (String(resultNames[i]).toLowerCase() === String(resultName).toLowerCase()) {
				index = i;
				break;
			}
		}

		if (index < 0) {
			return "min";
		}

		var goal = null;

		if (Array.isArray(resultMinMax)) {
			goal = resultMinMax[index];
		} else if (resultMinMax && typeof resultMinMax === "object") {
			goal = resultMinMax[resultName];
			if (goal && typeof goal === "object") {
				goal = goal[Object.keys(goal)[0]];
			}
		}

		return String(goal).toLowerCase() === "max" ? "max" : "min";
	}

	function getResultLabel(resultName, resultNames, resultMinMax) {
		return resultName + " (" + getResultGoal(resultName, resultNames, resultMinMax) + ")";
	}

	function getTrialIndex(headers, row, rowIndex) {
		var index = findHeaderIndex(headers, "trial_index");

		if (index >= 0 && row) {
			var parsed = parseNumeric(row[index]);
			if (parsed !== null) {
				return parsed;
			}
		}

		return rowIndex;
	}

	function isFailedTrial(headers, row) {
		var index = findHeaderIndex(headers, "trial_status");

		if (index < 0 || !row) {
			return false;
		}

		var status = trimString(row[index]).toUpperCase();
		return status === "FAILED" || status === "FAIL";
	}

	function getTrialIndexFromColumn(row, rowIndex, columnIndex) {
		if (columnIndex >= 0 && row) {
			var parsed = parseNumeric(row[columnIndex]);
			if (parsed !== null) {
				return parsed;
			}
		}

		return rowIndex;
	}

	function isFailedTrialFromColumn(row, columnIndex) {
		if (columnIndex < 0 || !row) {
			return false;
		}

		var status = trimString(row[columnIndex]).toUpperCase();
		return status === "FAILED" || status === "FAIL";
	}

	function collectLandscapeData(headers, rows, paramNames, resultName) {
		var resultIndex = findHeaderIndex(headers, resultName);
		var paramIndexes = [];
		var valid = [];
		var failed = [];

		for (var p = 0; p < paramNames.length; p++) {
			paramIndexes.push(findHeaderIndex(headers, paramNames[p]));
		}

		if (resultIndex < 0 || paramIndexes.length !== paramNames.length) {
			return { valid: valid, failed: failed };
		}

		for (var i = 0; i < paramIndexes.length; i++) {
			if (paramIndexes[i] < 0) {
				return { valid: valid, failed: failed };
			}
		}

		var trialIndexColumn = findHeaderIndex(headers, "trial_index");
		var statusColumn = findHeaderIndex(headers, "trial_status");

		for (var r = 0; r < rows.length; r++) {
			var row = rows[r];
			var values = [];
			var hasAllParams = true;

			for (var c = 0; c < paramIndexes.length; c++) {
				var value = parseNumeric(row[paramIndexes[c]]);

				if (value === null) {
					hasAllParams = false;
					break;
				}

				values.push(value);
			}

			if (!hasAllParams) {
				continue;
			}

			var trialIndex = getTrialIndexFromColumn(row, r, trialIndexColumn);
			var point = {
				values: values,
				loss: parseNumeric(row[resultIndex]),
				order: trialIndex,
				trial: trialIndex,
				rowIndex: r
			};

			if (isFailedTrialFromColumn(row, statusColumn)) {
				failed.push(point);
			} else if (point.loss !== null) {
				valid.push(point);
			}
		}

		valid.sort(function (left, right) {
			return left.order - right.order || left.rowIndex - right.rowIndex;
		});

		failed.sort(function (left, right) {
			return left.order - right.order || left.rowIndex - right.rowIndex;
		});

		return { valid: valid, failed: failed };
	}

	function collectLandscapeDataForResults(headers, rows, paramNames, resultNames) {
		var resultIndexes = [];
		var paramIndexes = [];
		var buckets = [];
		var failed = [];

		for (var r = 0; r < resultNames.length; r++) {
			resultIndexes.push(findHeaderIndex(headers, resultNames[r]));
			buckets.push({ valid: [], failed: [] });
		}

		for (var p = 0; p < paramNames.length; p++) {
			paramIndexes.push(findHeaderIndex(headers, paramNames[p]));
		}

		if (resultIndexes.some(function (index) { return index < 0; }) ||
			paramIndexes.some(function (index) { return index < 0; })) {
			return { buckets: buckets, failed: failed };
		}

		var trialIndexColumn = findHeaderIndex(headers, "trial_index");
		var statusColumn = findHeaderIndex(headers, "trial_status");

		for (var i = 0; i < rows.length; i++) {
			var row = rows[i];
			var values = [];
			var hasAllParams = true;

			for (var c = 0; c < paramIndexes.length; c++) {
				var value = parseNumeric(row[paramIndexes[c]]);

				if (value === null) {
					hasAllParams = false;
					break;
				}

				values.push(value);
			}

			if (!hasAllParams) {
				continue;
			}

			var trialIndex = getTrialIndexFromColumn(row, i, trialIndexColumn);
			var failedTrial = isFailedTrialFromColumn(row, statusColumn);

			if (failedTrial) {
				failed.push({
					values: values,
					loss: null,
					order: trialIndex,
					trial: trialIndex,
					rowIndex: i
				});
			}

			for (var b = 0; b < resultIndexes.length; b++) {
				var loss = parseNumeric(row[resultIndexes[b]]);

				if (!failedTrial && loss !== null) {
					buckets[b].valid.push({
						values: values,
						loss: loss,
						order: trialIndex,
						trial: trialIndex,
						rowIndex: i
					});
				}
			}
		}

		function sortPoints(left, right) {
			return left.order - right.order || left.rowIndex - right.rowIndex;
		}

		failed.sort(sortPoints);

		for (var k = 0; k < buckets.length; k++) {
			buckets[k].valid.sort(sortPoints);
			buckets[k].failed = failed;
		}

		return { buckets: buckets, failed: failed };
	}

	function findBestPoint(points, goal) {
		if (!points || !points.length) {
			return null;
		}

		var best = points[0];

		for (var i = 1; i < points.length; i++) {
			var value = points[i].loss;
			var bestValue = best.loss;

			if (goal === "max" ? value > bestValue : value < bestValue) {
				best = points[i];
			}
		}

		return best;
	}

	function sampleEvenly(values, maxCount) {
		if (!Array.isArray(values) || values.length <= maxCount) {
			return values;
		}

		var result = [];
		var step = (values.length - 1) / (maxCount - 1);

		for (var i = 0; i < maxCount; i++) {
			result.push(values[Math.round(i * step)]);
		}

		return result;
	}

	function computeExtent(points, valueIndex) {
		var min = Infinity;
		var max = -Infinity;

		for (var i = 0; i < points.length; i++) {
			var value = points[i].values[valueIndex];

			if (value < min) {
				min = value;
			}

			if (value > max) {
				max = value;
			}
		}

		return { min: min, max: max };
	}

	function padRange(min, max, ratio) {
		if (!(max > min)) {
			return null;
		}

		var padding = (max - min) * ratio;

		return {
			min: min - padding,
			max: max + padding
		};
	}

	function interpolate1D(points) {
		if (!Array.isArray(points) || points.length < 2) {
			return null;
		}

		var sorted = points.slice().sort(function (left, right) {
			return left.values[0] - right.values[0];
		});
		var extent = computeExtent(sorted, 0);
		var range = padRange(extent.min, extent.max, 0.04);

		if (!range) {
			return null;
		}

		var gridSize = Math.max(140, Math.min(420, sorted.length * 6));
		var neighborCount = Math.min(6, sorted.length);
		var sortedX = new Float64Array(sorted.length);
		var sortedZ = new Float64Array(sorted.length);

		for (var s = 0; s < sorted.length; s++) {
			sortedX[s] = sorted[s].values[0];
			sortedZ[s] = sorted[s].loss;
		}

		var gridX = new Array(gridSize);
		var gridZ = new Array(gridSize);
		var averageZ = 0;

		for (var a = 0; a < sorted.length; a++) {
			averageZ += sortedZ[a];
		}
		averageZ /= sorted.length;

		for (var g = 0; g < gridSize; g++) {
			var x = range.min + (range.max - range.min) * g / (gridSize - 1);
			var low = 0;
			var high = sortedX.length;

			while (low < high) {
				var mid = (low + high) >> 1;
				if (sortedX[mid] < x) {
					low = mid + 1;
				} else {
					high = mid;
				}
			}

			var left = low - 1;
			var right = low;
			var picked = [];
			var maxDistance = 0;

			while (picked.length < neighborCount && (left >= 0 || right < sortedX.length)) {
				var useLeft = right >= sortedX.length ||
					(left >= 0 && Math.abs(sortedX[left] - x) <= Math.abs(sortedX[right] - x));
				var index = useLeft ? left : right;

				if (useLeft) {
					left--;
				} else {
					right++;
				}

				var distance = Math.abs(sortedX[index] - x);
				if (distance > maxDistance) {
					maxDistance = distance;
				}
				picked.push(index);
			}

			var sigma = maxDistance > 0 ? maxDistance / 2.5 : 1e-9;
			var sumWeight = 0;
			var sumWeightedX = 0;
			var sumWeightedX2 = 0;
			var sumWeightedZ = 0;
			var sumWeightedXZ = 0;

			for (var k = 0; k < picked.length; k++) {
				var dx = sortedX[picked[k]] - x;
				var weight = Math.exp(-0.5 * (dx / sigma) * (dx / sigma));
				var weightedZ = weight * sortedZ[picked[k]];

				sumWeight += weight;
				sumWeightedX += weight * dx;
				sumWeightedX2 += weight * dx * dx;
				sumWeightedZ += weightedZ;
				sumWeightedXZ += weightedZ * dx;
			}

			var determinant = sumWeight * sumWeightedX2 - sumWeightedX * sumWeightedX;
			var localValue = averageZ;

			if (sumWeight > 0) {
				if (Math.abs(determinant) > 1e-12) {
					localValue = (sumWeightedZ * sumWeightedX2 - sumWeightedXZ * sumWeightedX) / determinant;
				} else {
					localValue = sumWeightedZ / sumWeight;
				}
			}

			gridX[g] = x;
			gridZ[g] = localValue;
		}

		return {
			x: gridX,
			z: gridZ,
			range: range
		};
	}

	function buildKernel(radius, sigma) {
		var size = radius * 2 + 1;
		var kernel = new Array(size);

		for (var y = 0; y < size; y++) {
			kernel[y] = new Array(size);

			for (var x = 0; x < size; x++) {
				var dx = x - radius;
				var dy = y - radius;
				kernel[y][x] = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
			}
		}

		return kernel;
	}

	function smoothGrid(grid, nx, ny, passes) {
		if (passes <= 0) {
			return;
		}

		var temp = new Float64Array(grid.length);

		for (var pass = 0; pass < passes; pass++) {
			for (var y = 0; y < ny; y++) {
				var rowOffset = y * nx;

				for (var x = 0; x < nx; x++) {
					var left = x > 0 ? grid[rowOffset + x - 1] : grid[rowOffset + x];
					var center = grid[rowOffset + x];
					var right = x < nx - 1 ? grid[rowOffset + x + 1] : grid[rowOffset + x];
					temp[rowOffset + x] = (left + 2 * center + right) / 4;
				}
			}

			for (var x2 = 0; x2 < nx; x2++) {
				for (var y2 = 0; y2 < ny; y2++) {
					var top = y2 > 0 ? temp[(y2 - 1) * nx + x2] : temp[y2 * nx + x2];
					var current = temp[y2 * nx + x2];
					var bottom = y2 < ny - 1 ? temp[(y2 + 1) * nx + x2] : temp[y2 * nx + x2];
					grid[y2 * nx + x2] = (top + 2 * current + bottom) / 4;
				}
			}
		}
	}

	function interpolate2D(points, gridSize, options) {
		if (!Array.isArray(points) || points.length < 3) {
			return null;
		}

		options = options || {};

		var maxGapCells = options.maxGapCells;
		if (maxGapCells === undefined) {
			maxGapCells = Infinity;
		}

		var failedPoints = Array.isArray(options.failedPoints) ? options.failedPoints : [];
		var failedRadiusCells = options.failedRadiusCells;
		if (failedRadiusCells === undefined) {
			failedRadiusCells = Math.max(1, Math.round((gridSize || DEFAULT_GRID_SIZE) * 0.03));
		}

		var extentX = computeExtent(points, 0);
		var extentY = computeExtent(points, 1);
		var rangeX = padRange(extentX.min, extentX.max, 0.04);
		var rangeY = padRange(extentY.min, extentY.max, 0.04);

		if (!rangeX || !rangeY) {
			return null;
		}

		var nx = gridSize || DEFAULT_GRID_SIZE;
		var ny = gridSize || DEFAULT_GRID_SIZE;
		var total = nx * ny;
		var weightedLoss = new Float64Array(total);
		var weightGrid = new Float64Array(total);
		var radius = 2;
		var kernel = buildKernel(radius, 1.2);

		for (var p = 0; p < points.length; p++) {
			var gx = (points[p].values[0] - rangeX.min) / (rangeX.max - rangeX.min) * (nx - 1);
			var gy = (points[p].values[1] - rangeY.min) / (rangeY.max - rangeY.min) * (ny - 1);
			var centerX = Math.round(gx);
			var centerY = Math.round(gy);

			for (var oy = -radius; oy <= radius; oy++) {
				var iy = centerY + oy;

				if (iy < 0 || iy >= ny) {
					continue;
				}

				for (var ox = -radius; ox <= radius; ox++) {
					var ix = centerX + ox;

					if (ix < 0 || ix >= nx) {
						continue;
					}

					var weight = kernel[oy + radius][ox + radius];
					var index = iy * nx + ix;
					weightedLoss[index] += points[p].loss * weight;
					weightGrid[index] += weight;
				}
			}
		}

		var grid = new Float64Array(total);
		var filled = new Uint8Array(total);
		var distance = new Int32Array(total);
		var queue = new Int32Array(total);
		var head = 0;
		var tail = 0;

		for (var i = 0; i < total; i++) {
			distance[i] = -1;

			if (weightGrid[i] > 1e-12) {
				grid[i] = weightedLoss[i] / weightGrid[i];
				filled[i] = 1;
				distance[i] = 0;
				queue[tail++] = i;
			}
		}

		if (tail === 0) {
			return null;
		}

		while (head < tail) {
			var index2 = queue[head++];
			var x2 = index2 % nx;
			var y2 = (index2 / nx) | 0;
			var neighbors = [
				x2 > 0 ? index2 - 1 : -1,
				x2 < nx - 1 ? index2 + 1 : -1,
				y2 > 0 ? index2 - nx : -1,
				y2 < ny - 1 ? index2 + nx : -1
			];

			for (var n = 0; n < neighbors.length; n++) {
				var neighbor = neighbors[n];

				if (neighbor >= 0 && !filled[neighbor]) {
					filled[neighbor] = 1;
					grid[neighbor] = grid[index2];
					distance[neighbor] = distance[index2] + 1;
					queue[tail++] = neighbor;
				}
			}
		}

		smoothGrid(grid, nx, ny, 2);

		if (isFinite(maxGapCells)) {
			for (var gapIndex = 0; gapIndex < total; gapIndex++) {
				if (distance[gapIndex] > maxGapCells) {
					grid[gapIndex] = NaN;
				}
			}
		}

		for (var failedIndex = 0; failedIndex < failedPoints.length; failedIndex++) {
			var failedPoint = failedPoints[failedIndex];
			var failedGridX = (failedPoint.values[0] - rangeX.min) / (rangeX.max - rangeX.min) * (nx - 1);
			var failedGridY = (failedPoint.values[1] - rangeY.min) / (rangeY.max - rangeY.min) * (ny - 1);
			var failedCenterX = Math.round(failedGridX);
			var failedCenterY = Math.round(failedGridY);

			for (var holeY = -failedRadiusCells; holeY <= failedRadiusCells; holeY++) {
				var holeGridY = failedCenterY + holeY;

				if (holeGridY < 0 || holeGridY >= ny) {
					continue;
				}

				for (var holeX = -failedRadiusCells; holeX <= failedRadiusCells; holeX++) {
					var holeGridX = failedCenterX + holeX;

					if (holeGridX < 0 || holeGridX >= nx) {
						continue;
					}

					if (holeX * holeX + holeY * holeY <= failedRadiusCells * failedRadiusCells) {
						grid[holeGridY * nx + holeGridX] = NaN;
					}
				}
			}
		}

		var xValues = new Array(nx);
		var yValues = new Array(ny);
		var zRows = new Array(ny);

		for (var xi = 0; xi < nx; xi++) {
			xValues[xi] = rangeX.min + (rangeX.max - rangeX.min) * xi / (nx - 1);
		}

		for (var yi = 0; yi < ny; yi++) {
			yValues[yi] = rangeY.min + (rangeY.max - rangeY.min) * yi / (ny - 1);
			zRows[yi] = new Array(nx);

			for (var xi2 = 0; xi2 < nx; xi2++) {
				zRows[yi][xi2] = grid[yi * nx + xi2];
			}
		}

		return {
			x: xValues,
			y: yValues,
			z: zRows,
			rangeX: rangeX,
			rangeY: rangeY
		};
	}

	function surfaceHeightAt(interpolation, x, y) {
		if (!interpolation || !isFinite(x) || !isFinite(y)) {
			return null;
		}

		var nx = interpolation.x.length;
		var ny = interpolation.y.length;

		if (nx < 2 || ny < 2) {
			return null;
		}

		var fx = (x - interpolation.x[0]) / (interpolation.x[nx - 1] - interpolation.x[0]) * (nx - 1);
		var fy = (y - interpolation.y[0]) / (interpolation.y[ny - 1] - interpolation.y[0]) * (ny - 1);

		if (fx < 0 || fy < 0 || fx > nx - 1 || fy > ny - 1) {
			return null;
		}

		var ix = Math.floor(fx);
		var iy = Math.floor(fy);

		if (ix >= nx - 1) {
			ix = nx - 2;
		}

		if (iy >= ny - 1) {
			iy = ny - 2;
		}

		var tx = fx - ix;
		var ty = fy - iy;
		var z00 = interpolation.z[iy][ix];
		var z10 = interpolation.z[iy][ix + 1];
		var z01 = interpolation.z[iy + 1][ix];
		var z11 = interpolation.z[iy + 1][ix + 1];

		if (!isFinite(z00) || !isFinite(z10) || !isFinite(z01) || !isFinite(z11)) {
			return null;
		}

		var top = z00 * (1 - tx) + z10 * tx;
		var bottom = z01 * (1 - tx) + z11 * tx;
		return top * (1 - ty) + bottom * ty;
	}

	function interpolate1DHeightAt(interpolation, x) {
		if (!interpolation || !isFinite(x) || interpolation.x.length < 2) {
			return null;
		}

		var nx = interpolation.x.length;

		if (x < interpolation.x[0] || x > interpolation.x[nx - 1]) {
			return null;
		}

		var fx = (x - interpolation.x[0]) / (interpolation.x[nx - 1] - interpolation.x[0]) * (nx - 1);
		var ix = Math.floor(fx);

		if (ix >= nx - 1) {
			ix = nx - 2;
		}

		var tx = fx - ix;
		return interpolation.z[ix] * (1 - tx) + interpolation.z[ix + 1] * tx;
	}

	function buildKernel3D(radius, sigma) {
		var size = radius * 2 + 1;
		var kernel = new Float64Array(size * size * size);
		var sigmaSquared = 2 * sigma * sigma;

		for (var z = 0; z < size; z++) {
			for (var y = 0; y < size; y++) {
				for (var x = 0; x < size; x++) {
					var dx = x - radius;
					var dy = y - radius;
					var dz = z - radius;
					kernel[(z * size + y) * size + x] = Math.exp(-(dx * dx + dy * dy + dz * dz) / sigmaSquared);
				}
			}
		}

		return { kernel: kernel, size: size, radius: radius };
	}

	function smoothGrid3D(grid, nx, ny, nz, passes) {
		if (passes <= 0) {
			return;
		}

		var total = nx * ny * nz;
		var temp = new Float64Array(total);

		function blurX(source, target) {
			for (var z = 0; z < nz; z++) {
				for (var y = 0; y < ny; y++) {
					var rowOffset = (z * ny + y) * nx;

					for (var x = 0; x < nx; x++) {
						var left = x > 0 ? source[rowOffset + x - 1] : source[rowOffset + x];
						var center = source[rowOffset + x];
						var right = x < nx - 1 ? source[rowOffset + x + 1] : source[rowOffset + x];
						target[rowOffset + x] = (left + 2 * center + right) / 4;
					}
				}
			}
		}

		function blurY(source, target) {
			for (var z = 0; z < nz; z++) {
				for (var y = 0; y < ny; y++) {
					for (var x = 0; x < nx; x++) {
						var top = y > 0 ? source[(z * ny + y - 1) * nx + x] : source[(z * ny + y) * nx + x];
						var center = source[(z * ny + y) * nx + x];
						var bottom = y < ny - 1 ? source[(z * ny + y + 1) * nx + x] : source[(z * ny + y) * nx + x];
						target[(z * ny + y) * nx + x] = (top + 2 * center + bottom) / 4;
					}
				}
			}
		}

		function blurZ(source, target) {
			for (var z = 0; z < nz; z++) {
				for (var y = 0; y < ny; y++) {
					for (var x = 0; x < nx; x++) {
						var top = z > 0 ? source[((z - 1) * ny + y) * nx + x] : source[(z * ny + y) * nx + x];
						var center = source[(z * ny + y) * nx + x];
						var bottom = z < nz - 1 ? source[((z + 1) * ny + y) * nx + x] : source[(z * ny + y) * nx + x];
						target[(z * ny + y) * nx + x] = (top + 2 * center + bottom) / 4;
					}
				}
			}
		}

		for (var pass = 0; pass < passes; pass++) {
			blurX(grid, temp);
			blurY(temp, grid);
			blurZ(grid, temp);
			grid.set(temp);
		}
	}

	function interpolate3D(points, gridSize) {
		if (!Array.isArray(points) || points.length < 4) {
			return null;
		}

		var extentX = computeExtent(points, 0);
		var extentY = computeExtent(points, 1);
		var extentZ = computeExtent(points, 2);
		var rangeX = padRange(extentX.min, extentX.max, 0.05);
		var rangeY = padRange(extentY.min, extentY.max, 0.05);
		var rangeZ = padRange(extentZ.min, extentZ.max, 0.05);

		if (!rangeX || !rangeY || !rangeZ) {
			return null;
		}

		var size = gridSize || Math.max(14, Math.min(26, Math.round(Math.cbrt(points.length) * 4)));
		var nx = size;
		var ny = size;
		var nz = size;
		var total = nx * ny * nz;
		var weightedLoss = new Float64Array(total);
		var weightGrid = new Float64Array(total);
		var kernelData = buildKernel3D(2, 1.2);
		var kernel = kernelData.kernel;
		var kernelSize = kernelData.size;
		var radius = kernelData.radius;

		for (var p = 0; p < points.length; p++) {
			var gx = (points[p].values[0] - rangeX.min) / (rangeX.max - rangeX.min) * (nx - 1);
			var gy = (points[p].values[1] - rangeY.min) / (rangeY.max - rangeY.min) * (ny - 1);
			var gz = (points[p].values[2] - rangeZ.min) / (rangeZ.max - rangeZ.min) * (nz - 1);
			var centerX = Math.round(gx);
			var centerY = Math.round(gy);
			var centerZ = Math.round(gz);

			for (var oz = -radius; oz <= radius; oz++) {
				var iz = centerZ + oz;

				if (iz < 0 || iz >= nz) {
					continue;
				}

				for (var oy = -radius; oy <= radius; oy++) {
					var iy = centerY + oy;

					if (iy < 0 || iy >= ny) {
						continue;
					}

					for (var ox = -radius; ox <= radius; ox++) {
						var ix = centerX + ox;

						if (ix < 0 || ix >= nx) {
							continue;
						}

						var weight = kernel[((oz + radius) * kernelSize + (oy + radius)) * kernelSize + (ox + radius)];
						var index = (iz * ny + iy) * nx + ix;
						weightedLoss[index] += points[p].loss * weight;
						weightGrid[index] += weight;
					}
				}
			}
		}

		var grid = new Float64Array(total);
		var filled = new Uint8Array(total);
		var queue = new Int32Array(total);
		var head = 0;
		var tail = 0;

		for (var i = 0; i < total; i++) {
			if (weightGrid[i] > 1e-12) {
				grid[i] = weightedLoss[i] / weightGrid[i];
				filled[i] = 1;
				queue[tail++] = i;
			}
		}

		if (tail === 0) {
			return null;
		}

		var planeSize = nx * ny;

		while (head < tail) {
			var index2 = queue[head++];
			var ix2 = index2 % nx;
			var iy2 = ((index2 / nx) | 0) % ny;
			var iz2 = (index2 / planeSize) | 0;

			if (ix2 > 0 && !filled[index2 - 1]) {
				filled[index2 - 1] = 1;
				grid[index2 - 1] = grid[index2];
				queue[tail++] = index2 - 1;
			}

			if (ix2 < nx - 1 && !filled[index2 + 1]) {
				filled[index2 + 1] = 1;
				grid[index2 + 1] = grid[index2];
				queue[tail++] = index2 + 1;
			}

			if (iy2 > 0 && !filled[index2 - nx]) {
				filled[index2 - nx] = 1;
				grid[index2 - nx] = grid[index2];
				queue[tail++] = index2 - nx;
			}

			if (iy2 < ny - 1 && !filled[index2 + nx]) {
				filled[index2 + nx] = 1;
				grid[index2 + nx] = grid[index2];
				queue[tail++] = index2 + nx;
			}

			if (iz2 > 0 && !filled[index2 - planeSize]) {
				filled[index2 - planeSize] = 1;
				grid[index2 - planeSize] = grid[index2];
				queue[tail++] = index2 - planeSize;
			}

			if (iz2 < nz - 1 && !filled[index2 + planeSize]) {
				filled[index2 + planeSize] = 1;
				grid[index2 + planeSize] = grid[index2];
				queue[tail++] = index2 + planeSize;
			}
		}

		smoothGrid3D(grid, nx, ny, nz, 2);

		var xValues = new Array(nx);
		var yValues = new Array(ny);
		var zValues = new Array(nz);

		for (var xi = 0; xi < nx; xi++) {
			xValues[xi] = rangeX.min + (rangeX.max - rangeX.min) * xi / (nx - 1);
		}

		for (var yi = 0; yi < ny; yi++) {
			yValues[yi] = rangeY.min + (rangeY.max - rangeY.min) * yi / (ny - 1);
		}

		for (var zi = 0; zi < nz; zi++) {
			zValues[zi] = rangeZ.min + (rangeZ.max - rangeZ.min) * zi / (nz - 1);
		}

		// Plotly's isosurface trace expects four equally long 1-D arrays with
		// the x, y, z coordinates and the value of every grid vertex.
		var vertexCount = nx * ny * nz;
		var xFlat = new Array(vertexCount);
		var yFlat = new Array(vertexCount);
		var zFlat = new Array(vertexCount);
		var valueFlat = new Array(vertexCount);
		var vertexIndex = 0;

		for (var zIndex = 0; zIndex < nz; zIndex++) {
			for (var yIndex = 0; yIndex < ny; yIndex++) {
				for (var xIndex = 0; xIndex < nx; xIndex++) {
					xFlat[vertexIndex] = xValues[xIndex];
					yFlat[vertexIndex] = yValues[yIndex];
					zFlat[vertexIndex] = zValues[zIndex];
					valueFlat[vertexIndex] = grid[(zIndex * ny + yIndex) * nx + xIndex];
					vertexIndex++;
				}
			}
		}

		return {
			x: xFlat,
			y: yFlat,
			z: zFlat,
			value: valueFlat,
			xValues: xValues,
			yValues: yValues,
			zValues: zValues,
			rangeX: rangeX,
			rangeY: rangeY,
			rangeZ: rangeZ
		};
	}

	function reverseColorscale(scale) {
		return scale.map(function (stop) {
			return [1 - stop[0], stop[1]];
		}).sort(function (left, right) {
			return left[0] - right[0];
		});
	}

	function getLossColorscale(goal) {
		return goal === "max" ? reverseColorscale(LOSS_COLORSCALE) : LOSS_COLORSCALE.slice();
	}

	function formatNumber(value) {
		if (value === null || value === undefined || !isFinite(value)) {
			return "–";
		}

		var absolute = Math.abs(value);

		if (absolute !== 0 && (absolute < 1e-3 || absolute >= 1e6)) {
			return value.toExponential(3);
		}

		return String(Number(value.toPrecision(6)));
	}

	function getTheme() {
		var dark = String(root && root.theme).toLowerCase() === "dark";

		return {
			dark: dark,
			text: dark ? "#f8fafc" : "#1f2937",
			muted: dark ? "#cbd5e1" : "#475569",
			grid: dark ? "rgba(148, 163, 184, 0.28)" : "rgba(100, 116, 139, 0.24)",
			axisLine: dark ? "rgba(226, 232, 240, 0.55)" : "rgba(71, 85, 105, 0.55)",
			path: dark ? "#f472b6" : "#db2777",
			start: "#22c55e",
			end: "#f59e0b",
			best: "#3b82f6",
			failed: "#ef4444"
		};
	}

	function axisTitle(text, colors) {
		return {
			text: text,
			font: {
				color: colors.text,
				size: 13
			}
		};
	}

	function sceneAxis(text, colors) {
		return {
			title: axisTitle(text, colors),
			color: colors.muted,
			gridcolor: colors.grid,
			zerolinecolor: colors.grid,
			linecolor: colors.axisLine,
			backgroundcolor: "rgba(0,0,0,0)",
			showbackground: false,
			tickfont: {
				color: colors.muted
			}
		};
	}

	function baseLayout(colors, height, mode) {
		var layout = {
			autosize: true,
			height: height,
			paper_bgcolor: "rgba(0,0,0,0)",
			plot_bgcolor: "rgba(0,0,0,0)",
			font: {
				color: colors.text,
				size: 13
			},
			margin: mode === 1 ? { l: 66, r: 20, t: 48, b: 56 } : { l: 8, r: 8, t: 42, b: 8 },
			showlegend: true,
			legend: {
				orientation: "h",
				x: 0.5,
				xanchor: "center",
				y: 1.02,
				yanchor: "bottom",
				font: {
					color: colors.muted,
					size: 11
				},
				bgcolor: "rgba(0,0,0,0)"
			},
			hoverlabel: {
				bgcolor: colors.dark ? "#111827" : "#ffffff",
				bordercolor: colors.axisLine,
				font: {
					color: colors.text
				}
			}
		};

		return layout;
	}

	function pathPoints(points, maxCount) {
		var sampled = sampleEvenly(points, maxCount);
		var values = [];

		for (var i = 0; i < sampled.length; i++) {
			values.push(sampled[i]);
		}

		return values;
	}

	function markerTrace(colors, name, symbol, color, size, points, dimension, zValue) {
		var trace = {
			type: dimension >= 2 ? "scatter3d" : "scatter",
			mode: "markers",
			name: name,
			marker: {
				symbol: symbol,
				color: color,
				size: size,
				line: {
					color: colors.dark ? "#111827" : "#ffffff",
					width: 1
				}
			},
			hoverinfo: "skip",
			showlegend: true
		};

		if (dimension >= 3) {
			trace.x = [points.values[0]];
			trace.y = [points.values[1]];
			trace.z = [points.values[2]];
		} else if (dimension === 2) {
			trace.x = [points.values[0]];
			trace.y = [points.values[1]];
			trace.z = [zValue];
		} else {
			trace.x = [points.values[0]];
			trace.y = [zValue];
		}

		return trace;
	}

	function surfaceInterpolationOptions(valid, failed) {
		var gridSize = DEFAULT_GRID_SIZE;
		var extentX = computeExtent(valid, 0);
		var extentY = computeExtent(valid, 1);
		var rangeX = Math.max(1e-9, extentX.max - extentX.min);
		var rangeY = Math.max(1e-9, extentY.max - extentY.min);
		var area = rangeX * rangeY;
		var spacing = Math.sqrt(area / Math.max(1, valid.length));
		var cellSize = (rangeX / (gridSize - 1) + rangeY / (gridSize - 1)) / 2;
		var maxGapCells = Math.max(3, Math.round((spacing / cellSize) * 1.1));
		maxGapCells = Math.min(maxGapCells, Math.round(gridSize * 0.18));

		return {
			gridSize: gridSize,
			maxGapCells: maxGapCells,
			failedPoints: failed,
			failedRadiusCells: Math.max(2, Math.round(maxGapCells * 0.45))
		};
	}

	function markerTrace2D(colors, name, symbol, color, size, point) {
		return {
			type: "scatter",
			mode: "markers",
			name: name,
			x: [point.values[0]],
			y: [point.values[1]],
			marker: {
				symbol: symbol,
				color: color,
				size: size,
				line: {
					color: colors.dark ? "#111827" : "#ffffff",
					width: 1
				}
			},
			hoverinfo: "skip",
			showlegend: true
		};
	}

	function markerTrace3DAt(colors, name, symbol, color, size, point, z) {
		return {
			type: "scatter3d",
			mode: "markers",
			name: name,
			x: [point.values[0]],
			y: [point.values[1]],
			z: [z],
			marker: {
				symbol: symbol,
				color: color,
				size: size,
				line: {
					color: colors.dark ? "#111827" : "#ffffff",
					width: 1
				}
			},
			hoverinfo: "skip",
			showlegend: true
		};
	}

	function sortedQuantile(sortedValues, fraction) {
		if (!sortedValues.length) {
			return null;
		}

		var index = Math.max(0, Math.min(sortedValues.length - 1, Math.floor(fraction * (sortedValues.length - 1))));
		return sortedValues[index];
	}

	function filterDataBySlice(data, slice) {
		if (!data || !slice || slice === "all") {
			return data;
		}

		var zValues = [];

		for (var i = 0; i < data.valid.length; i++) {
			zValues.push(data.valid[i].values[2]);
		}

		if (!zValues.length) {
			return data;
		}

		zValues.sort(function (left, right) { return left - right; });

		var low = sortedQuantile(zValues, 1 / 3);
		var high = sortedQuantile(zValues, 2 / 3);

		function keep(point) {
			var value = point.values[2];

			if (slice === "low") {
				return value <= low;
			}

			if (slice === "middle") {
				return value > low && value <= high;
			}

			if (slice === "high") {
				return value > high;
			}

			return true;
		}

		return {
			valid: data.valid.filter(keep),
			failed: data.failed.filter(keep)
		};
	}

	function render1D(container, data, context) {
		var colors = context.colors;
		var valid = data.valid;
		var failed = data.failed;
		var interpolation = interpolate1D(valid);

		if (!interpolation) {
			showPlotMessage(container, "Not enough data to build a 1D result landscape.");
			return;
		}

		var resultName = context.resultName;
		var resultLabel = context.resultLabel || resultName;
		var colorscale = getLossColorscale(context.goal);
		var lossMin = Infinity;
		var lossMax = -Infinity;

		for (var i = 0; i < valid.length; i++) {
			lossMin = Math.min(lossMin, valid[i].loss);
			lossMax = Math.max(lossMax, valid[i].loss);
		}

		var lossRange = lossMax - lossMin;
		var baseline = lossMin - (lossRange > 0 ? lossRange * 0.12 : 1);
		var path = pathPoints(valid, MAX_PATH_POINTS);
		var best = findBestPoint(valid, context.goal);
		var traces = [];

		function curveHeightForPoint(point) {
			var curveHeight = interpolate1DHeightAt(interpolation, point.values[0]);
			return curveHeight === null ? point.loss : curveHeight;
		}

		traces.push({
			type: "scatter",
			mode: "lines",
			name: "Result landscape",
			x: interpolation.x,
			y: interpolation.z,
			line: {
				color: colors.muted,
				width: 3
			},
			hoverinfo: "skip"
		});

		traces.push({
			type: "scatter",
			mode: "markers",
			name: "Observed trials",
			x: valid.map(function (point) { return point.values[0]; }),
			y: valid.map(function (point) { return point.loss; }),
			marker: {
				size: 5,
				color: valid.map(function (point) { return point.loss; }),
				colorscale: colorscale,
				showscale: true,
				colorbar: {
					title: resultLabel,
					thickness: 12,
					len: 0.62,
					outlinewidth: 0,
					tickfont: {
						color: colors.muted,
						size: 10
					},
					titlefont: {
						color: colors.text,
						size: 11
					}
				},
				line: {
					color: colors.dark ? "#111827" : "#ffffff",
					width: 0.5
				}
			},
			customdata: valid.map(function (point) { return point.trial; }),
			hovertemplate: "trial %{customdata}<br>" + resultLabel + ": %{y:.6g}<extra></extra>",
			showlegend: false
		});

		traces.push({
			type: "scatter",
			mode: "lines",
			name: "Training path",
			x: path.map(function (point) { return point.values[0]; }),
			y: path.map(curveHeightForPoint),
			line: {
				color: colors.path,
				width: 2
			},
			opacity: 0.8,
			hoverinfo: "skip"
		});

		traces.push(markerTrace(colors, "Start", "circle", colors.start, 11, valid[0], 1, curveHeightForPoint(valid[0])));
		traces.push(markerTrace(colors, "End", "square", colors.end, 10, valid[valid.length - 1], 1, curveHeightForPoint(valid[valid.length - 1])));

		if (best) {
			traces.push(markerTrace(colors, "Best", "star", colors.best, 13, best, 1, curveHeightForPoint(best)));
		}

		if (context.showFailed && failed.length) {
			var failedPoints = sampleEvenly(failed, MAX_MARKER_POINTS);
			traces.push({
				type: "scatter",
				mode: "markers",
				name: "Failed (no result)",
				x: failedPoints.map(function (point) { return point.values[0]; }),
				y: failedPoints.map(function () { return baseline; }),
				marker: {
					symbol: "x",
					color: colors.failed,
					size: 10,
					line: {
						color: colors.failed,
						width: 2
					}
				},
				customdata: failedPoints.map(function (point) { return point.trial; }),
				hovertemplate: "failed trial %{customdata}<br>no result value<extra></extra>",
				showlegend: true
			});
		}

		var layout = baseLayout(colors, 440, 1);
		layout.xaxis = {
			title: axisTitle(context.paramNames[0], colors),
			color: colors.muted,
			gridcolor: colors.grid,
			zerolinecolor: colors.grid,
			linecolor: colors.axisLine,
			tickfont: {
				color: colors.muted
			}
		};
		layout.yaxis = {
			title: axisTitle(resultLabel, colors),
			color: colors.muted,
			gridcolor: colors.grid,
			zerolinecolor: colors.grid,
			linecolor: colors.axisLine,
			tickfont: {
				color: colors.muted
			}
		};

		plotlyReact(container, traces, layout);
	}

	function render2D(container, data, context) {
		var colors = context.colors;
		var valid = data.valid;
		var failed = data.failed;
		var surfaceOptions = surfaceInterpolationOptions(valid, failed);
		var interpolation = interpolate2D(valid, surfaceOptions.gridSize, surfaceOptions);

		if (!interpolation) {
			showPlotMessage(container, "Not enough data to build a 2D result contour.");
			return;
		}

		var resultName = context.resultName;
		var resultLabel = context.resultLabel || resultName;
		var colorscale = getLossColorscale(context.goal);
		var lossMin = Infinity;
		var lossMax = -Infinity;

		for (var i = 0; i < valid.length; i++) {
			lossMin = Math.min(lossMin, valid[i].loss);
			lossMax = Math.max(lossMax, valid[i].loss);
		}

		var path = pathPoints(valid, MAX_PATH_POINTS);
		var best = findBestPoint(valid, context.goal);
		var traces = [];

		traces.push({
			type: "contour",
			name: "Result contour",
			x: interpolation.x,
			y: interpolation.y,
			z: interpolation.z,
			colorscale: colorscale,
			cmin: lossMin,
			cmax: lossMax,
			showscale: true,
			colorbar: {
				title: resultLabel,
				thickness: 12,
				len: 0.62,
				outlinewidth: 0,
				tickfont: {
					color: colors.muted,
					size: 10
				},
				titlefont: {
					color: colors.text,
					size: 11
				}
			},
			contours: {
				coloring: "heatmap",
				showlines: true,
				showlabels: false
			},
			line: {
				width: 0.5,
				color: colors.dark ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.22)"
			},
			hovertemplate: context.paramNames[0] + ": %{x:.5g}<br>" +
				context.paramNames[1] + ": %{y:.5g}<br>" +
				resultLabel + ": %{z:.5g}<extra></extra>"
		});

		traces.push({
			type: "scatter",
			mode: "lines",
			name: "Training path",
			x: path.map(function (point) { return point.values[0]; }),
			y: path.map(function (point) { return point.values[1]; }),
			line: {
				color: colors.path,
				width: 2.5
			},
			opacity: 0.9,
			hoverinfo: "skip"
		});

		traces.push(markerTrace2D(colors, "Start", "circle", colors.start, 9, valid[0]));
		traces.push(markerTrace2D(colors, "End", "square", colors.end, 9, valid[valid.length - 1]));

		if (best) {
			traces.push(markerTrace2D(colors, "Best", "star", colors.best, 12, best));
		}

		if (context.showFailed && failed.length) {
			var failedPoints = sampleEvenly(failed, MAX_MARKER_POINTS);
			traces.push({
				type: "scatter",
				mode: "markers",
				name: "Failed (hole)",
				x: failedPoints.map(function (point) { return point.values[0]; }),
				y: failedPoints.map(function (point) { return point.values[1]; }),
				marker: {
					symbol: "x",
					color: colors.failed,
					size: 9,
					line: {
						color: colors.failed,
						width: 2
					}
				},
				customdata: failedPoints.map(function (point) { return point.trial; }),
				hovertemplate: "failed trial %{customdata}<br>topological hole, no result value<extra></extra>",
				showlegend: true
			});
		}

		var layout = baseLayout(colors, 620, 1);
		layout.xaxis = {
			title: axisTitle(context.paramNames[0], colors),
			color: colors.muted,
			gridcolor: colors.grid,
			zerolinecolor: colors.grid,
			linecolor: colors.axisLine,
			tickfont: {
				color: colors.muted
			}
		};
		layout.yaxis = {
			title: axisTitle(context.paramNames[1], colors),
			color: colors.muted,
			gridcolor: colors.grid,
			zerolinecolor: colors.grid,
			linecolor: colors.axisLine,
			tickfont: {
				color: colors.muted
			}
		};

		plotlyReact(container, traces, layout);
	}

	function render3D(container, data, context) {
		var colors = context.colors;
		var valid = data.valid;
		var failed = data.failed;

		if (!valid.length) {
			showPlotMessage(container, "No valid trials for a 3D result surface.");
			return;
		}

		var surfaceOptions = surfaceInterpolationOptions(valid, failed);
		var interpolation = interpolate2D(valid, surfaceOptions.gridSize, surfaceOptions);

		if (!interpolation) {
			showPlotMessage(container, "Not enough data to build a 3D result surface.");
			return;
		}

		var resultName = context.resultName;
		var resultLabel = context.resultLabel || resultName;
		var colorscale = getLossColorscale(context.goal);
		var lossMin = Infinity;
		var lossMax = -Infinity;

		for (var i = 0; i < valid.length; i++) {
			lossMin = Math.min(lossMin, valid[i].loss);
			lossMax = Math.max(lossMax, valid[i].loss);
		}

		var lossRange = lossMax - lossMin;
		var surfaceLineOffset = lossRange > 0 ? lossRange * 0.004 : 0.001;
		var path = pathPoints(valid, MAX_PATH_POINTS);
		var best = findBestPoint(valid, context.goal);
		var traces = [];

		traces.push({
			type: "surface",
			name: "Result surface",
			x: interpolation.x,
			y: interpolation.y,
			z: interpolation.z,
			colorscale: colorscale,
			cmin: lossMin,
			cmax: lossMax,
			showscale: true,
			colorbar: {
				title: resultLabel,
				thickness: 12,
				len: 0.62,
				outlinewidth: 0,
				tickfont: {
					color: colors.muted,
					size: 10
				},
				titlefont: {
					color: colors.text,
					size: 11
				}
			},
			contours: {
				z: {
					show: true,
					usecolormap: true
				}
			},
			opacity: 0.96,
			lighting: {
				ambient: 0.68,
				diffuse: 0.8,
				specular: 0.15,
				roughness: 0.9
			},
			hovertemplate: context.paramNames[0] + ": %{x:.5g}<br>" +
				context.paramNames[1] + ": %{y:.5g}<br>" +
				resultLabel + ": %{z:.5g}<extra></extra>"
		});

		var segments = [];
		var currentSegment = [];

		for (var p = 0; p < path.length; p++) {
			var height = surfaceHeightAt(interpolation, path[p].values[0], path[p].values[1]);

			if (height === null) {
				if (currentSegment.length) {
					segments.push(currentSegment);
					currentSegment = [];
				}
			} else {
				currentSegment.push({
					point: path[p],
					height: height
				});
			}
		}

		if (currentSegment.length) {
			segments.push(currentSegment);
		}

		for (var s = 0; s < segments.length && s < 40; s++) {
			var segment = segments[s];

			if (segment.length < 2) {
				continue;
			}

			traces.push({
				type: "scatter3d",
				mode: "lines",
				name: "Training path",
				x: segment.map(function (item) { return item.point.values[0]; }),
				y: segment.map(function (item) { return item.point.values[1]; }),
				z: segment.map(function (item) { return item.height + surfaceLineOffset; }),
				line: {
					color: colors.path,
					width: 4
				},
				opacity: 0.95,
				showlegend: s === 0,
				hoverinfo: "skip"
			});
		}

		function heightFor(point) {
			var surfaceHeight = surfaceHeightAt(interpolation, point.values[0], point.values[1]);
			return (surfaceHeight === null ? point.loss : surfaceHeight) + surfaceLineOffset;
		}

		traces.push(markerTrace3DAt(colors, "Start", "circle", colors.start, 8, valid[0], heightFor(valid[0])));
		traces.push(markerTrace3DAt(colors, "End", "square", colors.end, 8, valid[valid.length - 1], heightFor(valid[valid.length - 1])));

		if (best) {
			traces.push(markerTrace3DAt(colors, "Best", "diamond", colors.best, 10, best, heightFor(best)));
		}

		if (context.showPoints) {
			var scatterPoints = sampleEvenly(valid, MAX_MARKER_POINTS);

			traces.push({
				type: "scatter3d",
				mode: "markers",
				name: "Trials",
				x: scatterPoints.map(function (point) { return point.values[0]; }),
				y: scatterPoints.map(function (point) { return point.values[1]; }),
				z: scatterPoints.map(function (point) { return point.loss; }),
				marker: {
					size: 3,
					color: scatterPoints.map(function (point) { return point.loss; }),
					colorscale: colorscale,
					showscale: false,
					opacity: 0.45,
					line: {
						color: colors.dark ? "#111827" : "#ffffff",
						width: 0
					}
				},
				customdata: scatterPoints.map(function (point) {
					return [point.trial, point.loss];
				}),
				hovertemplate: "trial %{customdata[0]}<br>" + resultLabel + ": %{customdata[1]:.6g}<extra></extra>"
			});
		}

		if (context.showFailed && failed.length) {
			var failedPoints = sampleEvenly(failed, MAX_MARKER_POINTS);
			var holeBase = lossMin - (lossRange > 0 ? lossRange * 0.15 : 1);

			traces.push({
				type: "scatter3d",
				mode: "markers",
				name: "Failed (hole)",
				x: failedPoints.map(function (point) { return point.values[0]; }),
				y: failedPoints.map(function (point) { return point.values[1]; }),
				z: failedPoints.map(function () { return holeBase; }),
				marker: {
					symbol: "x",
					color: colors.failed,
					size: 8,
					line: {
						color: colors.failed,
						width: 2
					}
				},
				customdata: failedPoints.map(function (point) { return point.trial; }),
				hovertemplate: "failed trial %{customdata}<br>topological hole, no result value<extra></extra>",
				showlegend: true
			});
		}

		var layout = baseLayout(colors, 680, 3);
		layout.scene = {
			xaxis: sceneAxis(context.paramNames[0], colors),
			yaxis: sceneAxis(context.paramNames[1], colors),
			zaxis: sceneAxis(resultLabel, colors),
			aspectmode: "auto",
			camera: {
				eye: { x: 1.55, y: 1.55, z: 1.15 }
			}
		};

		plotlyReact(container, traces, layout);
	}

	function plotlyReact(container, traces, layout) {
		if (!root || typeof root.Plotly === "undefined") {
			showPlotMessage(container, "Plotly is not available.");
			return;
		}

		var config = {
			responsive: true,
			displaylogo: false,
			modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d", "select2d"]
		};

		try {
			if (typeof root.Plotly.react === "function") {
				root.Plotly.react(container, traces, layout, config);
			} else {
				root.Plotly.newPlot(container, traces, layout, config);
			}
		} catch (error) {
			showPlotMessage(container, "Could not render the landscape: " + error.message);
		}
	}

	function showPlotMessage(container, message) {
		container.innerHTML = "";
		var note = document.createElement("div");
		note.className = "landscape-message";
		note.textContent = message;
		container.appendChild(note);
	}

	function showLoading(container, message) {
		container.innerHTML = "";

		var loading = document.createElement("div");
		loading.className = "landscape-loading";

		var spinner = document.createElement("div");
		spinner.className = "landscape-loading-spinner";

		var text = document.createElement("div");
		text.className = "landscape-loading-text";
		text.textContent = message;

		loading.appendChild(spinner);
		loading.appendChild(text);
		container.appendChild(loading);
	}

	function purgePlots(container) {
		if (!root || typeof root.Plotly === "undefined" || typeof root.Plotly.purge !== "function") {
			return;
		}

		Array.prototype.forEach.call(container.querySelectorAll(".js-plotly-plot"), function (plot) {
			try {
				root.Plotly.purge(plot);
			} catch (error) {
				// Ignore plots that are already gone.
			}
		});
	}

	function createField(labelText, select) {
		var label = document.createElement("label");
		label.className = "landscape-field";

		var text = document.createElement("span");
		text.textContent = labelText;

		label.appendChild(text);
		label.appendChild(select);

		return label;
	}

	function createSelect(id, options) {
		var select = document.createElement("select");
		select.id = id;

		for (var i = 0; i < options.length; i++) {
			var option = document.createElement("option");
			option.value = options[i].value;
			option.textContent = options[i].label;
			select.appendChild(option);
		}

		return select;
	}

	function sanitizeIdPart(value) {
		return String(value).replace(/[^a-zA-Z0-9_-]+/g, "_");
	}

	function captureState(plotDiv) {
		function value(id) {
			var element = plotDiv.querySelector("#" + id);
			return element ? element.value : "";
		}

		var state = {
			mode: value("landscape_dimension"),
			x: value("landscape_param_x"),
			y: value("landscape_param_y"),
			z: value("landscape_param_z"),
			result: value("landscape_result"),
			slice: value("landscape_slice"),
			showFailed: false,
			showPoints: false
		};

		var failed = plotDiv.querySelector("#landscape_show_failed");
		state.showFailed = !!(failed && failed.checked);

		var points = plotDiv.querySelector("#landscape_show_points");
		state.showPoints = !!(points && points.checked);

		return state.mode ? state : null;
	}

	function applyPreviousState(state, numericParams, resultNames, maxDimension) {
		if (!state) {
			return {
				mode: maxDimension,
				x: numericParams[0] || "",
				y: numericParams[1] || "",
				z: numericParams[2] || "",
				result: resultNames[0] || "",
				slice: "all",
				showFailed: false,
				showPoints: false
			};
		}

		var mode = parseInt(state.mode, 10);
		if (!isFinite(mode) || mode < 1) {
			mode = maxDimension;
		}
		mode = Math.min(mode, maxDimension);

		var x = numericParams.indexOf(state.x) >= 0 ? state.x : numericParams[0] || "";
		var y = numericParams.indexOf(state.y) >= 0 ? state.y : numericParams[1] || "";
		var z = numericParams.indexOf(state.z) >= 0 ? state.z : numericParams[2] || "";

		return {
			mode: mode,
			x: x,
			y: y,
			z: z,
			result: resultNames.indexOf(state.result) >= 0 ? state.result : resultNames[0] || "",
			slice: ["all", "low", "middle", "high"].indexOf(state.slice) >= 0 ? state.slice : "all",
			showFailed: !!state.showFailed,
			showPoints: !!state.showPoints
		};
	}

	function plotLandscape() {
		var plotDiv = document.getElementById("plotLandscape");

		if (!plotDiv) {
			return;
		}

		if (window.jQuery) {
			if (window.jQuery(plotDiv).data("loaded") === "true") {
				return;
			}
		} else if (plotDiv.dataset.loaded === "true") {
			return;
		}

		var previousState = captureState(plotDiv);
		purgePlots(plotDiv);
		plotDiv.innerHTML = "";

		var colors = getTheme();
		plotDiv.classList.toggle("landscape-dark", colors.dark);

		var headers = root.tab_results_headers_json;
		var rows = root.tab_results_csv_json;
		var resultNames = Array.isArray(root.result_names) ? root.result_names : [];
		var resultMinMax = root.result_min_max || [];
		var specialColumns = Array.isArray(root.special_col_names) ? root.special_col_names : [];

		if (!Array.isArray(headers) || !Array.isArray(rows) || !rows.length || !resultNames.length) {
			showPlotMessage(plotDiv, "No result data is available for a result landscape.");
			markLoaded(plotDiv);
			return;
		}

		var numericParams = getNumericParameterColumns(headers, rows, resultNames, specialColumns);

		if (!numericParams.length) {
			showPlotMessage(plotDiv, "No numeric parameter columns are available for a result landscape.");
			markLoaded(plotDiv);
			return;
		}

		var maxDimension = Math.min(3, numericParams.length);
		var initialState = applyPreviousState(previousState, numericParams, resultNames, maxDimension);
		var renderToken = 0;
		var updateTimer = null;

		var controls = document.createElement("div");
		controls.className = "landscape-controls";

		var dimensionOptions = [];

		for (var dimension = 1; dimension <= maxDimension; dimension++) {
			var dimensionLabel = dimension + "D";

			if (dimension === 2) {
				dimensionLabel = "2D contour";
			} else if (dimension === 3) {
				dimensionLabel = "3D surface (height = result)";
			}

			dimensionOptions.push({
				value: String(dimension),
				label: dimensionLabel
			});
		}

		var dimensionSelect = createSelect("landscape_dimension", dimensionOptions);
		dimensionSelect.value = String(initialState.mode);

		var xSelect = createSelect("landscape_param_x", numericParams.map(function (name) {
			return { value: name, label: name };
		}));

		var ySelect = createSelect("landscape_param_y", numericParams.map(function (name) {
			return { value: name, label: name };
		}));

		var zSelect = createSelect("landscape_param_z", numericParams.map(function (name) {
			return { value: name, label: name };
		}));

		xSelect.value = initialState.x;
		ySelect.value = initialState.y;
		zSelect.value = initialState.z;

		var resultSelect = createSelect("landscape_result", resultNames.map(function (name) {
			return { value: name, label: getResultLabel(name, resultNames, resultMinMax) };
		}));
		resultSelect.value = initialState.result;

		var sliceSelect = createSelect("landscape_slice", [
			{ value: "all", label: "All (average)" },
			{ value: "low", label: "Low third" },
			{ value: "middle", label: "Middle third" },
			{ value: "high", label: "High third" }
		]);
		sliceSelect.value = initialState.slice || "all";

		var xField = createField("Parameter X", xSelect);
		var yField = createField("Parameter Y", ySelect);
		var zField = createField("Slice parameter", zSelect);
		var sliceField = createField("Slice", sliceSelect);
		var resultField = createField("RESULT (3D)", resultSelect);

		var failedLabel = document.createElement("label");
		failedLabel.className = "landscape-toggle";

		var failedInput = document.createElement("input");
		failedInput.type = "checkbox";
		failedInput.id = "landscape_show_failed";
		failedInput.checked = initialState.showFailed;

		var failedText = document.createElement("span");
		failedText.textContent = "Show failed trials as red × (no result)";

		failedLabel.appendChild(failedInput);
		failedLabel.appendChild(failedText);

		var pointsLabel = document.createElement("label");
		pointsLabel.className = "landscape-toggle";

		var pointsInput = document.createElement("input");
		pointsInput.type = "checkbox";
		pointsInput.id = "landscape_show_points";
		pointsInput.checked = initialState.showPoints;

		var pointsText = document.createElement("span");
		pointsText.textContent = "Show trial points";

		pointsLabel.appendChild(pointsInput);
		pointsLabel.appendChild(pointsText);

		controls.appendChild(createField("Dimension", dimensionSelect));
		controls.appendChild(xField);
		controls.appendChild(yField);
		controls.appendChild(zField);
		controls.appendChild(sliceField);
		controls.appendChild(resultField);
		controls.appendChild(pointsLabel);
		controls.appendChild(failedLabel);

		var hint = document.createElement("div");
		hint.className = "landscape-hint";

		var hintText = "Height is the result value. The pink line is the training trajectory in trial order.";

		if (maxDimension === 1) {
			hintText += " Only one varying numeric parameter is available, so only 1D landscapes can be rendered for this run.";
		} else {
			hintText += " 2D shows a flat result contour. 3D shows a result surface whose height and color are the result.";

			if (maxDimension >= 3) {
				hintText += " The third parameter is used as a slice selector.";
			}

			hintText += " White gaps are regions without data; red X marks failed trials, which are cut out as holes.";
		}

		hint.textContent = hintText;

		var status = document.createElement("div");
		status.className = "landscape-status";

		var plotsContainer = document.createElement("div");
		plotsContainer.className = "landscape-plots";

		plotDiv.appendChild(controls);
		plotDiv.appendChild(hint);
		plotDiv.appendChild(status);
		plotDiv.appendChild(plotsContainer);

		function currentParamNames() {
			var mode = parseInt(dimensionSelect.value, 10);
			var names = [xSelect.value];

			if (mode >= 2) {
				names.push(ySelect.value);
			}

			if (mode >= 3) {
				names.push(zSelect.value);
			}

			return names;
		}

		function updateVisibility() {
			var mode = parseInt(dimensionSelect.value, 10);

			yField.style.display = mode >= 2 ? "" : "none";
			zField.style.display = mode >= 3 ? "" : "none";
			sliceField.style.display = mode >= 3 ? "" : "none";
			resultField.style.display = mode >= 3 ? "" : "none";
			pointsLabel.style.display = mode >= 3 ? "" : "none";
		}

		function ensureDistinctSelections() {
			var mode = parseInt(dimensionSelect.value, 10);
			var selects = [xSelect, ySelect, zSelect].slice(0, mode);
			var used = {};

			for (var i = 0; i < selects.length; i++) {
				var value = selects[i].value;

				if (value && !used[value]) {
					used[value] = true;
					continue;
				}

				var replacement = "";

				for (var p = 0; p < numericParams.length; p++) {
					if (!used[numericParams[p]]) {
						replacement = numericParams[p];
						break;
					}
				}

				if (replacement) {
					selects[i].value = replacement;
					used[replacement] = true;
				}
			}
		}

		function scheduleRender(delay) {
			if (updateTimer !== null) {
				clearTimeout(updateTimer);
			}

			updateTimer = setTimeout(function () {
				updateTimer = null;
				render();
			}, typeof delay === "number" ? delay : 80);
		}

		function render() {
			if (plotDiv.offsetParent === null) {
				markUnloaded(plotDiv);
				return;
			}

			var token = ++renderToken;
			var mode = parseInt(dimensionSelect.value, 10);
			var paramNames = currentParamNames();
			var selectedResult = resultSelect.value;
			var showFailed = failedInput.checked;
			var showPoints = pointsInput.checked;
			var slice = sliceSelect.value;

			if (paramNames.length !== mode || paramNames.some(function (name) { return !name; })) {
				purgePlots(plotsContainer);
				showPlotMessage(plotsContainer, "Please choose a parameter for every dimension.");
				return;
			}

			purgePlots(plotsContainer);
			showLoading(plotsContainer, "Building result landscape…");
			status.textContent = "Rendering…";

			// Let the browser paint the spinner before the potentially expensive
			// interpolation and Plotly rendering starts.
			setTimeout(function () {
				if (token !== renderToken) {
					return;
				}

				try {
					renderPreparedData(token, mode, paramNames, selectedResult, showFailed, showPoints, slice);
				} catch (error) {
					console.error("Could not render the result landscape:", error);
					purgePlots(plotsContainer);
					showPlotMessage(plotsContainer, "Could not render the result landscape: " + error.message);
					status.textContent = "";
				}
			}, 30);
		}

		function renderPreparedData(token, mode, paramNames, selectedResult, showFailed, showPoints, slice) {
			var resultsToRender = mode >= 3 ? [selectedResult] : resultNames.slice();
			var collected = collectLandscapeDataForResults(headers, rows, paramNames, resultsToRender);
			var prepared = [];
			var failedTotal = collected.failed.length;

			for (var i = 0; i < resultsToRender.length; i++) {
				var resultName = resultsToRender[i];
				var data = collected.buckets[i];

				if (mode >= 3) {
					data = filterDataBySlice(data, slice);
				}

				if (!data.valid.length) {
					continue;
				}

				var goal = getResultGoal(resultName, resultNames, resultMinMax);

				prepared.push({
					resultName: resultName,
					label: resultName + " (" + goal + ")",
					goal: goal,
					data: data
				});
			}

			if (!prepared.length) {
				purgePlots(plotsContainer);
				showPlotMessage(plotsContainer, "No numeric trials are available for the selected parameters.");
				status.textContent = "";
				return;
			}

			purgePlots(plotsContainer);
			plotsContainer.innerHTML = "";

			var failedStatus = showFailed
				? failedTotal + " failed trial" + (failedTotal === 1 ? "" : "s") + " marked"
				: "failed trials hidden";

			if (mode < 3) {
				status.textContent = prepared.length + " result" + (prepared.length === 1 ? "" : "s") +
					" · " + failedStatus;
			} else {
				status.textContent = "Showing " + prepared[0].label +
					(slice !== "all" ? " · slice: " + slice : "") +
					" · " + failedStatus;
			}

			for (var p = 0; p < prepared.length; p++) {
				if (token !== renderToken) {
					return;
				}

				var entry = prepared[p];
				var card = document.createElement("div");
				card.className = "landscape-card";

				var cardHeader = document.createElement("div");
				cardHeader.className = "landscape-card-header";

				var cardTitle = document.createElement("div");
				cardTitle.className = "landscape-card-title";
				cardTitle.textContent = entry.label;

				var best = findBestPoint(entry.data.valid, entry.goal);

				var cardMeta = document.createElement("div");
				cardMeta.className = "landscape-card-meta";
				cardMeta.textContent = entry.data.valid.length + " valid trials" +
					(best ? " · best " + formatNumber(best.loss) + " at trial " + best.trial : "");

				cardHeader.appendChild(cardTitle);
				cardHeader.appendChild(cardMeta);
				card.appendChild(cardHeader);

				var plot = document.createElement("div");
				plot.className = "landscape-plot";
				plot.id = "landscape_plot_" + sanitizeIdPart(entry.resultName) + "_" + p;
				card.appendChild(plot);
				plotsContainer.appendChild(card);

				var context = {
					colors: colors,
					goal: entry.goal,
					paramNames: paramNames,
					resultName: entry.resultName,
					resultLabel: entry.label,
					showFailed: showFailed,
					showPoints: showPoints
				};

				if (mode === 1) {
					render1D(plot, entry.data, context);
				} else if (mode === 2) {
					render2D(plot, entry.data, context);
				} else {
					render3D(plot, entry.data, context);
				}
			}
		}

		dimensionSelect.addEventListener("change", function () {
			updateVisibility();
			ensureDistinctSelections();
			scheduleRender();
		});

		[xSelect, ySelect, zSelect].forEach(function (select) {
			select.addEventListener("change", function () {
				ensureDistinctSelections();
				scheduleRender();
			});
		});

		resultSelect.addEventListener("change", scheduleRender);
		sliceSelect.addEventListener("change", scheduleRender);
		failedInput.addEventListener("change", scheduleRender);
		pointsInput.addEventListener("change", scheduleRender);

		updateVisibility();
		ensureDistinctSelections();
		scheduleRender(0);
		markLoaded(plotDiv);
		installThemeListener();
	}

	function markLoaded(plotDiv) {
		if (window.jQuery) {
			window.jQuery(plotDiv).data("loaded", "true");
		}

		plotDiv.dataset.loaded = "true";
	}

	function markUnloaded(plotDiv) {
		if (window.jQuery) {
			window.jQuery(plotDiv).data("loaded", false);
		}

		plotDiv.dataset.loaded = "false";
	}

	function refreshLandscape() {
		var plotDiv = document.getElementById("plotLandscape");

		if (!plotDiv) {
			return;
		}

		markUnloaded(plotDiv);

		if (plotDiv.offsetParent !== null) {
			plotLandscape();
		}
	}

	function installThemeListener() {
		if (installThemeListener.installed) {
			return;
		}

		var themeSelect = document.getElementById("themeSelect");

		if (!themeSelect) {
			return;
		}

		installThemeListener.installed = true;

		themeSelect.addEventListener("change", function () {
			setTimeout(function () {
				var plotDiv = document.getElementById("plotLandscape");

				if (!plotDiv) {
					return;
				}

				if (plotDiv.offsetParent !== null) {
					refreshLandscape();
				} else {
					markUnloaded(plotDiv);
				}
			}, 60);
		});
	}

	var api = {
		getNumericParameterColumns: getNumericParameterColumns,
		getResultGoal: getResultGoal,
		collectLandscapeData: collectLandscapeData,
		collectLandscapeDataForResults: collectLandscapeDataForResults,
		findBestPoint: findBestPoint,
		interpolate1D: interpolate1D,
		interpolate2D: interpolate2D,
		interpolate3D: interpolate3D,
		surfaceHeightAt: surfaceHeightAt,
		filterDataBySlice: filterDataBySlice,
		getResultLabel: getResultLabel,
		getLossColorscale: getLossColorscale,
		formatNumber: formatNumber,
		plotLandscape: plotLandscape,
		refreshLandscape: refreshLandscape
	};

	if (typeof window !== "undefined" && typeof document !== "undefined") {
		window.plotLandscape = plotLandscape;
		window.__landscape_refresh = refreshLandscape;
	}

	return api;
});

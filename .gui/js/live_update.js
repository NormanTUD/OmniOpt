"use strict";

(function () {
	if (window.__omniopt_share_live_started) {
		return;
	}

	window.__omniopt_share_live_started = true;

	var pollIntervalMs = typeof window.SHARE_LIVE_POLL_MS === "number"
		? window.SHARE_LIVE_POLL_MS
		: 2500;
	var debugEnabled = new URL(window.location.href).searchParams.get("live_debug") === "1";
	var liveEnabledStorageKey = "omniopt-share-live-updates-enabled";
	var currentVersion = null;
	var pollTimer = null;
	var updating = false;
	var pending = false;
	var liveEnabled = true;
	var forceRefreshOnce = false;

	function readLiveEnabled() {
		try {
			var value = window.localStorage.getItem(liveEnabledStorageKey);
			return value === null ? true : value === "1";
		} catch (error) {
			return true;
		}
	}

	function saveLiveEnabled(value) {
		try {
			window.localStorage.setItem(liveEnabledStorageKey, value ? "1" : "0");
		} catch (error) {
			// Storage can be unavailable in restricted contexts.
		}
	}

	function debugLog(message) {
		if (!debugEnabled) {
			return;
		}

		var element = document.getElementById("share-live-debug");

		if (!element) {
			element = document.createElement("pre");
			element.id = "share-live-debug";
			element.style.display = "none";
			document.body.appendChild(element);
		}

		element.textContent += message + "\n";
	}

	function debugDuration(label, startTime) {
		if (debugEnabled) {
			debugLog(label + ": " + (performance.now() - startTime).toFixed(1) + "ms");
		}
	}

	function buildLiveUrl(parameter, value) {
		var url = new URL(window.location.href);
		url.hash = "";
		url.searchParams.set(parameter, value);
		url.searchParams.set("_live_ts", String(Date.now()));
		return url.toString();
	}

	function schedulePoll(delay) {
		if (!liveEnabled) {
			if (pollTimer !== null) {
				clearTimeout(pollTimer);
				pollTimer = null;
			}

			return;
		}

		if (pollTimer !== null) {
			clearTimeout(pollTimer);
		}

		pollTimer = setTimeout(poll, typeof delay === "number" ? delay : pollIntervalMs);
	}

	async function fetchJson(url) {
		var response = await fetch(url, {
			cache: "no-store",
			credentials: "same-origin",
			headers: { "Accept": "application/json" }
		});

		if (!response.ok) {
			throw new Error("HTTP " + response.status);
		}

		var contentType = response.headers.get("content-type") || "";

		if (contentType.indexOf("application/json") === -1) {
			throw new Error("Expected JSON response");
		}

		return response.json();
	}

	function directTabs(menu) {
		return Array.from(menu.children).filter(function (child) {
			return child.getAttribute("role") === "tab";
		});
	}

	function directPanels(section) {
		return Array.from(section.children).filter(function (child) {
			return child.getAttribute("role") === "tabpanel";
		});
	}

	function captureViewState() {
		var section = document.querySelector(".tabs");
		var menu = section ? section.querySelector("menu[role='tablist']") : null;
		var activeButton = menu
			? directTabs(menu).find(function (button) {
				return button.getAttribute("aria-selected") === "true";
			})
			: null;
		var activeId = activeButton ? activeButton.getAttribute("aria-controls") : null;
		var activePanel = activeId ? document.getElementById(activeId) : null;
		var searchValue = "";
		var tableScrollTop = 0;

		if (activePanel) {
			var searchInput = activePanel.querySelector(".results-search");

			if (searchInput) {
				searchValue = searchInput.value;
			}

			var tableScroll = activePanel.querySelector(".results-table-scroll");

			if (tableScroll) {
				tableScrollTop = tableScroll.scrollTop;
			}
		}

		return {
			activeId: activeId,
			searchValue: searchValue,
			tableScrollTop: tableScrollTop,
			scrollX: window.scrollX,
			scrollY: window.scrollY
		};
	}

	function updateGlobals(liveState) {
		var data = liveState.json_data || {};

		Object.keys(data).forEach(function (key) {
			window[key] = data[key];
		});

		if (Array.isArray(liveState.result_names)) {
			window.result_names = liveState.result_names;
		}

		if (Array.isArray(liveState.result_min_max)) {
			window.result_min_max = liveState.result_min_max;
		}

		if (Array.isArray(liveState.special_col_names)) {
			window.special_col_names = liveState.special_col_names;
		}
	}

	function clearCachedPlotFlags(panel) {
		if (!window.jQuery) {
			return;
		}

		var elements = [panel].concat(Array.from(panel.querySelectorAll("[id]")));

		elements.forEach(function (element) {
			window.jQuery(element).data("loaded", false);
			window.jQuery(element).data("md5", null);
		});
	}

	function createTabButton(tab) {
		var button = document.createElement("button");
		button.className = "share_nav_button";
		button.setAttribute("role", "tab");
		button.setAttribute("aria-controls", tab.id);
		button.innerHTML = tab.name;

		if (tab.onclick) {
			button.setAttribute("onclick", tab.onclick);
		}

		return button;
	}

	function createTabPanel(tab) {
		var panel = document.createElement("article");
		panel.setAttribute("role", "tabpanel");
		panel.id = tab.id;
		panel.hidden = true;
		panel.dataset.shareContentHash = tab.content_hash || "";
		panel.innerHTML = tab.content || "";
		return panel;
	}

	function activateTab(button) {
		var container = button.closest(".tabs");

		if (!container) {
			return;
		}

		var menu = container.querySelector("menu[role='tablist']");

		if (menu) {
			directTabs(menu).forEach(function (tabButton) {
				tabButton.setAttribute("aria-selected", "false");
			});
		}

		directPanels(container).forEach(function (panel) {
			panel.hidden = true;
		});

		button.setAttribute("aria-selected", "true");

		var panel = document.getElementById(button.getAttribute("aria-controls"));

		if (panel) {
			panel.hidden = false;

			if (typeof load_lazy_panel === "function") {
				load_lazy_panel(panel);
			}

			var nestedTabs = panel.querySelector(".tabs");

			if (nestedTabs && typeof initialize_tabs === "function") {
				initialize_tabs();
			}
		}
	}

	function bindLiveTabButton(button) {
		button.addEventListener("click", function () {
			activateTab(button);
		});
	}

	function prepareTableRestore(panel, view) {
		if (!panel) {
			return;
		}

		var tableContainer = panel.querySelector("[data-lazy-results-table-id]");

		if (!tableContainer) {
			return;
		}

		window.__omniopt_results_restore = window.__omniopt_results_restore || {};
		window.__omniopt_results_restore[tableContainer.id] = {
			search: view.searchValue,
			scrollTop: view.tableScrollTop
		};
	}

	function restoreViewState(view) {
		requestAnimationFrame(function () {
			window.scrollTo(view.scrollX || 0, view.scrollY || 0);
		});
	}

	function reconcileTabs(liveTabs, view, resultsChanged) {
		var section = document.querySelector(".tabs");

		if (!section) {
			return false;
		}

		var menu = section.querySelector("menu[role='tablist']");

		if (!menu) {
			return false;
		}

		if (resultsChanged) {
			prepareTableRestore(view.activeId ? document.getElementById(view.activeId) : null, view);
		}

		var existingButtons = new Map();
		var existingPanels = new Map();

		directTabs(menu).forEach(function (button) {
			existingButtons.set(button.getAttribute("aria-controls"), button);
		});

		directPanels(section).forEach(function (panel) {
			existingPanels.set(panel.id, panel);
		});

		var wantedIds = new Set();

		liveTabs.forEach(function (tab) {
			if (!tab || !tab.id) {
				return;
			}

			wantedIds.add(tab.id);

			var button = existingButtons.get(tab.id);

			if (!button) {
				button = createTabButton(tab);
				bindLiveTabButton(button);
				menu.appendChild(button);
			} else {
				if (button.innerHTML !== tab.name) {
					button.innerHTML = tab.name;
				}

				if (tab.onclick) {
					if (button.getAttribute("onclick") !== tab.onclick) {
						button.setAttribute("onclick", tab.onclick);
					}
				} else {
					button.removeAttribute("onclick");
				}
			}

			existingButtons.set(tab.id, button);

			var panel = existingPanels.get(tab.id);

			if (!panel) {
				panel = createTabPanel(tab);
				section.appendChild(panel);
			} else {
				var contentChanged = tab.content_hash && panel.dataset.shareContentHash !== tab.content_hash;

				if ((tab.id === "tab_results" && resultsChanged) || contentChanged) {
					panel.innerHTML = tab.content || "";
					panel.dataset.shareContentHash = tab.content_hash || "";
				}
			}

			clearCachedPlotFlags(panel);
			existingPanels.set(tab.id, panel);
		});

		existingButtons.forEach(function (button, id) {
			if (!wantedIds.has(id)) {
				button.remove();
				existingButtons.delete(id);
			}
		});

		existingPanels.forEach(function (panel, id) {
			if (!wantedIds.has(id)) {
				panel.remove();
				existingPanels.delete(id);
			}
		});

		var desiredButtonOrder = liveTabs.map(function (tab) { return tab.id; }).filter(function (id) {
			return existingButtons.has(id);
		});
		var currentButtonOrder = directTabs(menu).map(function (button) {
			return button.getAttribute("aria-controls");
		});

		if (currentButtonOrder.join("\n") !== desiredButtonOrder.join("\n")) {
			desiredButtonOrder.forEach(function (id) {
				menu.appendChild(existingButtons.get(id));
			});
		}

		var desiredPanelOrder = liveTabs.map(function (tab) { return tab.id; }).filter(function (id) {
			return existingPanels.has(id);
		});
		var currentPanelOrder = directPanels(section).map(function (panel) {
			return panel.id;
		});

		if (currentPanelOrder.join("\n") !== desiredPanelOrder.join("\n")) {
			desiredPanelOrder.forEach(function (id) {
				section.appendChild(existingPanels.get(id));
			});
		}

		var activeButton = directTabs(menu).find(function (button) {
			return button.getAttribute("aria-controls") === view.activeId;
		});

		if (!activeButton) {
			activeButton = directTabs(menu)[0];
		}

		directTabs(menu).forEach(function (button) {
			button.setAttribute("aria-selected", "false");
		});

		directPanels(section).forEach(function (panel) {
			panel.hidden = true;
		});

		if (!activeButton) {
			return true;
		}

		activeButton.setAttribute("aria-selected", "true");

		var activePanel = document.getElementById(activeButton.getAttribute("aria-controls"));

		if (!activePanel) {
			return true;
		}

		activePanel.hidden = false;

		var lazyStart = performance.now();

		if (typeof load_lazy_panel === "function") {
			load_lazy_panel(activePanel);
		}

		debugDuration("load_lazy_panel", lazyStart);

		var activeTabId = activeButton.getAttribute("aria-controls");
		var liveRefresher = null;

		var landscapeTab = activeTabId === "tab_landscape";

		if (landscapeTab) {
			// Never rebuild the whole landscape tab on a live update. The
			// dedicated refresher reuses the existing cards and Plotly plots
			// and only runs when the result data actually changed.
			if (resultsChanged && typeof window.__omnioptLandscapeLiveRefresh === "function") {
				liveRefresher = window.__omnioptLandscapeLiveRefresh;
			}
		} else if (activeTabId === "tab_scatter_2d" && typeof window.__scatter2d_refresh === "function") {
			liveRefresher = window.__scatter2d_refresh;
		} else if (activeTabId === "tab_scatter_3d" && typeof window.__scatter3d_refresh === "function") {
			liveRefresher = window.__scatter3d_refresh;
		}

		if (liveRefresher) {
			var refreshStart = performance.now();

			try {
				liveRefresher();
			} catch (error) {
				console.error("Live update could not refresh the active plot:", error);
			}

			debugDuration("active plot refresh", refreshStart);

			if (window.jQuery) {
				if (activeTabId === "tab_scatter_2d") {
					window.jQuery("#plotScatter2d").data("loaded", "true");
				} else if (activeTabId === "tab_scatter_3d") {
					window.jQuery("#plotScatter3d").data("loaded", "true");
				}
			}
		} else if (!landscapeTab && activeButton.getAttribute("onclick")) {
			var plotStart = performance.now();
			var onclickCode = activeButton.getAttribute("onclick") || "";
			var needsPlotly = /(^|[^A-Za-z0-9_])(plot[A-Za-z0-9_]*|Plot[A-Za-z0-9_]*|load_pareto_graph_from_idxs|createParallelPlot)\b/.test(onclickCode);

			function runPlotOnclick() {
				try {
					activeButton.onclick();
				} catch (error) {
					console.error("Live update could not refresh the active plot:", error);
				}

				debugDuration("active plot render", plotStart);
			}

			if (needsPlotly && typeof window.ensurePlotlyLoaded === "function") {
				window.ensurePlotlyLoaded(runPlotOnclick);
			} else {
				runPlotOnclick();
			}
		}

		restoreViewState(view);
		return true;
	}

	async function refreshLiveData(newVersion) {
		updating = true;
		var view = captureViewState();
		var refreshStart = performance.now();

		try {
			var fetchStart = performance.now();
			var liveState = await fetchJson(buildLiveUrl("live_state", "1"));
			debugDuration("live_state fetch+parse", fetchStart);

			if (!liveEnabled || !liveState || !Array.isArray(liveState.tabs)) {
				return;
			}

			var resultsChanged = !liveState.results_version ||
				liveState.results_version !== window.share_live_results_version;

			var globalsStart = performance.now();
			updateGlobals(liveState);
			debugDuration("update globals", globalsStart);

			var reconcileStart = performance.now();

			if (reconcileTabs(liveState.tabs, view, resultsChanged)) {
				currentVersion = liveState.version || newVersion;

				if (liveState.results_version) {
					window.share_live_results_version = liveState.results_version;
				}
			}

			debugDuration("reconcile tabs", reconcileStart);
		} catch (error) {
			console.error("Live update failed:", error);
		} finally {
			debugDuration("refresh total", refreshStart);
			updating = false;

			if (pending) {
				pending = false;
				schedulePoll(0);
			}
		}
	}

	async function poll() {
		pollTimer = null;

		if (!liveEnabled) {
			return;
		}

		if (document.visibilityState === "hidden") {
			schedulePoll();
			return;
		}

		if (updating) {
			pending = true;
			schedulePoll();
			return;
		}

		try {
			var versionData = await fetchJson(buildLiveUrl("live_version", "1"));

			if (!liveEnabled) {
				return;
			}

			if (currentVersion === null && !forceRefreshOnce) {
				currentVersion = versionData.version || "";
			} else if (forceRefreshOnce || (versionData.version && versionData.version !== currentVersion)) {
				forceRefreshOnce = false;
				await refreshLiveData(versionData.version);
			}
		} catch (error) {
			// Network hiccups should never disturb the already visible page.
		}

		if (liveEnabled) {
			schedulePoll();
		}
	}

	function setLiveEnabled(value, persist) {
		liveEnabled = value === true;

		if (persist) {
			saveLiveEnabled(liveEnabled);
		}

		var toggle = document.getElementById("share_live_update_enabled");

		if (toggle) {
			toggle.checked = liveEnabled;
		}

		if (liveEnabled) {
			forceRefreshOnce = true;
			schedulePoll(0);
		} else {
			if (pollTimer !== null) {
				clearTimeout(pollTimer);
				pollTimer = null;
			}
		}
	}

	function startLiveUpdates() {
		if (window.location.protocol !== "http:" && window.location.protocol !== "https:") {
			return;
		}

		var url = new URL(window.location.href);

		if (url.searchParams.has("export") || url.searchParams.has("export_and_exit")) {
			return;
		}

		if (!document.querySelector(".tabs")) {
			return;
		}

		liveEnabled = readLiveEnabled();

		var toggle = document.getElementById("share_live_update_enabled");

		if (toggle) {
			toggle.checked = liveEnabled;
			toggle.addEventListener("change", function () {
				setLiveEnabled(toggle.checked, true);
			});
		}

		if (liveEnabled) {
			schedulePoll(300);
		}
	}

	if (debugEnabled) {
		window.__share_live_force_refresh = function () {
			forceRefreshOnce = true;
			return poll();
		};
	}

	if (document.readyState === "complete") {
		startLiveUpdates();
	} else {
		window.addEventListener("load", startLiveUpdates);
	}

	document.addEventListener("visibilitychange", function () {
		if (document.visibilityState === "visible" && !updating) {
			schedulePoll(200);
		}
	});
})();

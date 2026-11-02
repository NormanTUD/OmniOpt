"use strict";

(function () {
	if (window.__omniopt_share_live_started) {
		return;
	}

	window.__omniopt_share_live_started = true;

	var pollIntervalMs = typeof window.SHARE_LIVE_POLL_MS === "number"
		? window.SHARE_LIVE_POLL_MS
		: 2500;
	var currentVersion = null;
	var pollTimer = null;
	var updating = false;
	var pending = false;

	function buildLiveUrl(parameter, value) {
		var url = new URL(window.location.href);
		url.hash = "";
		url.searchParams.set(parameter, value);
		url.searchParams.set("_live_ts", String(Date.now()));
		return url.toString();
	}

	function schedulePoll(delay) {
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

			menu.appendChild(button);

			var panel = existingPanels.get(tab.id);

			if (!panel) {
				panel = createTabPanel(tab);
			} else if ((tab.id === "tab_results" && resultsChanged) || panel.innerHTML !== (tab.content || "")) {
				panel.innerHTML = tab.content || "";
			}

			clearCachedPlotFlags(panel);
			section.appendChild(panel);
		});

		existingButtons.forEach(function (button, id) {
			if (!wantedIds.has(id)) {
				button.remove();
			}
		});

		existingPanels.forEach(function (panel, id) {
			if (!wantedIds.has(id)) {
				panel.remove();
			}
		});

		directPanels(section).forEach(function (panel) {
			panel.querySelectorAll("pre[data-log-file]").forEach(function (pre) {
				delete pre.dataset.loaded;
			});
		});

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

		if (typeof load_lazy_panel === "function") {
			load_lazy_panel(activePanel);
		}

		if (activeButton.getAttribute("onclick")) {
			try {
				activeButton.onclick();
			} catch (error) {
				console.error("Live update could not refresh the active plot:", error);
			}
		}

		restoreViewState(view);
		return true;
	}

	async function refreshLiveData(newVersion) {
		updating = true;
		var view = captureViewState();

		try {
			var liveState = await fetchJson(buildLiveUrl("live_state", "1"));

			if (!liveState || !Array.isArray(liveState.tabs)) {
				return;
			}

			var resultsChanged = !liveState.results_version ||
				liveState.results_version !== window.share_live_results_version;

			updateGlobals(liveState);

			if (reconcileTabs(liveState.tabs, view, resultsChanged)) {
				currentVersion = liveState.version || newVersion;

				if (liveState.results_version) {
					window.share_live_results_version = liveState.results_version;
				}
			}
		} catch (error) {
			console.error("Live update failed:", error);
		} finally {
			updating = false;

			if (pending) {
				pending = false;
				schedulePoll(0);
			}
		}
	}

	async function poll() {
		pollTimer = null;

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

			if (currentVersion === null) {
				currentVersion = versionData.version || "";
			} else if (versionData.version && versionData.version !== currentVersion) {
				await refreshLiveData(versionData.version);
			}
		} catch (error) {
			// Network hiccups should never disturb the already visible page.
		}

		schedulePoll();
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

		schedulePoll(300);
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

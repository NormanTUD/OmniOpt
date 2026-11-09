"use strict";

const CACHE_PREFIX = "omniopt-pwa-";
const CACHE_NAME = CACHE_PREFIX + "v5";
const OFFLINE_PAGE = "offline.html";
const PRECACHE_LIST = "offline_assets.php";

const CORE_ASSETS = [
	"./",
	"./index.php",
	"./gui.php",
	"./tutorials.php",
	"./share.php",
	"./404.php",
	"./offline.html",
	"./manifest.webmanifest",
	"./style.css",
	"./prism.css",
	"./css/share.css",
	"./css/xp.css",
	"./css/mermaid.min.css",
	"./new_logo.svg",
	"./favicon.ico",
	"./icons/icon-192.png",
	"./icons/icon-512.png",
	"./icons/apple-touch-icon.png",
	"./main.js",
	"./initialization.js",
	"./darkmode.js",
	"./search.js",
	"./footer.js",
	"./share.js",
	"./js/results_table.js",
	"./js/results_analysis.js",
	"./js/landscape.js",
	"./js/share_functions.js",
	"./js/gridjs.umd.js",
	"./js/parallel.js",
	"./js/plot_parameter_distribution_by_status.js",
	"./js/arm_distribution.js",
	"./js/kde_histogram.js",
	"./js/stacked_histogram.js",
	"./js/job_info_plots.js"
];

function scopedUrl(path) {
	return new URL(path, self.registration.scope).href;
}

function isCacheable(response) {
	return response && (response.ok || response.type === "opaque");
}

async function putInCache(cache, request, response) {
	if (!isCacheable(response)) {
		return;
	}

	try {
		await cache.put(request, response.clone());
	} catch (error) {
		// Some responses (for example partial or redirected ones) cannot be
		// stored by every browser. They still work online.
	}
}

function offlineResponse() {
	return new Response("Offline: This resource has not been cached yet.", {
		status: 503,
		statusText: "Offline",
		headers: { "Content-Type": "text/plain; charset=utf-8" }
	});
}

function isStaticAsset(url) {
	return /\.(?:css|js|mjs|json|webmanifest|png|jpe?g|gif|svg|webp|ico|woff2?|ttf|otf|eot|wasm)$/i.test(url.pathname);
}

async function precache() {
	const cache = await caches.open(CACHE_NAME);
	const urls = new Set(CORE_ASSETS.map(scopedUrl));

	try {
		const manifestUrl = scopedUrl(PRECACHE_LIST) + "?cache=" + encodeURIComponent(CACHE_NAME);
		const response = await fetch(manifestUrl, { cache: "no-store", credentials: "same-origin" });

		if (response.ok) {
			const assets = await response.json();

			if (Array.isArray(assets)) {
				for (const asset of assets) {
					try {
						urls.add(scopedUrl(asset));
					} catch (error) {
						// Ignore malformed entries.
					}
				}
			}
		}
	} catch (error) {
		// The core assets below are still cached if the manifest is unavailable.
	}

	await Promise.all(Array.from(urls).map(async function (url) {
		try {
			const request = new Request(url, { cache: "reload", credentials: "same-origin" });
			const response = await fetch(request);
			await putInCache(cache, request, response);
		} catch (error) {
			// A single unavailable asset should not abort the whole installation.
		}
	}));
}

async function networkFirst(request) {
	const cache = await caches.open(CACHE_NAME);

	try {
		const response = await fetch(request);
		await putInCache(cache, request, response);
		return response;
	} catch (error) {
		const cached = await cache.match(request);

		if (cached) {
			return cached;
		}

		return offlineResponse();
	}
}

async function matchNavigationCache(cache, request) {
	const cached = await cache.match(request);

	if (cached) {
		return cached;
	}

	const url = new URL(request.url);

	if (/\.php$/i.test(url.pathname)) {
		const withoutPhp = new URL(url.href);
		withoutPhp.pathname = withoutPhp.pathname.replace(/\.php$/i, "");
		return cache.match(withoutPhp.href);
	}

	if (!/\.[^/]+$/.test(url.pathname)) {
		const withPhp = new URL(url.href);
		withPhp.pathname += ".php";
		return cache.match(withPhp.href);
	}

	return undefined;
}

async function navigationNetworkFirst(request) {
	const cache = await caches.open(CACHE_NAME);

	try {
		const response = await fetch(request);
		await putInCache(cache, request, response);
		return response;
	} catch (error) {
		const cached = await matchNavigationCache(cache, request);

		if (cached) {
			return cached;
		}

		const offline = await cache.match(scopedUrl(OFFLINE_PAGE));

		if (offline) {
			return offline;
		}

		return offlineResponse();
	}
}

async function staleWhileRevalidate(request) {
	const cache = await caches.open(CACHE_NAME);
	const cached = await cache.match(request);

	const network = fetch(request).then(async function (response) {
		await putInCache(cache, request, response);
		return response;
	}).catch(function () {
		return null;
	});

	if (cached) {
		return cached;
	}

	const response = await network;

	if (response) {
		return response;
	}

	return offlineResponse();
}

self.addEventListener("install", function (event) {
	event.waitUntil((async function () {
		await precache();
		await self.skipWaiting();
	})());
});

self.addEventListener("activate", function (event) {
	event.waitUntil((async function () {
		const cacheNames = await caches.keys();

		await Promise.all(cacheNames.map(function (cacheName) {
			if (cacheName.startsWith(CACHE_PREFIX) && cacheName !== CACHE_NAME) {
				return caches.delete(cacheName);
			}

			return Promise.resolve(false);
		}));

		await self.clients.claim();
	})());
});

self.addEventListener("fetch", function (event) {
	const request = event.request;

	if (request.method !== "GET") {
		return;
	}

	const url = new URL(request.url);

	if (url.protocol !== "http:" && url.protocol !== "https:") {
		return;
	}

	if (request.cache === "only-if-cached" && request.mode !== "same-origin") {
		return;
	}

	if (url.searchParams.has("live_version") || url.searchParams.has("live_state")) {
		event.respondWith(fetch(request));
		return;
	}

	if (request.mode === "navigate") {
		event.respondWith(navigationNetworkFirst(request));
		return;
	}

	if (url.origin !== self.location.origin) {
		event.respondWith(staleWhileRevalidate(request));
		return;
	}

	if (isStaticAsset(url)) {
		if (/\.(?:js|css|webmanifest)$/i.test(url.pathname)) {
			event.respondWith(networkFirst(request));
		} else {
			event.respondWith(staleWhileRevalidate(request));
		}

		return;
	}

	event.respondWith(networkFirst(request));
});

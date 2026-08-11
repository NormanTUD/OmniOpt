// Helper for .tests/js_syntax.py: parse-check every *.js file in a
// single node process.  Replaces a per-file `node --check` invocation
// (~50 ms startup each) with one shared process.
//
// Usage: node js_syntax_helper.js <dir>
//
// Prints "<relative-path>: <error message>" to STDOUT for every failing
// file and exits with 0 on success, 1 on any failure.

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const dir = process.argv[2];
if (!dir) {
    console.error("usage: node js_syntax_helper.js <dir>");
    process.exit(2);
}

let bad = 0;
function checkFile(full) {
    const code = fs.readFileSync(full, "utf8");
    try {
        // Wrap in a Script so syntax errors are caught identically to
        // `node --check`.  Wrapping in a Function would be slightly
        // faster but allows some illegal-syntax forms.
        new vm.Script(code, { filename: full });
    } catch (e) {
        console.log(`${path.relative(dir, full)}: ${e.message}`);
        bad++;
    }
}

function walk(d) {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
        const full = path.join(d, entry.name);
        if (entry.isDirectory()) {
            walk(full);
        } else if (entry.isFile() && full.endsWith(".js")) {
            checkFile(full);
        }
    }
}

try {
    walk(dir);
} catch (e) {
    console.error(`walk failed: ${e.message}`);
    process.exit(2);
}
process.exit(bad === 0 ? 0 : 1);

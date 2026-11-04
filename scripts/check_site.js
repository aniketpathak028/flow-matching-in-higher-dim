// Checks the website without a browser: stubs Plotly, runs docs/ scripts, reports charts with no data or NaN.
// Usage: node scripts/check_site.js docs
const fs = require("fs"), path = require("path"), vm = require("vm");
const dir = process.argv[2];
const html = fs.readFileSync(path.join(dir, "index.html"), "utf8");
const inline = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
const els = {}, charts = [];
const el = (id) => (els[id] ||= { id, innerHTML: "", style: {}, classList: { add() {}, remove() {}, toggle() {} },
  addEventListener() {}, querySelectorAll: () => [], setAttribute() {}, getAttribute: () => null });
const ctx = {
  console, Math, JSON, isFinite, Number, Object, Array, String, MutationObserver: class { observe() {} },
  getComputedStyle: () => ({ getPropertyValue: () => "#000" }),
  Plotly: { react: (e, traces) => charts.push([e.id, traces]), newPlot: (e, traces) => charts.push([e.id, traces]) },
};
ctx.window = ctx;
ctx.matchMedia = () => ({ matches: false, addEventListener() {} });
ctx.document = { documentElement: el("html"), getElementById: el, querySelector: (s) => el(s), querySelectorAll: () => [],
  addEventListener() {}, body: el("body") };
vm.createContext(ctx);
for (const f of ["data.js", "charts.js"]) vm.runInContext(fs.readFileSync(path.join(dir, f), "utf8"), ctx);
for (const s of inline) vm.runInContext(s, ctx);
let bad = 0;
for (const [id, traces] of charts) {
  const ys = traces.flatMap((t) => [...(t.x || []), ...(t.y || [])]).filter((v) => typeof v === "number");
  const nan = ys.filter((v) => !isFinite(v)).length;
  if (!ys.length || nan) { bad++; console.log("PROBLEM", id, ys.length, "values", nan, "NaN"); }
}
console.log(`${charts.length} charts, ${bad} with problems`);

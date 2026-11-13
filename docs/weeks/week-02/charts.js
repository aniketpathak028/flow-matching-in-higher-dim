/* Charts for the presentation page. Data: window.DATA (from results/summary.json).
   Every number is a mean over 3 seeds; error bars are 95% confidence intervals. */
(function () {
  const V = window.DATA.variants;
  const metric = (name, key) => (V[name] && V[name].metrics[key]) || [NaN, 0, 0];
  const curve = (name, key) => (V[name] && V[name].curves[key]) || [];

  function theme() {
    const css = getComputedStyle(document.documentElement);
    const get = (k) => css.getPropertyValue(k).trim();
    return {
      ink: get("--ink"), muted: get("--muted"), grid: get("--grid"), card: get("--card"),
      series: [get("--s1"), get("--s2"), get("--s3"), get("--s4")],
      font: get("--font-body") || "sans-serif",
    };
  }

  function layout(t, extra) {
    const axis = {
      gridcolor: t.grid, zerolinecolor: t.grid, linecolor: t.grid, tickcolor: t.grid,
      tickfont: { color: t.muted, size: 12 }, titlefont: { color: t.muted, size: 12 }, automargin: true,
    };
    const base = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: t.font, color: t.ink, size: 13 },
      margin: { l: 10, r: 16, t: 34, b: 44 }, height: 320,
      showlegend: false, hovermode: "closest",
      hoverlabel: { bgcolor: t.card, bordercolor: t.grid, font: { color: t.ink, family: t.font } },
      legend: { orientation: "h", y: -0.22, x: 0, font: { color: t.muted, size: 12 } },
      title: { font: { size: 14, color: t.ink }, x: 0, xanchor: "left", y: 0.98 },
    };
    const out = Object.assign(base, extra || {});
    out.xaxis = Object.assign({}, axis, (extra || {}).xaxis || {});
    out.yaxis = Object.assign({}, axis, (extra || {}).yaxis || {});
    return out;
  }

  const fmt = (v) => (!isFinite(v) ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 0.01 ? v.toPrecision(3) : v.toExponential(2));
  const tip = (label, key) => `<b>${label}</b><br>%{customdata[0]} ± %{customdata[1]}<extra></extra>`;

  /* Horizontal bars, one per variant. items = [[label, variant], ...] */
  function bars(el, items, key, opts) {
    const t = theme();
    const vals = items.map(([, n]) => metric(n, key));
    const scale = opts.scale || 1;
    const colors = items.map((_, i) => (opts.highlight && opts.highlight.includes(i) ? t.series[1] : t.series[0]));
    const trace = {
      type: "bar", orientation: "h",
      y: items.map(([l]) => l), x: vals.map((v) => v[0] * scale),
      error_x: { type: "data", array: vals.map((v) => v[1] * scale), color: t.muted, thickness: 1.2, width: 4 },
      marker: { color: colors }, width: 0.55,
      customdata: vals.map((v) => [fmt(v[0] * scale), fmt(v[1] * scale)]),
      hovertemplate: `%{y}<br>${opts.unit || ""}%{customdata[0]} ± %{customdata[1]}<extra></extra>`,
    };
    Plotly.react(el, [trace], layout(t, {
      title: { text: opts.title }, height: Math.max(200, 54 * items.length + 90),
      xaxis: { type: opts.log ? "log" : "linear", title: { text: opts.xTitle || "" }, range: opts.range },
      yaxis: { autorange: "reversed" },
      shapes: opts.vline != null ? [{ type: "line", x0: opts.vline, x1: opts.vline, yref: "paper", y0: 0, y1: 1,
        line: { color: t.muted, dash: "dot", width: 1 } }] : [],
    }), { displayModeBar: false, responsive: true });
  }

  /* Grouped vertical bars. groups = category labels; series = [{label, values:[[mean,ci],...]}] */
  function grouped(el, groups, series, opts) {
    const t = theme();
    const traces = series.map((s, i) => ({
      type: "bar", name: s.label, x: groups, y: s.values.map((v) => v[0]),
      error_y: { type: "data", array: s.values.map((v) => v[1]), color: t.muted, thickness: 1.2, width: 4 },
      marker: { color: t.series[s.color != null ? s.color : i] },
      customdata: s.values.map((v) => [fmt(v[0]), fmt(v[1])]),
      hovertemplate: `%{x} · ${s.label}<br>%{customdata[0]} ± %{customdata[1]}${opts.unit || ""}<extra></extra>`,
    }));
    Plotly.react(el, traces, layout(t, {
      title: { text: opts.title }, barmode: "group", bargap: 0.35, bargroupgap: 0.08, showlegend: true,
      yaxis: { type: opts.log ? "log" : "linear", title: { text: opts.yTitle || "" } },
    }), { displayModeBar: false, responsive: true });
  }

  /* Lines with error bars. series = [{label, points:[[x, variant], ...], dash}] */
  function lines(el, series, key, opts) {
    const t = theme();
    const traces = series.map((s, i) => {
      const vals = s.points.map(([, n, k]) => metric(n, k || key));
      return {
        type: "scatter", mode: "lines+markers", name: s.label,
        x: s.points.map(([x]) => x), y: vals.map((v) => v[0]),
        error_y: { type: "data", array: vals.map((v) => v[1]), color: t.series[s.color != null ? s.color : i], thickness: 1, width: 3 },
        line: { color: t.series[s.color != null ? s.color : i], width: 2, dash: s.dash || "solid" },
        marker: { size: 8, color: t.series[s.color != null ? s.color : i], line: { color: t.card, width: 2 } },
        customdata: vals.map((v) => [fmt(v[0]), fmt(v[1])]),
        hovertemplate: `<b>${s.label}</b><br>${opts.xName || "x"} = %{x}<br>%{customdata[0]} ± %{customdata[1]}<extra></extra>`,
      };
    });
    const shapes = [];
    if (opts.band) shapes.push({ type: "rect", x0: opts.band[0], x1: opts.band[1], yref: "paper", y0: 0, y1: 1,
      fillcolor: t.grid, opacity: 0.55, line: { width: 0 }, layer: "below" });
    Plotly.react(el, traces, layout(t, {
      title: { text: opts.title }, showlegend: true, shapes,
      xaxis: { type: opts.logx ? "log" : "linear", title: { text: opts.xTitle || "" }, tickvals: opts.tickvals },
      yaxis: { type: opts.logy ? "log" : "linear", title: { text: opts.yTitle || "" }, range: opts.yrange },
      annotations: opts.bandLabel ? [{ x: (opts.band[0] + opts.band[1]) / 2, yref: "paper", y: 1.0, text: opts.bandLabel,
        showarrow: false, font: { size: 11, color: t.muted } }] : [],
    }), { displayModeBar: false, responsive: true });
  }

  /* Curves over rollout steps. series = [{label, y, dash, color}] */
  function curves(el, series, opts) {
    const t = theme();
    const traces = series.map((s, i) => ({
      type: "scatter", mode: "lines", name: s.label,
      x: s.y.map((_, k) => k + 1), y: s.y.map((v) => v * (opts.scale || 1)),
      line: { color: t.series[s.color != null ? s.color : i], width: s.dash ? 1.4 : 2.2, dash: s.dash || "solid" },
      hovertemplate: `<b>${s.label}</b><br>step %{x}: %{y:.3g}${opts.unit || ""}<extra></extra>`,
    }));
    const shapes = opts.hline != null ? [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: opts.hline, y1: opts.hline,
      line: { color: t.muted, dash: "dash", width: 1 } }] : [];
    Plotly.react(el, traces, layout(t, {
      title: { text: opts.title }, showlegend: true, shapes, hovermode: "x unified",
      xaxis: { title: { text: opts.xTitle || "rollout step (0.01 s each)" } },
      yaxis: { type: opts.logy ? "log" : "linear", title: { text: opts.yTitle || "" }, range: opts.yrange },
    }), { displayModeBar: false, responsive: true });
  }

  /* Numbers table for one experiment: rows = its variants, columns = [[key, title, scale], ...] */
  function table(el, exp, columns) {
    const head = "<tr><th>Variant</th><th>D</th>" + columns.map((c) => `<th>${c[1]}</th>`).join("") + "</tr>";
    const rows = Object.entries(exp.variants).map(([label, name]) => {
      const dim = V[name] ? V[name].settings.state_dim : "–";
      const cells = columns.map(([key, , scale]) => {
        const [m, ci] = metric(name, key);
        return `<td class="n">${isFinite(m) ? fmt(m * (scale || 1)) + " <span class='ci'>± " + fmt(ci * (scale || 1)) + "</span>" : "–"}</td>`;
      });
      return `<tr><td>${label}</td><td class="n">${dim}</td>${cells.join("")}</tr>`;
    });
    el.innerHTML = `<table><thead>${head}</thead><tbody>${rows.join("")}</tbody></table>`;
  }

  window.Charts = { metric, curve, bars, grouped, lines, curves, table };

  // Re-draw every chart when the theme changes.
  const redraws = [];
  window.chart = (fn) => { redraws.push(fn); fn(); };
  const redraw = () => redraws.forEach((fn) => fn());
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();

"use strict";
const $ = (id) => document.getElementById(id);
const API = ""; // same origin: the page is served by the API it calls

const ATTR = [
  ["job_level","num",1,7,3],["tenure_months","num",0,600,30],["total_experience_years","num",0,60,6.5],["previous_employers_count","num",0,40,2],
  ["commute_distance_km","num",0,500,12],["annual_fixed_pay","num",0,1e9,900000],["comp_ratio","num",0,10,0.95],["raises_received","num",0,80,2],
  ["last_raise_pct","num",0,300,6],["months_since_last_raise","num",0,600,14],["promotions_received","num",0,40,1],["months_since_last_promotion","num",0,600,20],
  ["internal_transfers","num",0,40,0],["last_performance_rating","num",0,10,3],["last_engagement_score","num",0,10,3.4],["manager_changes_12m","num",0,24,1],
  ["training_hours_12m","num",0,3000,18],["attendance_rate_3m","num",0,1,0.93],["late_arrivals_3m","num",0,100,4],["avg_overtime_hours_per_month_3m","num",0,400,8],
  ["avg_hours_worked_3m","num",0,24,8.4],["wfh_share_3m","num",0,1,0.2],["unpaid_leave_days_6m","num",0,200,1],["paid_leave_days_6m","num",0,200,5],
  ["attendance_rate_change_3m","num",-1,1,-0.02],["avg_unpaid_absence_days_3m","num",0,31,0.3],["avg_variable_pay_3m","num",0,1e8,5000],
  ["department_id","cat",0,0,"D01"],["role_id","cat",0,0,"R001"],["work_location","cat",0,0,"Bengaluru"],["work_mode","cat",0,0,"Hybrid"],
  ["employment_type","cat",0,0,"Full-time"],["highest_education","cat",0,0,"Bachelors"],
];
const PB = [["job_level","num",1,7,3],["prior_experience_years","num",0,60,5],["market_index","num",0.1,10,1.2],["role_id","cat",0,0,"R001"],
  ["department_id","cat",0,0,"D01"],["work_location","cat",0,0,"Bengaluru"],["highest_education","cat",0,0,"Bachelors"]];

const label = (n) => n.replace(/_/g, " ");
function buildFields(host, defs) {
  defs.forEach(([name, kind, lo, hi]) => {
    const l = document.createElement("label"); l.append(label(name));
    const i = document.createElement("input"); i.name = name; i.dataset.kind = kind;
    if (kind === "num") { i.type = "number"; i.step = "any"; i.min = lo; i.max = hi; i.placeholder = `${lo} – ${hi}`; } else { i.type = "text"; i.maxLength = 64; }
    const e = document.createElement("span"); e.className = "err"; l.append(i, e); host.append(l);
  });
}
function fillSample(form, defs) { defs.forEach(([n, , , , v]) => { form.elements[n].value = v; }); }
function collect(form, defs) {
  const out = {}; let ok = true;
  defs.forEach(([n, kind, lo, hi]) => {
    const el = form.elements[n], err = el.nextElementSibling; el.classList.remove("invalid"); err.textContent = "";
    const raw = el.value.trim(); if (raw === "") return;
    if (kind === "num") {
      const v = Number(raw);
      if (!Number.isFinite(v) || v < lo || v > hi) { ok = false; el.classList.add("invalid"); err.textContent = `Enter ${lo} – ${hi}`; return; }
      out[n] = v;
    } else out[n] = raw;
  });
  return ok ? out : null;
}

async function call(path, body) {
  const key = $("apiKey").value.trim();
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 30000);
  try {
    const r = await fetch(API + path, { method: body ? "POST" : "GET", signal: ctl.signal,
      headers: { "Content-Type": "application/json", ...(key ? { "X-API-Key": key } : {}) }, body: body ? JSON.stringify(body) : undefined });
    let j = null; try { j = await r.json(); } catch (_) {}
    if (!r.ok) {
      const e = j && j.error ? j.error.message : `HTTP ${r.status}`;
      if (r.status === 401) {
        throw new Error(key
          ? "The API key was rejected. Verify it matches a current CORTEXHR_API_KEYS value on the API host."
          : "Paste the raw key configured as CORTEXHR_API_KEYS on the API host into the API key field above.");
      }
      throw new Error(e);
    }
    return j;
  } catch (e) { throw new Error(e.name === "AbortError" ? "The request timed out (the server may be waking up). Please retry." : e.message); }
  finally { clearTimeout(t); }
}
function busy(btn, on) { btn.disabled = on; if (on) { btn.dataset.t = btn.textContent; btn.textContent = "Working…"; } else btn.textContent = btn.dataset.t || btn.textContent; }
function show(box, node, isErr) { box.hidden = false; box.className = "result" + (isErr ? " error" : ""); box.replaceChildren(node); }
const txt = (tag, t, cls) => { const e = document.createElement(tag); e.textContent = t; if (cls) e.className = cls; return e; };
function table(rows, head) {
  const t = document.createElement("table"); if (head) { const r = t.insertRow(); head.forEach((h) => { const c = document.createElement("th"); c.textContent = h; r.append(c); }); }
  rows.forEach((row) => { const r = t.insertRow(); row.forEach((v) => { r.insertCell().textContent = v; }); }); return t;
}

async function refreshStatus() {
  try {
    const h = await (await fetch(API + "/v1/health")).json();
    $("dot").className = "dot " + (h.all_models_loaded ? "ok" : "warn");
    $("statusText").textContent = h.all_models_loaded ? "API online · all models loaded" : `API online · ${h.models_unavailable.length} model(s) unavailable`;
    $("models").replaceChildren(...Object.entries(h.models).map(([n, m]) => txt("span", `${label(n)}: ${m.status}${m.mode ? " (" + m.mode + ")" : ""}`, "chip " + (m.status === "loaded" ? "ok" : "bad"))));
  } catch (_) { $("dot").className = "dot bad"; $("statusText").textContent = "API unreachable"; }
}

$("nav").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  document.querySelectorAll("nav button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".panel").forEach((p) => { p.hidden = p.id !== "tab-" + b.dataset.tab; });
});
const saved = sessionStorage.getItem("cortexhr_key"); if (saved) $("apiKey").value = saved;
$("apiKey").addEventListener("change", () => sessionStorage.setItem("cortexhr_key", $("apiKey").value.trim()));

buildFields($("attrFields"), ATTR); buildFields($("pbFields"), PB);
$("attrSample").onclick = () => fillSample($("attrForm"), ATTR);
$("attrForm").addEventListener("submit", async (ev) => {
  ev.preventDefault(); const body = collect($("attrForm"), ATTR); if (!body) return;
  const btn = ev.submitter; busy(btn, true); show($("attrOut"), txt("span", "Scoring…"));
  try {
    const r = await call("/v1/attrition/predict", { records: [body] }); const p = r.results[0].attrition_probability_12m;
    const box = document.createElement("div");
    box.append(txt("div", (p * 100).toFixed(1) + "%", "big"), txt("div", "estimated probability of leaving within 12 months"), txt("p", `Model ${r.model_version}. ${r.risk_band_note}`));
    show($("attrOut"), box);
  } catch (e) { show($("attrOut"), txt("span", e.message), true); } finally { busy(btn, false); }
});

$("chatForm").addEventListener("submit", async (ev) => {
  ev.preventDefault(); const m = $("chatMsg").value.trim(); if (!m) return; $("chatMsg").value = "";
  const log = $("chatLog"); log.append(txt("div", m, "msg me"));
  const btn = ev.submitter; busy(btn, true);
  try {
    const r = await call("/v1/chat", { message: m }); const d = txt("div", r.reply, "msg bot");
    d.append(txt("small", r.fallback ? `fallback (${r.fallback_reason})` : `intent: ${r.intent} · confidence ${(r.confidence * 100).toFixed(0)}%`)); log.append(d);
  } catch (e) { log.append(txt("div", e.message, "msg bot")); } finally { busy(btn, false); log.scrollTop = log.scrollHeight; }
});

$("fcForm").addEventListener("submit", async (ev) => {
  ev.preventDefault(); const btn = ev.submitter; busy(btn, true);
  try {
    const r = await call("/v1/forecast/headcount", { department_id: $("fcDept").value.trim() || "ALL", horizon_months: Number($("fcH").value) });
    const box = document.createElement("div");
    box.append(txt("strong", `${r.mode === "live" ? "Live forecast" : "PRECOMPUTED result (not live)"} · ${r.department_id} · trained through ${r.trained_through}`),
      table(r.forecast.map((p) => [p.month, p.expected_headcount, p.lower_80, p.upper_80]), ["Month", "Expected", "Lower 80%", "Upper 80%"]), txt("p", r.note));
    show($("fcOut"), box);
  } catch (e) { show($("fcOut"), txt("span", e.message), true); } finally { busy(btn, false); }
});

$("pbForm").addEventListener("submit", async (ev) => {
  ev.preventDefault(); const body = collect($("pbForm"), PB); if (!body) return;
  if (body.job_level === undefined) { $("pbForm").elements.job_level.classList.add("invalid"); return; }
  const btn = ev.submitter; busy(btn, true);
  try {
    const r = await call("/v1/payband/analyze", body); const fmt = (v) => (v == null ? "n/a" : "INR " + v.toLocaleString("en-IN"));
    const box = document.createElement("div");
    box.append(txt("div", fmt(r.predicted_salary_mid), "big"), txt("div", `Band: ${fmt(r.recommended_salary_low)} – ${fmt(r.recommended_salary_high)}`), txt("p", r.band_status), txt("p", r.decision_note));
    show($("pbOut"), box);
  } catch (e) { show($("pbOut"), txt("span", e.message), true); } finally { busy(btn, false); }
});

refreshStatus(); setInterval(refreshStatus, 60000);

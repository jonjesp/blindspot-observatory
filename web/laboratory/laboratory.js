import { inspectProbe, resetViewer, setHoldoutViews, showCutout, showDetections } from "./laboratory-viewer.js";

const element = (id) => document.getElementById(id);
const text = (id, value) => { element(id).textContent = value; };
const show = (id, visible) => { element(id).hidden = !visible; };
const percent = (value) => `${(100 * value).toFixed(2)}%`;
const percentagePoints = (value) => `${(100 * value).toFixed(2)} pp`;
const usd = (value) => `USD ${value.toFixed(6)}`;
const probeNames = { probe_sky: "Sky background", probe_psf: "Point-spread function", probe_flux: "Source flux" };
const probeOrder = (names) => names.map((name) => probeNames[name] ?? name).join(" → ") || "None";

let catalog;
let busy = false;

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error);
  return payload;
}

function setStatus(message, failed = false) {
  text("status", message);
  element("status").classList.toggle("error", failed);
}

function setOptions(id, choices) {
  element(id).replaceChildren(...choices.map(([value, label]) => new Option(label, value)));
}

function selectedModel() {
  return catalog.models.find(({ model }) => model === element("model").value);
}

function selectCases() {
  const exposure = element("field").value;
  setOptions("case", catalog.cases.filter(({ field }) => field === exposure).map(
    ({ id, offset_x, offset_y }) => [id, `${id} · x ${offset_x}, y ${offset_y}`],
  ));
}

function updateControls() {
  const cid = element("case").value;
  const profile = selectedModel();
  const live = element("mode").value === "live";
  const reservation = profile.reservations_usd[cid];
  const duplicate = catalog.live_attempts.includes(`${profile.model}/${cid}`);
  const canRunLive = catalog.live_enabled && !duplicate && reservation <= catalog.remaining_usd;
  const canReplay = profile.archived_cases.includes(cid);
  text("selection", `${element("field").value} / ${cid} · ${profile.name} · ${live ? "Live" : "Archive"}`);
  const detail = live
    ? `Estimated reservation ${usd(reservation)}. Remaining session allowance ${usd(catalog.remaining_usd)}.`
    : "Recompute pixel measurements and replay the archived decision. Zero provider calls.";
  text("mode-detail", live && duplicate ? "This model and cutout already ran in the current session."
    : live && !catalog.live_enabled ? "Live requests are unavailable in this session."
    : !live && !canReplay ? "No archived response is available for this selection." : detail);
  for (const id of ["field", "case", "model", "mode", "confirm-live"]) element(id).disabled = busy;
  element("mode").querySelector('[value="live"]').disabled = !catalog.live_enabled && !live;
  show("live-consent", live);
  text("run", busy ? "Analysis running" : "Run analysis");
  element("run").disabled = busy || (live ? !canRunLive || !element("confirm-live").checked : !canReplay);
}

function recoveryBar(label, count, total, isReference) {
  const row = document.createElement("div");
  row.className = "bar-row";
  const name = document.createElement("span");
  name.textContent = label;
  const bar = document.createElement("progress");
  bar.className = isReference ? "bar-reference" : "bar-measurement";
  bar.setAttribute("aria-hidden", "true");
  bar.max = total;
  bar.value = count;
  const value = document.createElement("span");
  value.className = "bar-value";
  value.textContent = `${count}/${total}`;
  row.append(name, bar, value);
  return row;
}

function chartGroup(title, tag = "section") {
  const group = document.createElement(tag);
  group.className = "chart-group";
  const heading = document.createElement("h3");
  heading.textContent = title;
  group.append(heading);
  return group;
}

function renderPilots(observation) {
  const { recovery_counts: counts, sources_per_batch: sources } = observation.pilot_measurements;
  element("pilot-chart").replaceChildren(...Object.entries(counts).map(([band, { A, B }]) => {
    const group = chartGroup(band[0].toUpperCase() + band.slice(1));
    const sum = (batches) => batches.reduce((total, count) => total + count, 0);
    group.append(
      recoveryBar("Arm A", sum(A), sources * A.length, true),
      recoveryBar("Arm B", sum(B), sources * B.length, false),
    );
    return group;
  }));
  text("indicators", Object.entries(observation.indicators).map(([name, value]) =>
    `${name.replaceAll("_", " ")} ${percent(value)}`,
  ).join(" · "));
  text("observation", JSON.stringify(observation, null, 2));
  show("pilot-empty", false);
  show("pilots", true);
}

function renderReply(decision) {
  text("reply-decision", decision ? decision.decision : "Invalid response");
  text("justification", decision?.justification ?? "The response failed protocol validation.");
  text("reply-probes", `Selected probes · ${probeOrder(decision?.probes ?? [])}`);
  show("reply-empty", false);
  show("reply", true);
}

function renderProgress(run) {
  for (const stage of document.querySelectorAll("[data-stage]")) {
    const active = stage.dataset.stage === run.stage;
    stage.classList.toggle("done", run.stages.includes(stage.dataset.stage) && !active);
    if (active) stage.setAttribute("aria-current", "step");
    else stage.removeAttribute("aria-current");
  }
  const messages = {
    queued: "Analysis queued.",
    image: "Verifying FITS input.",
    measurement: "Computing pilot and holdout measurements.",
    decision: element("mode").value === "live" ? "Awaiting the provider response." : "Validating the archived response.",
    scoring: "Scoring both policies and tracing detector matches.",
    complete: "Analysis complete.",
  };
  setStatus(run.error ?? messages[run.stage], run.state === "failed");
  if (run.image && element("cutout").hidden) showCutout(run.image);
  if (run.observation && element("pilots").hidden) {
    renderPilots(run.observation);
    showDetections(run.original_detections);
  }
  if (run.model_reply && element("reply").hidden) renderReply(run.model_reply.decision);
}

function outcome(trace, isControl) {
  if (isControl) return trace.success ? "Correct abstention" : "False alarm";
  return trace.success ? "Case closed" : "Case unresolved";
}

function renderProvenance(result) {
  const response = result.response;
  const usage = response.usage;
  const entries = [
    ["Provider calls this run", result.api_calls],
    ["Returned model", response.response_model],
    ["Response status", response.state.replaceAll("_", " ")],
    ["Finish reason", response.finish_reason ?? "Unavailable"],
    ["Request SHA-256", result.request_sha256],
    ["Prompt tokens", usage.prompt_tokens],
    ["Cached prompt tokens", usage.cached_tokens],
    ["Completion tokens", usage.completion_tokens],
    ["Reasoning tokens", response.reasoning_tokens ?? "Unavailable"],
    ["Provider time", `${response.elapsed_seconds.toFixed(2)} s`],
    ["Calculated provider cost", usd(response.cost_usd)],
  ];
  text("request-scope", result.mode === "replay"
    ? "Token counts, provider time, and cost describe the historical request."
    : "Provider time covers the model request. Cost is calculated from the reported token usage.");
  element("provenance").replaceChildren(...entries.flatMap(([label, value]) => {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    return [term, description];
  }));
}

function renderHoldout(result) {
  const entries = Object.entries(result.measurements);
  show("empty-probes", entries.length === 0);
  setHoldoutViews(result.holdout_views, probeNames);
  element("holdout-chart").replaceChildren(...entries.map(([probe, measurement]) => {
    const group = chartGroup(probeNames[probe], "article");
    const summary = document.createElement("p");
    summary.className = "probe-summary";
    summary.textContent = `Gain ${percentagePoints(measurement.gain)} · lower bound ${percentagePoints(measurement.lower_bound)} · ${measurement.closed ? "criterion met" : "criterion unmet"}`;
    const inspect = document.createElement("button");
    inspect.type = "button";
    inspect.className = "inspect";
    inspect.textContent = "Inspect batches";
    inspect.setAttribute("aria-label", `Inspect batches for ${probeNames[probe]}`);
    inspect.addEventListener("click", () => inspectProbe(probe));
    group.append(
      recoveryBar("Before", measurement.before, measurement.n, true),
      recoveryBar("After", measurement.after, measurement.n, false),
      summary, inspect,
    );
    return group;
  }));
}

function renderResult(result) {
  const control = result.definition.kind === "no_intervention";
  const decision = result.decision;
  text("run-source", result.mode === "replay" ? "Archived response" : "Live provider response");
  text("case-summary", `${result.definition.id} · ${result.definition.field} · ${result.definition.kind.replaceAll("_", " ")}`);
  text("model-title", selectedModel().name);
  text("decision", outcome(result.model_trace, control));
  text("model-trace", `Selected: ${probeOrder(decision?.probes ?? [])}. Executed: ${probeOrder(result.model_trace.executed)}. Continuation: ${probeOrder(result.continuation)}.`);
  text("baseline-outcome", outcome(result.baseline_trace, control));
  text("baseline-trace", `Executed: ${probeOrder(result.baseline_trace.executed)}.`);
  text("threshold", `Closure requires a gain of at least ${percentagePoints(result.gain_threshold)} and a conservative lower bound above zero.`);
  renderHoldout(result);
  renderProvenance(result);
  show("results", true);
}

async function runAnalysis(event) {
  event.preventDefault();
  if (busy) return;
  busy = true;
  const mode = element("mode").value;
  const payload = {
    case_id: element("case").value, model: element("model").value, mode,
    confirm_live: element("confirm-live").checked,
  };
  element("configuration").open = false;
  updateControls();
  const startedAt = performance.now();
  const updateElapsed = () => text("elapsed", `${((performance.now() - startedAt) / 1000).toFixed(1)} s`);
  updateElapsed();
  show("elapsed", true);
  const timer = setInterval(updateElapsed, 250);
  text("decision-stage", mode === "live" ? "Request live decision" : "Validate archived decision");
  show("results", false);
  resetViewer();
  show("pilots", false);
  show("pilot-empty", true);
  show("reply", false);
  show("reply-empty", true);
  renderProgress({ stage: "queued", stages: [] });
  try {
    const { id } = await api("/api/runs", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    let run;
    do {
      run = await api(`/api/runs/${id}`);
      renderProgress(run);
      if (run.state === "running") await new Promise((resolve) => setTimeout(resolve, 400));
    } while (run.state === "running");
    if (run.state === "complete") renderResult(run.result);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    clearInterval(timer);
    updateElapsed();
    try {
      catalog = await api("/api/catalog");
    } catch (error) {
      setStatus(error.message, true);
    }
    element("confirm-live").checked = false;
    busy = false;
    updateControls();
  }
}

async function initialize() {
  try {
    catalog = await api("/api/catalog");
    const fields = [...new Set(catalog.cases.map(({ field }) => field))];
    setOptions("field", fields.map((field) => [field, field]));
    selectCases();
    setOptions("model", catalog.models.map(({ model, name }) => [model, name]));
    element("model").value = "muse-spark-1.3";
    element("field").addEventListener("change", selectCases);
    element("setup").addEventListener("change", (event) => {
      if (event.target.id !== "confirm-live") element("confirm-live").checked = false;
      updateControls();
    });
    element("setup").addEventListener("submit", runAnalysis);
    updateControls();
    setStatus("Ready to analyze a case.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

initialize();

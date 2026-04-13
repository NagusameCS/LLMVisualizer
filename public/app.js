/* ── DOM refs ────────────────────────────────────────── */
const modelInput = document.getElementById("modelInput");
const visualizeBtn = document.getElementById("visualizeBtn");
const resultSection = document.getElementById("result");
const statusSection = document.getElementById("status");

const modelKey = document.getElementById("modelKey");
const titleEl = document.getElementById("title");
const descriptionEl = document.getElementById("description");
const variantCountEl = document.getElementById("variantCount");

const selectedVariantEl = document.getElementById("selectedVariant");
const selectedSizeEl = document.getElementById("selectedSize");
const selectedContextEl = document.getElementById("selectedContext");
const selectedInputTypeEl = document.getElementById("selectedInputType");

const barsEl = document.getElementById("bars");
const variantRowsEl = document.getElementById("variantRows");
const noteEl = document.getElementById("note");

const singleInput = document.getElementById("singleInput");
const compareInput = document.getElementById("compareInput");
const compareA = document.getElementById("compareA");
const compareB = document.getElementById("compareB");
const compareBtn = document.getElementById("compareBtn");
const compareResult = document.getElementById("compareResult");
const compareGrid = document.getElementById("compareGrid");

let lastPayload = null;
let lastComparePayload = null;

/* ── Mode tabs ───────────────────────────────────────── */
document.querySelectorAll(".mode-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".mode-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const mode = tab.dataset.mode;
    singleInput.classList.toggle("hidden", mode !== "single");
    compareInput.classList.toggle("hidden", mode !== "compare");
    resultSection.classList.add("hidden");
    compareResult.classList.add("hidden");
    clearStatus();
  });
});

/* ── Helpers ─────────────────────────────────────────── */
function setStatus(message, isError = false) {
  statusSection.classList.remove("hidden");
  statusSection.classList.toggle("error", isError);
  statusSection.textContent = message;
}

function clearStatus() {
  statusSection.classList.add("hidden");
  statusSection.classList.remove("error");
  statusSection.textContent = "";
}

function safeText(value, fallback = "-") {
  return value && String(value).trim() ? value : fallback;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/* ── URL state (shareable bookmarks) ─────────────────── */
function pushUrlState(params) {
  const url = new URL(window.location);
  url.search = "";
  for (const [key, val] of Object.entries(params)) {
    if (val) url.searchParams.set(key, val);
  }
  history.pushState(null, "", url);
}

function readUrlState() {
  const params = new URLSearchParams(window.location.search);
  return {
    model: params.get("model") || "",
    a: params.get("a") || "",
    b: params.get("b") || ""
  };
}

/* ── Bar chart renderer ──────────────────────────────── */
function renderBars(variants, container) {
  const target = container || barsEl;
  target.innerHTML = "";
  const maxSize = Math.max(...variants.map((v) => v.sizeGb || 0), 0);

  for (const variant of variants.slice(0, 18)) {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("p");
    label.className = "bar-label";
    label.textContent = variant.tag;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    const pct = maxSize > 0 && variant.sizeGb ? (variant.sizeGb / maxSize) * 100 : 12;
    fill.style.width = `${Math.max(pct, 6)}%`;
    track.appendChild(fill);

    const value = document.createElement("p");
    value.className = "bar-value";
    value.textContent = safeText(variant.size);

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    target.appendChild(row);
  }
}

/* ── Variant table renderer ──────────────────────────── */
function renderTable(variants, selectedName) {
  variantRowsEl.innerHTML = "";

  for (const variant of variants) {
    const tr = document.createElement("tr");
    if (selectedName && variant.fullName === selectedName) {
      tr.className = "focus";
    }

    tr.innerHTML = `
      <td>${escapeHtml(safeText(variant.tag))}</td>
      <td>${escapeHtml(safeText(variant.size))}</td>
      <td>${escapeHtml(safeText(variant.contextWindow))}</td>
      <td>${escapeHtml(safeText(variant.inputType))}</td>
      <td>${escapeHtml(safeText(variant.updated))}</td>
    `;

    variantRowsEl.appendChild(tr);
  }
}

/* ── Single model renderer ───────────────────────────── */
function renderResult(data) {
  const selected = data.selectedVariant || {};

  modelKey.textContent = `${data.model}${data.tag ? `:${data.tag}` : ""}`;
  titleEl.textContent = safeText(data.readmeTitle || data.title, data.model);
  descriptionEl.textContent = safeText(data.description);
  variantCountEl.textContent = `${data.totalVariants || 0} variants`;

  selectedVariantEl.textContent = safeText(selected.fullName);
  selectedSizeEl.textContent = safeText(selected.size);
  selectedContextEl.textContent = safeText(selected.contextWindow);
  selectedInputTypeEl.textContent = safeText(selected.inputType);

  renderBars(data.variants || []);
  renderTable(data.variants || [], selected.fullName);

  noteEl.textContent = safeText(data.note, "Metadata only.");
  resultSection.classList.remove("hidden");
  compareResult.classList.add("hidden");
}

/* ── Fetch model data ────────────────────────────────── */
async function fetchModel(url) {
  const response = await fetch(`/api/model?url=${encodeURIComponent(url)}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Could not fetch model details.");
  return payload;
}

/* ── Single visualize ────────────────────────────────── */
async function visualize(inputOverride) {
  const inputValue = (inputOverride || modelInput.value).trim();
  if (!inputValue) {
    setStatus("Paste an Ollama model link first.", true);
    return;
  }

  modelInput.value = inputValue;
  visualizeBtn.disabled = true;
  setStatus("Fetching model metadata from Ollama...");

  try {
    const data = await fetchModel(inputValue);
    lastPayload = data;
    clearStatus();
    renderResult(data);
    pushUrlState({ model: inputValue });
  } catch (error) {
    setStatus(error.message || "Unexpected error.", true);
  } finally {
    visualizeBtn.disabled = false;
  }
}

/* ── Comparison renderer ─────────────────────────────── */
function renderComparePanel(data, side) {
  const s = side; // "A" or "B"
  document.getElementById(`cmpKey${s}`).textContent = data.model;
  document.getElementById(`cmpTitle${s}`).textContent = safeText(data.readmeTitle || data.title, data.model);
  document.getElementById(`cmpDesc${s}`).textContent = safeText(data.description);
  document.getElementById(`cmpCount${s}`).textContent = `${data.totalVariants || 0} variants`;

  const sel = data.selectedVariant || {};
  document.getElementById(`cmpSize${s}`).textContent = safeText(sel.size);
  document.getElementById(`cmpCtx${s}`).textContent = safeText(sel.contextWindow);
  document.getElementById(`cmpInput${s}`).textContent = safeText(sel.inputType);
  document.getElementById(`cmpParams${s}`).textContent = (data.parameterHints || []).join(", ") || "-";

  renderBars(data.variants || [], document.getElementById(`cmpBars${s}`));
}

async function runCompare(aOverride, bOverride) {
  const urlA = (aOverride || compareA.value).trim();
  const urlB = (bOverride || compareB.value).trim();
  if (!urlA || !urlB) {
    setStatus("Provide two Ollama model links to compare.", true);
    return;
  }

  compareA.value = urlA;
  compareB.value = urlB;
  compareBtn.disabled = true;
  setStatus("Fetching both models...");

  try {
    const [dataA, dataB] = await Promise.all([fetchModel(urlA), fetchModel(urlB)]);
    lastComparePayload = { a: dataA, b: dataB };
    clearStatus();
    renderComparePanel(dataA, "A");
    renderComparePanel(dataB, "B");
    resultSection.classList.add("hidden");
    compareResult.classList.remove("hidden");
    pushUrlState({ a: urlA, b: urlB });
  } catch (error) {
    setStatus(error.message || "Unexpected error.", true);
  } finally {
    compareBtn.disabled = false;
  }
}

/* ── Export: JSON ─────────────────────────────────────── */
function downloadJson(payload, filename) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ── Export: PNG (uses html2canvas) ───────────────────── */
async function downloadPng(element, filename) {
  if (typeof html2canvas === "undefined") {
    setStatus("html2canvas library not loaded yet. Try again in a moment.", true);
    return;
  }
  const canvas = await html2canvas(element, {
    backgroundColor: "#f3efe3",
    scale: 2,
    useCORS: true
  });
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = filename;
  a.click();
}

/* ── Share (copy link) ───────────────────────────────── */
function copyShareLink() {
  navigator.clipboard.writeText(window.location.href).then(() => {
    setStatus("Shareable link copied to clipboard!");
    setTimeout(clearStatus, 2500);
  });
}

/* ── Event listeners ─────────────────────────────────── */
visualizeBtn.addEventListener("click", () => visualize());
modelInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); visualize(); }
});

compareBtn.addEventListener("click", () => runCompare());
[compareA, compareB].forEach((input) => {
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); runCompare(); }
  });
});

// Single-model export buttons
document.getElementById("exportJson").addEventListener("click", () => {
  if (lastPayload) downloadJson(lastPayload, `${lastPayload.model}-metadata.json`);
});
document.getElementById("exportPng").addEventListener("click", () => {
  downloadPng(resultSection, `${lastPayload?.model || "model"}-visualization.png`);
});
document.getElementById("shareBtn").addEventListener("click", copyShareLink);

// Compare export buttons
document.getElementById("cmpExportJson").addEventListener("click", () => {
  if (lastComparePayload) downloadJson(lastComparePayload, "model-comparison.json");
});
document.getElementById("cmpExportPng").addEventListener("click", () => {
  downloadPng(compareGrid, "model-comparison.png");
});
document.getElementById("cmpShareBtn").addEventListener("click", copyShareLink);

/* ── Boot: restore from URL state ────────────────────── */
(function boot() {
  const state = readUrlState();

  if (state.a && state.b) {
    // Activate compare mode
    document.querySelector('[data-mode="compare"]').click();
    compareA.value = state.a;
    compareB.value = state.b;
    runCompare(state.a, state.b);
  } else if (state.model) {
    modelInput.value = state.model;
    visualize(state.model);
  } else {
    visualize();
  }
})();

window.addEventListener("popstate", () => {
  const state = readUrlState();
  if (state.a && state.b) {
    document.querySelector('[data-mode="compare"]').click();
    compareA.value = state.a;
    compareB.value = state.b;
    runCompare(state.a, state.b);
  } else if (state.model) {
    document.querySelector('[data-mode="single"]').click();
    modelInput.value = state.model;
    visualize(state.model);
  }
});
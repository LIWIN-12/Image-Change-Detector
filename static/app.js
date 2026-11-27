const $ = (id) => document.getElementById(id);

const BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";

function status(id, text, isError = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function bindPreview(inputId, imgId) {
  const input = $(inputId);
  const img = $(imgId);
  input.addEventListener("change", () => {
    const file = input.files[0];
    if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
    img.classList.remove("loaded");
    img.removeAttribute("src");
    if (!file) return;
    img.dataset.url = URL.createObjectURL(file);
    img.src = img.dataset.url;
    img.classList.add("loaded");
  });
}

["before-input:before-preview", "after-input:after-preview",
 "tamper-input:tamper-preview", "ai-input:ai-preview"].forEach((pair) => {
  const [input, img] = pair.split(":");
  bindPreview(input, img);
});

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => {
      b.classList.toggle("active", b === button);
    });
    document.querySelectorAll(".page").forEach((page) => {
      page.classList.toggle("active", page.id === button.dataset.tab);
    });
  });
});

[["threshold", "threshold-value"], ["min-area", "min-area-value"]].forEach(([input, output]) => {
  $(input).addEventListener("input", () => { $(output).textContent = $(input).value; });
});

function setImage(id, src) {
  const img = $(id);
  img.src = src || BLANK;
}

function tags(id, items, warn) {
  const box = $(id);
  box.textContent = "";
  if (!items.length) {
    const tag = document.createElement("span");
    tag.className = "tag ok";
    tag.textContent = "None found";
    box.appendChild(tag);
    return;
  }
  items.forEach((item) => {
    const tag = document.createElement("span");
    tag.className = warn ? "tag warn" : "tag";
    tag.textContent = item;
    box.appendChild(tag);
  });
}

function renderTable(rows) {
  const body = $("region-table").querySelector("tbody");
  body.textContent = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    ["x", "y", "w", "h", "area", "mean_diff"].forEach((key) => {
      const td = document.createElement("td");
      td.textContent = row[key];
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

async function post(path, form) {
  const response = await fetch(path, { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

async function run(buttonId, statusId, resultId, path, form, render) {
  const button = $(buttonId);
  button.disabled = true;
  button.textContent = "Working…";
  status(statusId, "Analyzing…");
  $(resultId).hidden = true;
  try {
    const data = await post(path, form);
    render(data);
    status(statusId, "Done.");
    $(resultId).hidden = false;
  } catch (error) {
    status(statusId, error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = button.dataset.label;
  }
}

document.querySelectorAll("button.primary").forEach((button) => {
  button.dataset.label = button.textContent;
});

function renderCompare(data) {
  $("stat-regions").textContent = data.regions.length;
  $("stat-area").textContent = data.regions.reduce((max, r) => Math.max(max, r.area), 0);
  $("stat-diff").textContent = data.mean_diff;
  $("stat-aligned").textContent = data.aligned ? "Yes" : "No";
  setImage("out-marked", data.marked);
  setImage("out-mask", data.mask);
  setImage("out-after", data.after);
  setImage("out-diff", data.diff);
  renderTable(data.regions);
  $("marked-download").href = data.marked;
}

function renderTamper(data) {
  $("stat-ela").textContent = data.ela;
  $("stat-pairs").textContent = data.pairs;
  $("stat-ratio").textContent = data.ratio;
  setImage("tamper-heat", data.heat);
  setImage("tamper-noise", data.noise);
  setImage("tamper-marked", data.marked);
  if (!data.is_jpeg) status("tamper-status", "Done. ELA is most meaningful for JPEG uploads.");
}

function renderAI(data) {
  const verdict = $("ai-verdict");
  verdict.className = "verdict " + data.band.toLowerCase().replace(" ", "-");
  $("ai-band").textContent = data.band;
  $("ai-reason").textContent = data.reason;
  $("ai-probability").textContent = Math.round(data.probability * 100) + "%";
  $("ai-meter").style.width = Math.round(data.probability * 100) + "%";
  tags("ai-hits", data.hits, true);
  tags("ai-flags", data.flags, true);
  $("ai-smoothness").textContent = data.smoothness;
  $("ai-clipped").textContent = (data.clipped * 100).toFixed(2) + "%";
}

$("compare-run").addEventListener("click", async () => {
  const before = $("before-input").files[0];
  const after = $("after-input").files[0];
  if (!before || !after) return status("compare-status", "Choose both images.", true);
  const form = new FormData();
  form.append("before", before);
  form.append("after", after);
  form.append("threshold", $("threshold").value);
  form.append("min_area", $("min-area").value);
  form.append("align", $("align").checked ? "true" : "false");
  await run("compare-run", "compare-status", "compare-result", "/api/compare", form, renderCompare);
});

$("tamper-run").addEventListener("click", async () => {
  const file = $("tamper-input").files[0];
  if (!file) return status("tamper-status", "Choose an image.", true);
  const form = new FormData();
  form.append("image", file);
  await run("tamper-run", "tamper-status", "tamper-result", "/api/tamper", form, renderTamper);
});

$("ai-run").addEventListener("click", async () => {
  const file = $("ai-input").files[0];
  if (!file) return status("ai-status", "Choose an image.", true);
  const form = new FormData();
  form.append("image", file);
  await run("ai-run", "ai-status", "ai-result", "/api/ai", form, renderAI);
});

setImage("out-marked", BLANK);
setImage("out-mask", BLANK);
setImage("out-after", BLANK);
setImage("out-diff", BLANK);
setImage("tamper-heat", BLANK);
setImage("tamper-noise", BLANK);
setImage("tamper-marked", BLANK);
setImage("ai-image", BLANK);

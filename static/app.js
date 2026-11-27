const $ = (id) => document.getElementById(id);

const BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";

function status(id, text, isError = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function slotParts(slot) {
  return {
    input: $(slot + "-input"),
    image: $(slot + "-preview"),
    zone: document.querySelector('.dropzone[data-slot="' + slot + '"]'),
    prompt: document.querySelector('.dropzone[data-slot="' + slot + '"] .drop-prompt'),
    meta: $(slot + "-meta"),
    clear: $(slot + "-clear"),
  };
}

function resetSlot(slot) {
  const part = slotParts(slot);
  if (part.image.dataset.url) URL.revokeObjectURL(part.image.dataset.url);
  delete part.image.dataset.url;
  part.image.onload = null;
  part.image.onerror = null;
  part.input.value = "";
  part.image.hidden = true;
  part.image.removeAttribute("src");
  part.prompt.hidden = false;
  part.meta.textContent = "EMPTY SLOT :";
  part.clear.hidden = true;
  part.zone.classList.remove("dragging");
}

function loadSlot(slot, file) {
  if (!file || !file.type.startsWith("image/")) return resetSlot(slot);
  const part = slotParts(slot);
  if (part.image.dataset.url) URL.revokeObjectURL(part.image.dataset.url);
  part.image.onload = null;
  part.image.onerror = null;
  part.image.hidden = true;
  part.prompt.hidden = false;
  part.clear.hidden = true;
  part.meta.textContent = file.name.toUpperCase() + " / READING…";

  const url = URL.createObjectURL(file);
  part.image.dataset.url = url;
  part.image.onload = () => {
    part.meta.textContent = file.name.toUpperCase() + " / " +
      part.image.naturalWidth + "×" + part.image.naturalHeight;
    part.image.hidden = false;
    part.prompt.hidden = true;
    part.clear.hidden = false;
  };
  part.image.onerror = () => { resetSlot(slot); };
  part.image.src = url;
}

["before", "after", "tamper", "ai"].forEach((slot) => {
  const part = slotParts(slot);
  part.input.addEventListener("change", () => loadSlot(slot, part.input.files[0]));

  part.zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    part.zone.classList.add("dragging");
  });
  part.zone.addEventListener("dragleave", () => {
    part.zone.classList.remove("dragging");
  });
  part.zone.addEventListener("drop", (event) => {
    event.preventDefault();
    part.zone.classList.remove("dragging");
    const file = event.dataTransfer.files[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    part.input.files = transfer.files;
    part.input.dispatchEvent(new Event("change", { bubbles: true }));
  });

  part.clear.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    resetSlot(slot);
  });
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

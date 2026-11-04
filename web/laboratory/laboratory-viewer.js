const byId = (id) => document.getElementById(id);
const canvas = byId("cutout");
const context = canvas.getContext("2d");
const raster = document.createElement("canvas");
const rasterContext = raster.getContext("2d");
const pixelScale = 4;

let original;
let originalDetections;
let holdouts;
let displayed;

function drawRaster(frame) {
  raster.width = frame.width;
  raster.height = frame.height;
  const image = rasterContext.createImageData(frame.width, frame.height);
  const bytes = atob(frame.gray8);
  for (let index = 0; index < bytes.length; index += 1) {
    const offset = index * 4;
    const value = bytes.charCodeAt(index);
    image.data.set([value, value, value, 255], offset);
  }
  rasterContext.putImageData(image, 0, 0);
  displayed = frame;
}

function mark(positions, color, shape) {
  for (const [row, column] of positions) {
    const x = column + 0.5;
    const y = row + 0.5;
    context.beginPath();
    if (shape === "circle") context.arc(x, y, 2.1, 0, Math.PI * 2);
    if (shape === "square") context.rect(x - 2.8, y - 2.8, 5.6, 5.6);
    if (shape === "diamond") {
      context.moveTo(x, y - 3);
      context.lineTo(x + 3, y);
      context.lineTo(x, y + 3);
      context.lineTo(x - 3, y);
      context.closePath();
    }
    context.strokeStyle = "#0b1015";
    context.lineWidth = 0.65;
    context.stroke();
    context.strokeStyle = color;
    context.lineWidth = 0.29;
    context.stroke();
  }
}

function currentBatches() {
  const view = byId("frame").value;
  if (view === "before") return holdouts?.before;
  if (view === "after") return holdouts?.after_by_probe[byId("probe").value];
  return undefined;
}

function drawFrame() {
  if (!original) return;
  const batchIndex = Number(byId("batch").value);
  const batches = currentBatches();
  const frame = batches?.[batchIndex] ?? original;
  const isBatch = frame !== original;
  const detections = isBatch ? frame.detections : originalDetections ?? [];
  byId("probe-field").hidden = byId("frame").value !== "after";
  byId("batch-field").hidden = !isBatch;
  byId("injected-layer").hidden = !isBatch;
  byId("recovered-layer").hidden = !isBatch;

  if (displayed !== frame) drawRaster(frame);
  canvas.width = frame.width * pixelScale;
  canvas.height = frame.height * pixelScale;
  context.scale(pixelScale, pixelScale);
  context.imageSmoothingEnabled = false;
  context.drawImage(raster, 0, 0);
  if (byId("layer-detections").checked) mark(detections, "#9db8eb", "circle");
  if (isBatch && byId("layer-injected").checked) mark(frame.injected, "#ffc982", "diamond");
  if (isBatch && byId("layer-recovered").checked) {
    mark(frame.matches.map(([, detection]) => detections[detection]), "#8ee0ce", "square");
  }

  const detail = isBatch ? describeBatch(frame, batches, batchIndex) : (
    originalDetections ? `${originalDetections.length} detections on the original cutout.`
      : "Detector markers appear after measurement."
  );
  byId("batch-detail").textContent = detail;
  byId("batch-note").hidden = !isBatch;
  canvas.setAttribute("aria-label", isBatch ? `Synthetic holdout image. ${detail}` : "Verified original FITS cutout");
}

function describeBatch(frame, batches, index) {
  const recovered = batches.reduce((sum, batch) => sum + batch.matches.length, 0);
  const injected = batches.reduce((sum, batch) => sum + batch.injected.length, 0);
  return `Batch ${index + 1} of ${batches.length} · ${frame.matches.length}/${frame.injected.length} recovered · ${recovered}/${injected} across all batches`;
}

export function resetViewer() {
  original = undefined;
  originalDetections = undefined;
  holdouts = undefined;
  displayed = undefined;
  byId("frame").value = "original";
  byId("zoom").value = "fit";
  byId("viewport").dataset.zoom = "fit";
  byId("viewport").scrollTo(0, 0);
  for (const option of byId("frame").options) option.disabled = option.value !== "original";
  for (const id of ["viewer-tools", "cutout", "layers", "batch-detail", "batch-note"]) byId(id).hidden = true;
  byId("image-empty").hidden = false;
  byId("image-size").textContent = "FITS";
  byId("image-detail").removeAttribute("title");
  byId("image-detail").textContent = "Percentile stretch for display. Measurements use original pixel values.";
}

export function showCutout(image) {
  original = image;
  byId("image-empty").hidden = true;
  for (const id of ["viewer-tools", "cutout", "layers", "batch-detail"]) byId(id).hidden = false;
  byId("image-size").textContent = `${image.field} / ${image.case_id} · ${image.width} × ${image.height} px`;
  byId("image-detail").title = `FITS SHA-256 ${image.sha256}`;
  byId("image-detail").textContent = `1st–99.5th percentile stretch per frame · FITS SHA-256 ${image.sha256.slice(0, 12)}…`;
  drawFrame();
}

export function showDetections(detections) {
  originalDetections = detections;
  drawFrame();
}

export function setHoldoutViews(views, probeNames) {
  if (!views.before) return;
  holdouts = views;
  byId("probe").replaceChildren(...Object.keys(views.after_by_probe).map((probe) => new Option(probeNames[probe], probe)));
  byId("batch").replaceChildren(...views.before.map((_, index) => new Option(String(index + 1), String(index))));
  for (const option of byId("frame").options) option.disabled = false;
}

export function inspectProbe(probe) {
  byId("frame").value = "after";
  byId("probe").value = probe;
  byId("batch").value = "0";
  drawFrame();
  byId("frame").focus();
}

for (const id of ["frame", "probe", "batch", "layer-detections", "layer-injected", "layer-recovered"]) {
  byId(id).addEventListener("change", drawFrame);
}
byId("zoom").addEventListener("change", () => { byId("viewport").dataset.zoom = byId("zoom").value; });

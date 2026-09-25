"""
uvicorn app:app --reload
"""
import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from model.segformer_model import GlaucomaSegFormer
from predict import compute_cdr, create_overlay, glaucoma_risk
from utils import mask_to_rgb, load_checkpoint


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pth"
IMG_SIZE = int(os.getenv("IMG_SIZE", "512"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))


app = FastAPI(
    title="Fundus Glaucoma Analysis",
    description="Upload fundus images and receive optic disc/cup segmentation, CDR values, and risk output.",
    version="1.0.0",
)


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fundus Glaucoma Analysis</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #111827;
      --muted: #64748b;
      --line: #dbe4ee;
      --panel: #ffffff;
      --surface: #f5f7fb;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --blue: #2563eb;
      --disc: #16a34a;
      --cup: #dc2626;
      --warning: #b45309;
      --danger: #b91c1c;
      --shadow: 0 18px 50px rgba(15, 23, 42, 0.10);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        linear-gradient(135deg, rgba(245, 247, 251, 0.98), rgba(236, 244, 241, 0.92)),
        radial-gradient(circle at 88% 8%, rgba(37, 99, 235, 0.12), transparent 28%),
        radial-gradient(circle at 8% 90%, rgba(15, 118, 110, 0.14), transparent 30%);
      color: var(--ink);
    }

    main {
      width: min(1240px, calc(100% - 32px));
      margin: 0 auto;
      padding: 26px 0 34px;
    }

    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: end;
      gap: 18px;
      padding-bottom: 18px;
    }

    .eyebrow {
      margin: 0 0 8px;
      color: var(--accent-dark);
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.9rem);
      line-height: 1;
      letter-spacing: 0;
      max-width: 760px;
    }

    .subhead {
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.6;
      max-width: 760px;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 38px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
      padding: 0 14px;
      border-radius: 999px;
      color: var(--accent-dark);
      font-size: 0.9rem;
      font-weight: 800;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    }

    .status-dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: #22c55e;
      box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.16);
    }

    .workspace {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }

    .panel,
    .result-panel {
      background: rgba(255, 255, 255, 0.95);
      border: 1px solid rgba(219, 228, 238, 0.95);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .panel {
      position: sticky;
      top: 18px;
      padding: 18px;
    }

    .panel-title {
      margin: 0 0 12px;
      font-size: 1rem;
    }

    .dropzone {
      display: grid;
      place-items: center;
      min-height: 260px;
      border: 1px dashed #8aa8b4;
      border-radius: 8px;
      background: linear-gradient(180deg, #f8fbff, #eef7f4);
      text-align: center;
      padding: 18px;
      cursor: pointer;
      transition: border-color 0.18s ease, background 0.18s ease, transform 0.18s ease;
    }

    .dropzone:hover,
    .dropzone.dragover {
      border-color: var(--accent);
      background: #eaf7f4;
      transform: translateY(-1px);
    }

    .drop-icon {
      width: 56px;
      height: 56px;
      margin: 0 auto 14px;
      border-radius: 999px;
      display: grid;
      place-items: center;
      background: #ccfbf1;
      color: var(--accent-dark);
    }

    .drop-icon svg {
      width: 27px;
      height: 27px;
      stroke-width: 2.4;
    }

    .drop-title {
      margin: 0;
      font-size: 1.05rem;
      font-weight: 850;
    }

    .drop-copy {
      margin: 8px auto 0;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.45;
      max-width: 230px;
    }

    .preview {
      display: none;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #0f172a;
      border-radius: 7px;
    }

    .dropzone.has-preview {
      padding: 10px;
      background: #0f172a;
      border-style: solid;
    }

    .dropzone.has-preview .drop-content { display: none; }
    .dropzone.has-preview .preview { display: block; }

    input[type="file"] { display: none; }

    button {
      width: 100%;
      min-height: 48px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 850;
      cursor: pointer;
      margin-top: 14px;
      transition: background 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
      box-shadow: 0 12px 22px rgba(15, 118, 110, 0.18);
    }

    button:hover {
      background: var(--accent-dark);
      transform: translateY(-1px);
    }

    button:disabled {
      opacity: 0.58;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }

    .file-name {
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }

    .error {
      margin-top: 10px;
      color: var(--danger);
      font-size: 0.92rem;
      font-weight: 750;
      min-height: 22px;
    }

    .legend {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }

    .legend-row {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 0.78rem;
      line-height: 1.2;
    }

    .swatch {
      width: 100%;
      height: 8px;
      border-radius: 999px;
    }

    .result-panel {
      min-height: 590px;
      overflow: hidden;
    }

    .empty-state {
      min-height: 590px;
      display: grid;
      place-items: center;
      padding: 28px;
      text-align: center;
      color: var(--muted);
      background:
        linear-gradient(rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.92)),
        repeating-linear-gradient(45deg, #f3f6fa 0, #f3f6fa 10px, #ffffff 10px, #ffffff 20px);
    }

    .empty-state strong {
      display: block;
      color: var(--ink);
      font-size: 1.15rem;
      margin-bottom: 7px;
    }

    .results-header {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      border-bottom: 1px solid var(--line);
      background: #fbfdff;
    }

    .metric {
      padding: 18px;
      border-right: 1px solid var(--line);
    }

    .metric:last-child { border-right: 0; }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 0.75rem;
      font-weight: 850;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .metric strong {
      display: block;
      margin-top: 8px;
      font-size: 1.55rem;
      line-height: 1.12;
      overflow-wrap: anywhere;
    }

    .risk-low { color: var(--disc); }
    .risk-moderate { color: var(--warning); }
    .risk-high { color: var(--danger); }

    .image-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 1px;
      background: var(--line);
    }

    .image-tile {
      background: white;
      padding: 14px;
    }

    .image-tile h2 {
      margin: 0 0 10px;
      color: #334155;
      font-size: 0.92rem;
      font-weight: 850;
    }

    .image-tile img {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #0f172a;
      border-radius: 7px;
    }

    .note {
      padding: 15px 18px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.5;
      border-top: 1px solid var(--line);
      background: #fbfdff;
    }

    .loading {
      min-height: 590px;
      display: grid;
      place-items: center;
      color: var(--muted);
      text-align: center;
    }

    .spinner {
      width: 42px;
      height: 42px;
      margin: 0 auto 14px;
      border: 4px solid #dbeafe;
      border-top-color: var(--blue);
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @media (max-width: 940px) {
      header {
        grid-template-columns: 1fr;
        align-items: start;
      }

      .workspace {
        grid-template-columns: 1fr;
      }

      .panel {
        position: static;
      }
    }

    @media (max-width: 720px) {
      main {
        width: min(100% - 22px, 1240px);
        padding-top: 18px;
      }

      .results-header,
      .image-grid {
        grid-template-columns: 1fr;
      }

      .metric {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }

      .metric:last-child {
        border-bottom: 0;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <p class="eyebrow">AI retinal screening</p>
        <h1>Fundus Glaucoma Analysis</h1>
        <p class="subhead">Upload a retinal fundus image to view optic disc and cup segmentation, CDR measurements, and the model risk output.</p>
      </div>
      <div class="status-pill"><span class="status-dot"></span>SegFormer Ready</div>
    </header>

    <section class="workspace">
      <aside class="panel">
        <h2 class="panel-title">Image Upload</h2>
        <label class="dropzone" id="dropzone" for="imageInput">
          <img class="preview" id="previewImage" alt="Selected fundus preview" />
          <div class="drop-content">
            <div class="drop-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M12 5v14"></path>
                <path d="M5 12h14"></path>
              </svg>
            </div>
            <p class="drop-title">Choose or drop fundus image</p>
            <p class="drop-copy">Supports JPG, PNG, BMP, or TIFF up to 15 MB.</p>
          </div>
        </label>
        <input id="imageInput" type="file" accept="image/*" />
        <button id="analyzeButton" disabled>Analyze Image</button>
        <div class="file-name" id="fileName">No file selected</div>
        <div class="error" id="errorBox"></div>

        <div class="legend" aria-label="Segmentation legend">
          <div class="legend-row"><span class="swatch" style="background:#16a34a"></span>Optic disc</div>
          <div class="legend-row"><span class="swatch" style="background:#dc2626"></span>Optic cup</div>
          <div class="legend-row"><span class="swatch" style="background:#020617"></span>Background</div>
        </div>
      </aside>

      <section class="result-panel" id="resultPanel">
        <div class="empty-state">
          <div>
            <strong>Awaiting image analysis</strong>
            <p>Results will include the original image, predicted mask, overlay, vCDR, aCDR, and risk classification.</p>
          </div>
        </div>
      </section>
    </section>
  </main>

  <script>
    const input = document.getElementById("imageInput");
    const dropzone = document.getElementById("dropzone");
    const button = document.getElementById("analyzeButton");
    const fileName = document.getElementById("fileName");
    const errorBox = document.getElementById("errorBox");
    const resultPanel = document.getElementById("resultPanel");
    const previewImage = document.getElementById("previewImage");
    let selectedFile = null;
    let previewUrl = null;

    function setFile(file) {
      selectedFile = file || null;
      button.disabled = !selectedFile;
      errorBox.textContent = "";
      fileName.textContent = selectedFile ? selectedFile.name : "No file selected";

      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = selectedFile ? URL.createObjectURL(selectedFile) : null;
      previewImage.src = previewUrl || "";
      dropzone.classList.toggle("has-preview", Boolean(selectedFile));
    }

    input.addEventListener("change", () => setFile(input.files[0]));

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (file) setFile(file);
    });

    function riskClass(risk) {
      const value = risk.toLowerCase();
      if (value.includes("high")) return "risk-high";
      if (value.includes("moderate")) return "risk-moderate";
      return "risk-low";
    }

    button.addEventListener("click", async () => {
      if (!selectedFile) return;
      button.disabled = true;
      button.textContent = "Analyzing...";
      errorBox.textContent = "";
      resultPanel.innerHTML = `
        <div class="loading">
          <div>
            <div class="spinner"></div>
            <strong>Running segmentation</strong>
            <p>Processing the fundus image and calculating CDR metrics.</p>
          </div>
        </div>
      `;

      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Prediction failed");

        resultPanel.innerHTML = `
          <div class="results-header">
            <div class="metric"><span>Vertical CDR</span><strong>${data.vcdr.toFixed(3)}</strong></div>
            <div class="metric"><span>Area CDR</span><strong>${data.acdr.toFixed(3)}</strong></div>
            <div class="metric"><span>Risk Output</span><strong class="${riskClass(data.risk)}">${data.risk}</strong></div>
          </div>
          <div class="image-grid">
            <div class="image-tile">
              <h2>Original</h2>
              <img src="${data.images.original}" alt="Uploaded fundus image" />
            </div>
            <div class="image-tile">
              <h2>Segmentation Mask</h2>
              <img src="${data.images.mask}" alt="Predicted segmentation mask" />
            </div>
            <div class="image-tile">
              <h2>Clinical Overlay</h2>
              <img src="${data.images.overlay}" alt="Segmentation overlay" />
            </div>
          </div>
          <div class="note">This output is model-assisted screening information and should be reviewed by a qualified clinician.</div>
        `;
      } catch (error) {
        errorBox.textContent = error.message;
        resultPanel.innerHTML = `
          <div class="empty-state">
            <div>
              <strong>Analysis could not complete</strong>
              <p>${error.message}</p>
            </div>
          </div>
        `;
      } finally {
        button.disabled = false;
        button.textContent = "Analyze Image";
      }
    });
  </script>
</body>
</html>
"""


def image_to_data_url(image: np.ndarray) -> str:
    pil_image = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def preprocess_image(image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
    transform = A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    original = np.array(image.convert("RGB"))
    tensor = transform(image=original)["image"].unsqueeze(0)
    return tensor, original


def clean_risk_label(risk: str) -> str:
    if "High Risk" in risk:
        return "High Risk - Suspect Glaucoma"
    return risk


@lru_cache(maxsize=1)
def get_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GlaucomaSegFormer().to(device)
    load_checkpoint(model, None, str(CHECKPOINT_PATH), device)
    model.eval()
    return model, device


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/health")
def health():
    checkpoint_exists = CHECKPOINT_PATH.exists()
    return {
        "status": "ok" if checkpoint_exists else "missing_checkpoint",
        "checkpoint": str(CHECKPOINT_PATH),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image is larger than {MAX_UPLOAD_MB} MB.")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file could not be opened as an image.") from exc

    try:
        model, device = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    tensor, original = preprocess_image(image)
    tensor = tensor.to(device)

    use_amp = device == "cuda"
    with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=use_amp):
        logits = model(tensor)

    pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    vcdr, acdr = compute_cdr(pred_mask)
    risk, _ = glaucoma_risk(vcdr)

    resized_original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    mask_rgb = mask_to_rgb(pred_mask)
    overlay = create_overlay(original, pred_mask)

    return {
        "filename": file.filename,
        "vcdr": vcdr,
        "acdr": acdr,
        "risk": clean_risk_label(risk),
        "images": {
            "original": image_to_data_url(resized_original),
            "mask": image_to_data_url(mask_rgb),
            "overlay": image_to_data_url(overlay),
        },
    }


if __name__ == "__main__":
    import socket

    import uvicorn

    def find_free_port(start_port: int = 8000, max_attempts: int = 20) -> int:
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"No free port found starting from {start_port}")

    port = int(os.getenv("PORT", str(find_free_port())))
    print(f"Starting FastAPI app on http://0.0.0.0:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

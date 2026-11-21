"""
research_v3/phase8_ground_truth/annotation_tool/app.py

Lightweight, local-only, privacy-preserving annotation tool for Phase 8.
Supports:
1. Mode A (Annotator A Blind Transcription & Visual Bounding Box Selection)
2. Mode B (Annotator B Blind Transcription & Visual Bounding Box Selection)
3. Mode C (Adjudication of Discrepancies, Bounding Box Consensus & Rationale)

Runs strictly on localhost (127.0.0.1:5005) with zero network calls.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify, render_template_string, send_file
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_ROOT = Path(os.environ.get("AAE_DATA_ROOT", r"F:\Automatic answer evaluation dataset"))

# Local private annotation storage
STORE_DIR = REPO_ROOT / "research_v3" / "phase8_ground_truth" / "annotations_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)
MODE_A_FILE = STORE_DIR / "annotations_mode_a.json"
MODE_B_FILE = STORE_DIR / "annotations_mode_b.json"
ADJ_FILE = STORE_DIR / "adjudications.json"
AUDIT_LOG_FILE = STORE_DIR / "audit_log.jsonl"

app = Flask(__name__)

# Initialize storage files if missing
for f in [MODE_A_FILE, MODE_B_FILE, ADJ_FILE]:
    if not f.exists():
        f.write_text("{}", encoding="utf-8")


def load_store(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_store(path: Path, data: Dict[str, Any]):
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def log_audit(action: str, record_id: str, reviewer_id: str, details: Dict[str, Any]):
    entry = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "record_id": record_id,
        "reviewer_id": reviewer_id,
        "details": details
    }
    with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Load booklet and page manifests
def get_booklets_info() -> Dict[str, Any]:
    pages_csv = REPO_ROOT / "research_v2" / "annotations" / "pages.csv"
    if not pages_csv.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(pages_csv)
    booklets = {}
    for sid, group in df.groupby("student_id"):
        booklets[sid] = {
            "student_id": sid,
            "pages": group.sort_values("page_index")[["page_id", "page_index", "image_relative_path", "width_px", "height_px"]].to_dict(orient="records")
        }
    return booklets


BOOKLETS = get_booklets_info()
SUBQUESTIONS = ["1a", "1b", "1c", "1d", "2a", "2b", "2c", "3a", "3b", "3c", "4"]


# ---------------------------------------------------------
# Web Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_SHELL)


@app.route("/api/booklets")
def api_booklets():
    return jsonify({
        "booklets": sorted(list(BOOKLETS.keys()), key=lambda s: int(s.replace("s", ""))),
        "subquestions": SUBQUESTIONS
    })


@app.route("/api/page_image/<student_id>/<page_index>")
def api_page_image(student_id, page_index):
    if student_id not in BOOKLETS:
        return "Booklet not found", 404
    pages = BOOKLETS[student_id]["pages"]
    idx = int(page_index) - 1
    if idx < 0 or idx >= len(pages):
        return "Page index out of bounds", 404
    rel_path = pages[idx]["image_relative_path"]
    full_path = DATA_ROOT / rel_path
    if not full_path.exists():
        return f"File not found: {full_path}", 404
    return send_file(full_path, mimetype="image/jpeg")


@app.route("/api/get_record", methods=["GET"])
def api_get_record():
    mode = request.args.get("mode", "A")
    student_id = request.args.get("student_id", "")
    subq = request.args.get("subquestion_id", "")
    key = f"{student_id}_{subq}"

    store_a = load_store(MODE_A_FILE)
    store_b = load_store(MODE_B_FILE)
    store_adj = load_store(ADJ_FILE)

    if mode == "A":
        rec = store_a.get(key, {})
        # Strictly blind: hide B and historical
        return jsonify({"record": rec, "mode": "A"})
    elif mode == "B":
        rec = store_b.get(key, {})
        # Strictly blind: hide A and historical
        return jsonify({"record": rec, "mode": "B"})
    elif mode == "ADJ":
        rec_a = store_a.get(key, {})
        rec_b = store_b.get(key, {})
        rec_adj = store_adj.get(key, {})
        # Load historical reference as unverified candidate
        hist_df_path = REPO_ROOT / "research_v2" / "annotations" / "transcripts.csv"
        hist_cand = ""
        if hist_df_path.exists():
            import pandas as pd
            hdf = pd.read_csv(hist_df_path)
            match = hdf[(hdf["student_id"] == student_id) & (hdf["subquestion_id"] == subq)]
            if len(match) > 0:
                hist_cand = str(match["adjudicated_transcript"].iloc[0])

        return jsonify({
            "record_a": rec_a,
            "record_b": rec_b,
            "record_adj": rec_adj,
            "historical_candidate": hist_cand,
            "mode": "ADJ"
        })
    return "Invalid mode", 400


@app.route("/api/save_record", methods=["POST"])
def api_save_record():
    payload = request.json or {}
    mode = payload.get("mode", "A")
    student_id = payload.get("student_id", "")
    subq = payload.get("subquestion_id", "")
    reviewer_id = payload.get("reviewer_id", "UNASSIGNED")
    key = f"{student_id}_{subq}"

    if not student_id or not subq:
        return jsonify({"error": "Missing student_id or subquestion_id"}), 400

    timestamp = datetime.now(timezone.utc).isoformat()

    if mode == "A":
        store = load_store(MODE_A_FILE)
        payload["timestamp_iso"] = timestamp
        payload["annotator_mode"] = "MODE_A_BLIND"
        store[key] = payload
        save_store(MODE_A_FILE, store)
        log_audit("SAVE_MODE_A", key, reviewer_id, {"status": payload.get("answer_status")})
        return jsonify({"status": "saved", "key": key, "mode": "A"})

    elif mode == "B":
        store = load_store(MODE_B_FILE)
        payload["timestamp_iso"] = timestamp
        payload["annotator_mode"] = "MODE_B_BLIND"
        store[key] = payload
        save_store(MODE_B_FILE, store)
        log_audit("SAVE_MODE_B", key, reviewer_id, {"status": payload.get("answer_status")})
        return jsonify({"status": "saved", "key": key, "mode": "B"})

    elif mode == "ADJ":
        store = load_store(ADJ_FILE)
        payload["timestamp_iso"] = timestamp
        store[key] = payload
        save_store(ADJ_FILE, store)
        log_audit("SAVE_ADJUDICATION", key, reviewer_id, {"final_status": payload.get("final_adjudicated_status")})
        return jsonify({"status": "saved", "key": key, "mode": "ADJ"})

    return jsonify({"error": "Invalid mode"}), 400


HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Phase 8 Local Offline Human Annotation & Adjudication Interface</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 0; background: #f4f6f9; color: #212529; }
    header { background: #1a252f; color: #fff; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; }
    header h1 { font-size: 1.1rem; margin: 0; font-weight: 600; }
    .badge { background: #e74c3c; color: #fff; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
    .badge-b { background: #3498db; }
    .badge-adj { background: #2ecc71; }
    .container { display: flex; height: calc(100vh - 56px); }
    .sidebar { width: 320px; background: #fff; border-right: 1px solid #dee2e6; padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
    .workspace { flex: 1; display: flex; flex-direction: column; background: #2c3e50; position: relative; }
    .toolbar { background: #34495e; color: #fff; padding: 8px 16px; display: flex; gap: 16px; align-items: center; font-size: 0.9rem; }
    .canvas-container { flex: 1; overflow: auto; position: relative; display: flex; justify-content: center; align-items: center; }
    canvas { border: 1px solid #7f8c8d; box-shadow: 0 4px 12px rgba(0,0,0,0.5); cursor: crosshair; }
    .form-group { display: flex; flex-direction: column; gap: 4px; }
    label { font-size: 0.82rem; font-weight: 600; color: #495057; }
    select, input, textarea { padding: 8px; border: 1px solid #ced4da; border-radius: 4px; font-size: 0.85rem; font-family: inherit; }
    textarea { height: 110px; resize: vertical; }
    .checkbox-group { display: flex; flex-direction: column; gap: 6px; font-size: 0.82rem; }
    .checkbox-group label { font-weight: normal; cursor: pointer; }
    button { padding: 9px 16px; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 0.85rem; }
    .btn-primary { background: #2980b9; color: #fff; }
    .btn-primary:hover { background: #2471a3; }
    .btn-success { background: #27ae60; color: #fff; }
    .btn-success:hover { background: #229954; }
    .btn-secondary { background: #7f8c8d; color: #fff; }
    .mode-selector { display: flex; gap: 8px; margin-bottom: 8px; }
    .mode-btn { flex: 1; padding: 6px; font-size: 0.78rem; border: 1px solid #ccc; background: #e9ecef; }
    .mode-btn.active { background: #1a252f; color: #fff; border-color: #1a252f; }
    .info-box { background: #e8f4f8; border-left: 4px solid #2980b9; padding: 8px 12px; font-size: 0.78rem; line-height: 1.4; }
    .diff-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 8px 12px; font-size: 0.78rem; }
  </style>
</head>
<body>
  <header>
    <h1>Phase 8: Offline Human Ground-Truth Annotation & Verification</h1>
    <div>
      <span id="mode-badge" class="badge">MODE A: ANNOTATOR A (BLIND)</span>
    </div>
  </header>

  <div class="container">
    <div class="sidebar">
      <div class="mode-selector">
        <button class="mode-btn active" onclick="switchMode('A')">Mode A (Rater 1)</button>
        <button class="mode-btn" onclick="switchMode('B')">Mode B (Rater 2)</button>
        <button class="mode-btn" onclick="switchMode('ADJ')">Adjudication</button>
      </div>

      <div class="form-group">
        <label>Reviewer Pseudonym ID</label>
        <input type="text" id="reviewer-id" value="REV_01">
      </div>

      <div class="form-group">
        <label>Select Student Booklet</label>
        <select id="select-booklet" onchange="onBookletChange()"></select>
      </div>

      <div class="form-group">
        <label>Select Subquestion</label>
        <select id="select-subq" onchange="loadCurrentRecord()"></select>
      </div>

      <div class="form-group">
        <label>Page Navigation</label>
        <div style="display:flex; gap:8px;">
          <button class="btn-secondary" onclick="prevPage()">&larr; Prev</button>
          <select id="select-page" style="flex:1;" onchange="renderCurrentPage()"></select>
          <button class="btn-secondary" onclick="nextPage()">Next &rarr;</button>
        </div>
      </div>

      <div class="form-group">
        <label>Answer Status</label>
        <select id="answer-status">
          <option value="written">written (Handwritten answer present)</option>
          <option value="omitted_confirmed">omitted_confirmed (Verified blank slot)</option>
          <option value="illegible">illegible (Unreadable ink strokes)</option>
          <option value="partially_legible">partially_legible (Bracket uncertain tokens)</option>
          <option value="unresolved">unresolved (Contested / Needs instructor)</option>
        </select>
      </div>

      <div class="form-group">
        <label>Bounding Box Coordinates [x0, y0, x1, y1]</label>
        <div style="display:flex; gap:4px;">
          <input type="number" id="box-x0" placeholder="x0" style="width:25%;">
          <input type="number" id="box-y0" placeholder="y0" style="width:25%;">
          <input type="number" id="box-x1" placeholder="x1" style="width:25%;">
          <input type="number" id="box-y1" placeholder="y1" style="width:25%;">
        </div>
        <small style="color:#6c757d; font-size:0.75rem;">Click and drag on the image to draw bounding box.</small>
      </div>

      <div class="form-group">
        <label>Verbatim Transcription (Faithful / No Corrections)</label>
        <textarea id="raw-transcript" placeholder="Transcribe student handwriting verbatim. Preserve all wrong math, misspelled words, and syntax..."></textarea>
      </div>

      <div class="checkbox-group">
        <label><input type="checkbox" id="chk-math"> Contains Math / Equations / Vectors</label>
        <label><input type="checkbox" id="chk-diag"> Contains Architecture Diagram</label>
        <label><input type="checkbox" id="chk-cross"> Contains Struck-Out / Crossed Text</label>
        <label><input type="checkbox" id="chk-over"> Contains Overwritten Characters</label>
      </div>

      <div id="adj-diff-panel" style="display:none;" class="diff-box">
        <strong>Pre-Adjudication Discrepancy View:</strong>
        <div id="diff-content" style="margin-top:6px; font-family:monospace;"></div>
      </div>

      <button class="btn-success" onclick="saveRecord()">Save Verification & Submit</button>
      <div id="save-status" style="font-size:0.8rem; text-align:center; color:#27ae60;"></div>
    </div>

    <div class="workspace">
      <div class="toolbar">
        <span>Zoom: <button class="btn-secondary" onclick="zoomIn()">+</button> <button class="btn-secondary" onclick="zoomOut()">-</button> <button class="btn-secondary" onclick="zoomReset()">100%</button></span>
        <span id="img-dim-info">Resolution: 3072 &times; 4096 px</span>
        <span style="margin-left:auto; font-size:0.8rem; color:#bdc3c7;">AAE_DATA_ROOT Local Offline Sandbox</span>
      </div>
      <div class="canvas-container">
        <canvas id="page-canvas"></canvas>
      </div>
    </div>
  </div>

  <script>
    let currentMode = "A";
    let bookletsData = {};
    let currentStudent = "";
    let currentSubq = "";
    let currentPageIndex = 1;
    let currentImage = new Image();
    let zoomLevel = 0.25;

    let isDrawing = false;
    let startX = 0, startY = 0;
    let box = { x0: 0, y0: 0, x1: 0, y1: 0 };

    const canvas = document.getElementById("page-canvas");
    const ctx = canvas.getContext("2d");

    async function init() {
      const res = await fetch("/api/booklets");
      bookletsData = await res.json();
      
      const bSelect = document.getElementById("select-booklet");
      bSelect.innerHTML = "";
      bookletsData.booklets.forEach(b => {
        const opt = document.createElement("option");
        opt.value = b;
        opt.text = b;
        bSelect.appendChild(opt);
      });

      const sSelect = document.getElementById("select-subq");
      sSelect.innerHTML = "";
      bookletsData.subquestions.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.text = s;
        sSelect.appendChild(opt);
      });

      currentStudent = bookletsData.booklets[0];
      currentSubq = bookletsData.subquestions[0];
      onBookletChange();
    }

    function switchMode(m) {
      currentMode = m;
      document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
      event.target.classList.add("active");

      const badge = document.getElementById("mode-badge");
      if (m === "A") {
        badge.className = "badge";
        badge.innerText = "MODE A: ANNOTATOR A (BLIND)";
        document.getElementById("reviewer-id").value = "REV_01";
        document.getElementById("adj-diff-panel").style.display = "none";
      } else if (m === "B") {
        badge.className = "badge badge-b";
        badge.innerText = "MODE B: ANNOTATOR B (BLIND)";
        document.getElementById("reviewer-id").value = "REV_02";
        document.getElementById("adj-diff-panel").style.display = "none";
      } else {
        badge.className = "badge badge-adj";
        badge.innerText = "MODE C: ADJUDICATION CONSENSUS";
        document.getElementById("reviewer-id").value = "ADJ_LEAD";
        document.getElementById("adj-diff-panel").style.display = "block";
      }
      loadCurrentRecord();
    }

    function onBookletChange() {
      currentStudent = document.getElementById("select-booklet").value;
      const pSelect = document.getElementById("select-page");
      pSelect.innerHTML = "";
      // Assume 5-7 pages
      for (let i = 1; i <= 7; i++) {
        const opt = document.createElement("option");
        opt.value = i;
        opt.text = "Page " + i;
        pSelect.appendChild(opt);
      }
      currentPageIndex = 1;
      renderCurrentPage();
      loadCurrentRecord();
    }

    function renderCurrentPage() {
      currentPageIndex = document.getElementById("select-page").value;
      currentImage.src = `/api/page_image/${currentStudent}/${currentPageIndex}`;
      currentImage.onload = () => {
        document.getElementById("img-dim-info").innerText = `Resolution: ${currentImage.naturalWidth} x ${currentImage.naturalHeight} px`;
        redraw();
      };
    }

    function redraw() {
      canvas.width = currentImage.naturalWidth * zoomLevel;
      canvas.height = currentImage.naturalHeight * zoomLevel;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

      // Draw active box
      if (box.x1 > box.x0 && box.y1 > box.y0) {
        ctx.strokeStyle = "#e74c3c";
        ctx.lineWidth = 3;
        ctx.strokeRect(
          box.x0 * zoomLevel,
          box.y0 * zoomLevel,
          (box.x1 - box.x0) * zoomLevel,
          (box.y1 - box.y0) * zoomLevel
        );
        ctx.fillStyle = "rgba(231, 76, 60, 0.15)";
        ctx.fillRect(
          box.x0 * zoomLevel,
          box.y0 * zoomLevel,
          (box.x1 - box.x0) * zoomLevel,
          (box.y1 - box.y0) * zoomLevel
        );
      }
    }

    canvas.addEventListener("mousedown", (e) => {
      isDrawing = true;
      const rect = canvas.getBoundingClientRect();
      startX = (e.clientX - rect.left) / zoomLevel;
      startY = (e.clientY - rect.top) / zoomLevel;
    });

    canvas.addEventListener("mousemove", (e) => {
      if (!isDrawing) return;
      const rect = canvas.getBoundingClientRect();
      const currX = (e.clientX - rect.left) / zoomLevel;
      const currY = (e.clientY - rect.top) / zoomLevel;
      box.x0 = Math.round(Math.min(startX, currX));
      box.y0 = Math.round(Math.min(startY, currY));
      box.x1 = Math.round(Math.max(startX, currX));
      box.y1 = Math.round(Math.max(startY, currY));
      updateBoxInputs();
      redraw();
    });

    canvas.addEventListener("mouseup", () => { isDrawing = false; });

    function updateBoxInputs() {
      document.getElementById("box-x0").value = box.x0;
      document.getElementById("box-y0").value = box.y0;
      document.getElementById("box-x1").value = box.x1;
      document.getElementById("box-y1").value = box.y1;
    }

    function zoomIn() { zoomLevel = Math.min(1.5, zoomLevel + 0.1); redraw(); }
    function zoomOut() { zoomLevel = Math.max(0.1, zoomLevel - 0.1); redraw(); }
    function zoomReset() { zoomLevel = 0.25; redraw(); }
    function prevPage() { if (currentPageIndex > 1) { currentPageIndex--; document.getElementById("select-page").value = currentPageIndex; renderCurrentPage(); } }
    function nextPage() { currentPageIndex++; document.getElementById("select-page").value = currentPageIndex; renderCurrentPage(); }

    async function loadCurrentRecord() {
      currentStudent = document.getElementById("select-booklet").value;
      currentSubq = document.getElementById("select-subq").value;
      const res = await fetch(`/api/get_record?mode=${currentMode}&student_id=${currentStudent}&subquestion_id=${currentSubq}`);
      const data = await res.json();

      if (currentMode === "A" || currentMode === "B") {
        const rec = data.record || {};
        document.getElementById("raw-transcript").value = rec.raw_transcript || "";
        document.getElementById("answer-status").value = rec.answer_status || "written";
        document.getElementById("chk-math").checked = !!rec.contains_math;
        document.getElementById("chk-diag").checked = !!rec.contains_diagram;
        document.getElementById("chk-cross").checked = !!rec.contains_crossed_out_text;
        document.getElementById("chk-over").checked = !!rec.contains_overwrite;
        if (rec.coordinates_pixel) {
          box = {
            x0: rec.coordinates_pixel.x_min,
            y0: rec.coordinates_pixel.y_min,
            x1: rec.coordinates_pixel.x_max,
            y1: rec.coordinates_pixel.y_max
          };
          updateBoxInputs();
          redraw();
        }
      } else {
        // Mode ADJ
        const ra = data.record_a || {};
        const rb = data.record_b || {};
        const radj = data.record_adj || {};
        document.getElementById("diff-content").innerHTML = `
          <div><strong>Annotator A:</strong> [${ra.answer_status || 'missing'}] ${ra.raw_transcript || '(none)'}</div>
          <div style="margin-top:4px;"><strong>Annotator B:</strong> [${rb.answer_status || 'missing'}] ${rb.raw_transcript || '(none)'}</div>
          <div style="margin-top:4px; color:#7f8c8d;"><strong>Hist. Candidate:</strong> ${data.historical_candidate || '(none)'}</div>
        `;
        document.getElementById("raw-transcript").value = radj.final_adjudicated_transcript || ra.raw_transcript || rb.raw_transcript || "";
        document.getElementById("answer-status").value = radj.final_adjudicated_status || "written";
      }
    }

    async function saveRecord() {
      const payload = {
        mode: currentMode,
        student_id: currentStudent,
        subquestion_id: currentSubq,
        reviewer_id: document.getElementById("reviewer-id").value,
        answer_status: document.getElementById("answer-status").value,
        raw_transcript: document.getElementById("raw-transcript").value,
        final_adjudicated_transcript: document.getElementById("raw-transcript").value,
        final_adjudicated_status: document.getElementById("answer-status").value,
        contains_math: document.getElementById("chk-math").checked,
        contains_diagram: document.getElementById("chk-diag").checked,
        contains_crossed_out_text: document.getElementById("chk-cross").checked,
        contains_overwrite: document.getElementById("chk-over").checked,
        page_index: parseInt(currentPageIndex),
        coordinates_pixel: {
          x_min: parseInt(document.getElementById("box-x0").value) || 0,
          y_min: parseInt(document.getElementById("box-y0").value) || 0,
          x_max: parseInt(document.getElementById("box-x1").value) || 100,
          y_max: parseInt(document.getElementById("box-y1").value) || 100
        }
      };

      const res = await fetch("/api/save_record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const resData = await res.json();
      const statusDiv = document.getElementById("save-status");
      statusDiv.innerText = "Saved successfully at " + new Date().toLocaleTimeString();
      setTimeout(() => statusDiv.innerText = "", 3000);
    }

    window.onload = init;
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"Starting Phase 8 Local Offline Human Annotation Tool on http://127.0.0.1:5005")
    print(f"Reading private source images from: {DATA_ROOT}")
    print(f"Saving verified annotation submissions to: {STORE_DIR}")
    app.run(host="127.0.0.1", port=5005, debug=False)

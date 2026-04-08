import argparse
import html
import json
import os
from collections import defaultdict
from pathlib import Path

from fer_meetings.temporal import sample_frame_times
from fer_meetings.utils import ensure_parent, read_csv_rows, write_csv_rows
from fer_meetings.video import open_video, read_frame_at


def parse_args():
    parser = argparse.ArgumentParser(description="Build an annotation pack with thumbnails and local HTML preview.")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV.")
    parser.add_argument(
        "--predictions",
        default="",
        help="Optional predictions CSV used only for weak suggestions in a separate column.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for HTML, thumbnails and CSV review sheet.")
    parser.add_argument(
        "--labels-output",
        default="",
        help="Optional path for the annotation CSV. Defaults to <output-dir>/annotation_sheet.csv.",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=3,
        help="Number of frames used in each thumbnail strip.",
    )
    parser.add_argument(
        "--thumb-height",
        type=int,
        default=160,
        help="Per-frame height in pixels for the thumbnail strip.",
    )
    return parser.parse_args()


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_prediction_index(prediction_rows):
    grouped = defaultdict(list)
    for row in prediction_rows:
        confidence = max(
            parse_float(row.get("smoothed_negative_prob")),
            parse_float(row.get("smoothed_neutral_prob")),
            parse_float(row.get("smoothed_positive_prob")),
        )
        grouped[row["clip_id"]].append(
            {
                "model_name": row.get("model_name", ""),
                "smoothed_label": row.get("smoothed_label", ""),
                "vote_label": row.get("vote_label", ""),
                "confidence": confidence,
                "face_detected_ratio": row.get("face_detected_ratio", ""),
            }
        )

    for clip_id in grouped:
        grouped[clip_id] = sorted(
            grouped[clip_id],
            key=lambda item: (-item["confidence"], item["model_name"]),
        )
    return grouped


def summarize_predictions(predictions):
    if not predictions:
        return {
            "suggested_label": "",
            "suggested_confidence": "",
            "suggestion_summary": "",
            "face_detected_ratio": "",
        }

    best = predictions[0]
    summary = "; ".join(
        f"{item['model_name']}={item['smoothed_label']} ({item['confidence']:.3f})"
        for item in predictions
    )
    return {
        "suggested_label": best["smoothed_label"],
        "suggested_confidence": f"{best['confidence']:.3f}",
        "suggestion_summary": summary,
        "face_detected_ratio": best["face_detected_ratio"],
    }


def annotate_frame(frame_bgr, timestamp_s, label):
    import cv2

    overlay = frame_bgr.copy()
    text = f"{label} {timestamp_s:.2f}s"
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def make_thumbnail_strip(video_path, clip_start_s, clip_end_s, output_path, frames_per_clip, thumb_height):
    import cv2

    capture = open_video(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS for video: {video_path}")

    thumbs = []
    timestamps = sample_frame_times(float(clip_start_s), float(clip_end_s), frames_per_clip)
    for index, timestamp_s in enumerate(timestamps, start=1):
        frame = read_frame_at(capture, fps, timestamp_s)
        if frame is None:
            continue
        scale = thumb_height / max(frame.shape[0], 1)
        width = max(1, int(round(frame.shape[1] * scale)))
        frame = cv2.resize(frame, (width, thumb_height), interpolation=cv2.INTER_AREA)
        frame = annotate_frame(frame, timestamp_s, f"f{index}")
        thumbs.append(frame)

    capture.release()
    if not thumbs:
        raise RuntimeError(f"Could not extract thumbnail frames for: {video_path}")

    strip = cv2.hconcat(thumbs)
    ensure_parent(output_path)
    ok = cv2.imwrite(str(output_path), strip)
    if not ok:
        raise RuntimeError(f"Could not write thumbnail strip: {output_path}")


def relative_posix_path(path, start):
    return Path(os.path.relpath(path, start)).as_posix()


def build_annotation_rows(
    manifest_rows,
    prediction_index,
    existing_labels,
    output_dir,
    thumbnails_dir,
    frames_per_clip,
    thumb_height,
):
    rows = []
    for row in manifest_rows:
        suggestions = summarize_predictions(prediction_index.get(row["clip_id"], []))
        existing = existing_labels.get(row["clip_id"], {})
        thumbnail_path = thumbnails_dir / f"{row['clip_id']}.jpg"
        make_thumbnail_strip(
            video_path=row["video_path"],
            clip_start_s=row["clip_start_s"],
            clip_end_s=row["clip_end_s"],
            output_path=thumbnail_path,
            frames_per_clip=frames_per_clip,
            thumb_height=thumb_height,
        )

        rows.append(
            {
                "clip_id": row["clip_id"],
                "split": row["split"],
                "video_file": row["video_file"],
                "meeting_id": row["meeting_id"],
                "camera": row["camera"],
                "clip_start_s": row["clip_start_s"],
                "clip_end_s": row["clip_end_s"],
                "video_path": row["video_path"],
                "thumbnail_path": relative_posix_path(thumbnail_path, output_dir),
                "rater_1_label": existing.get("rater_1_label", ""),
                "rater_2_label": existing.get("rater_2_label", ""),
                "adjudicated_label": existing.get("adjudicated_label", ""),
                "gold_label": existing.get("gold_label", ""),
                "annotator": existing.get("annotator", ""),
                "adjudicator": existing.get("adjudicator", ""),
                "exclude_from_gold": existing.get("exclude_from_gold", ""),
                "agreement_status": existing.get("agreement_status", ""),
                "notes": existing.get("notes", ""),
                "suggested_label": suggestions["suggested_label"],
                "suggested_confidence": suggestions["suggested_confidence"],
                "face_detected_ratio": suggestions["face_detected_ratio"],
                "suggestion_summary": suggestions["suggestion_summary"],
            }
        )
    return rows


def render_html(rows, output_dir, output_path):
    render_rows = []
    for row in rows:
        render_row = dict(row)
        render_row["video_rel_path"] = relative_posix_path(row["video_path"], output_dir)
        render_rows.append(render_row)

    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FER Meetings Annotation Pack</title>
  <style>
    :root {
      --bg: #f7f3eb;
      --panel: #fffaf2;
      --line: #dacfbf;
      --text: #1f1c17;
      --muted: #6f665c;
      --neg: #b3382c;
      --neu: #6c7278;
      --pos: #237a42;
      --accent: #124d5b;
      --warn: #aa7a10;
    }
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: var(--text); background: linear-gradient(180deg, #f7f3eb 0%, #efe6d7 100%); }
    h1 { margin: 0 0 8px; }
    p { margin: 0 0 16px; max-width: 980px; line-height: 1.45; }
    code { background: #f1ebdf; padding: 1px 4px; border-radius: 4px; }
    .toolbar {
      position: sticky;
      top: 12px;
      z-index: 3;
      display: grid;
      grid-template-columns: 1.6fr repeat(3, minmax(140px, 0.6fr)) auto auto auto;
      gap: 10px;
      align-items: center;
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      backdrop-filter: blur(6px);
      margin-bottom: 16px;
      box-shadow: 0 12px 28px rgba(31, 28, 23, 0.08);
    }
    .toolbar label { display: grid; gap: 4px; font-size: 12px; color: var(--muted); }
    .toolbar input, .toolbar select, .toolbar button, .toolbar textarea {
      font: inherit;
    }
    .toolbar input, .toolbar select {
      width: 100%;
      border: 1px solid var(--line);
      background: white;
      border-radius: 10px;
      padding: 8px 10px;
      color: var(--text);
    }
    .toolbar-actions {
      display: flex;
      gap: 8px;
      justify-content: flex-end;
      align-items: end;
    }
    .toolbar button {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 10px;
      padding: 9px 12px;
      cursor: pointer;
      font-weight: 600;
      white-space: nowrap;
    }
    .toolbar button.secondary {
      background: white;
      color: var(--accent);
    }
    .status-bar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 13px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(18, 77, 91, 0.08);
      border: 1px solid rgba(18, 77, 91, 0.15);
    }
    .chip strong { color: var(--text); }
    table { border-collapse: collapse; width: 100%; font-size: 13px; background: rgba(255, 250, 242, 0.96); border: 1px solid var(--line); }
    th, td { border: 1px solid var(--line); padding: 8px; vertical-align: top; }
    th { background: #efe5d3; position: sticky; top: 106px; z-index: 2; }
    img { max-width: 480px; height: auto; display: block; border-radius: 8px; }
    video { width: 320px; max-height: 220px; display: block; border-radius: 8px; background: #000; }
    .clip-meta { min-width: 210px; }
    .clip-meta strong { display: block; margin-bottom: 4px; }
    .control-cell { min-width: 280px; }
    .current-label {
      display: inline-block;
      padding: 5px 9px;
      border-radius: 999px;
      font-weight: 700;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.04em;
      margin-bottom: 8px;
    }
    .label-empty { background: #f2ece2; color: var(--muted); }
    .label-negative { background: rgba(179, 56, 44, 0.12); color: var(--neg); }
    .label-neutral { background: rgba(108, 114, 120, 0.14); color: var(--neu); }
    .label-positive { background: rgba(35, 122, 66, 0.12); color: var(--pos); }
    .segmented {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    .label-btn {
      border: 1px solid var(--line);
      background: white;
      color: var(--text);
      border-radius: 999px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
    }
    .label-btn[data-label="negative"].active { background: rgba(179, 56, 44, 0.12); color: var(--neg); border-color: rgba(179, 56, 44, 0.35); }
    .label-btn[data-label="neutral"].active { background: rgba(108, 114, 120, 0.14); color: var(--neu); border-color: rgba(108, 114, 120, 0.35); }
    .label-btn[data-label="positive"].active { background: rgba(35, 122, 66, 0.12); color: var(--pos); border-color: rgba(35, 122, 66, 0.35); }
    .label-btn[data-label=""] { color: var(--warn); }
    .field-row { display: grid; gap: 6px; margin: 8px 0; }
    .field-row label { font-size: 12px; color: var(--muted); }
    .field-row input[type="text"], .field-row textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      background: white;
      color: var(--text);
    }
    .field-row textarea { min-height: 72px; resize: vertical; }
    .inline-note { color: var(--muted); font-size: 12px; }
    .row-hidden { display: none; }
    .dirty-row { outline: 2px solid rgba(18, 77, 91, 0.18); outline-offset: -2px; }
    .helper { font-size: 12px; color: var(--muted); }
    @media (max-width: 1100px) {
      .toolbar { grid-template-columns: 1fr 1fr; }
      th { top: 210px; }
    }
  </style>
</head>
<body>
  <h1>Annotation Pack</h1>
  <p>
    Use this page to review clips and update <code>gold_label</code> directly from the browser.
    The columns <code>suggested_label</code> and <code>suggestion_summary</code> are weak model hints and must not replace human judgment.
    Changes are auto-saved locally in the browser and can be exported again as <code>annotation_sheet.csv</code>.
    For publication-quality annotation, prefer <code>rater_1_label</code>, <code>rater_2_label</code> and <code>adjudicated_label</code>.
  </p>
  <div class="toolbar">
    <label>
      Search
      <input id="searchInput" type="text" placeholder="clip_id, meeting, camera">
    </label>
    <label>
      Split
      <select id="splitFilter">
        <option value="all">All</option>
      </select>
    </label>
    <label>
      Label status
      <select id="statusFilter">
        <option value="all">All</option>
        <option value="labeled">Labeled</option>
        <option value="unlabeled">Unlabeled</option>
      </select>
    </label>
    <label>
      Gold label
      <select id="labelFilter">
        <option value="all">All</option>
        <option value="negative">negative</option>
        <option value="neutral">neutral</option>
        <option value="positive">positive</option>
      </select>
    </label>
    <div class="toolbar-actions">
      <button id="saveFileBtn" type="button">Guardar CSV</button>
      <button id="downloadBtn" class="secondary" type="button">Descargar copia</button>
      <button id="resetDraftBtn" class="secondary" type="button">Limpiar borrador</button>
    </div>
  </div>
  <div class="status-bar">
    <span class="chip"><strong id="visibleCount">0</strong> visibles</span>
    <span class="chip"><strong id="labeledCount">0</strong> con gold_label</span>
    <span class="chip"><strong id="dirtyCount">0</strong> cambios locales</span>
    <span class="chip"><strong id="saveStatus">Borrador local activo</strong></span>
  </div>
  <table>
    <thead>
      <tr>
        <th>clip</th>
        <th>split</th>
        <th>meeting</th>
        <th>camera</th>
        <th>window</th>
        <th>gold label</th>
        <th>frames</th>
        <th>video</th>
        <th>suggested_label</th>
        <th>confidence</th>
        <th>face_detected_ratio</th>
        <th>suggestion_summary</th>
      </tr>
    </thead>
    <tbody id="annotationTableBody"></tbody>
  </table>
  <script>
    const FIELDNAMES = __FIELDNAMES_JSON__;
    const STORAGE_KEY = "fer_meetings_annotation_pack_v2";
    const LABELS = ["negative", "neutral", "positive"];
    const annotationRows = __ROWS_JSON__;
    const draftState = new Map();
    const elements = {
      body: document.getElementById("annotationTableBody"),
      splitFilter: document.getElementById("splitFilter"),
      statusFilter: document.getElementById("statusFilter"),
      labelFilter: document.getElementById("labelFilter"),
      searchInput: document.getElementById("searchInput"),
      visibleCount: document.getElementById("visibleCount"),
      labeledCount: document.getElementById("labeledCount"),
      dirtyCount: document.getElementById("dirtyCount"),
      saveStatus: document.getElementById("saveStatus"),
      saveFileBtn: document.getElementById("saveFileBtn"),
      downloadBtn: document.getElementById("downloadBtn"),
      resetDraftBtn: document.getElementById("resetDraftBtn"),
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function labelClass(label) {
      if (!label) return "label-empty";
      return `label-${label}`;
    }

    function labelText(label) {
      return label || "unlabeled";
    }

    function normalizeBool(value) {
      return String(value || "").toLowerCase() === "true";
    }

    function rowSearchText(row) {
      return [row.clip_id, row.meeting_id, row.camera, row.split, row.video_file]
        .join(" ")
        .toLowerCase();
    }

    function currentRows() {
      return annotationRows.map((row) => draftState.get(row.clip_id) || row);
    }

    function serializeDrafts() {
      const changedRows = currentRows().filter((row, index) => {
        const base = annotationRows[index];
        return JSON.stringify(row) !== JSON.stringify(base);
      });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(changedRows));
    }

    function restoreDrafts() {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      try {
        const parsed = JSON.parse(raw);
        parsed.forEach((row) => {
          if (row && row.clip_id) {
            draftState.set(row.clip_id, row);
          }
        });
      } catch (error) {
        console.warn("Could not parse saved draft", error);
      }
    }

    function buildCsv(rows) {
      const escapeCsv = (value) => {
        const text = String(value ?? "");
        if (/[",\\n]/.test(text)) {
          return `"${text.replace(/"/g, '""')}"`;
        }
        return text;
      };
      const lines = [FIELDNAMES.join(",")];
      rows.forEach((row) => {
        lines.push(FIELDNAMES.map((field) => escapeCsv(row[field] ?? "")).join(","));
      });
      return lines.join("\\n") + "\\n";
    }

    function downloadCsv(filename = "annotation_sheet.csv") {
      const blob = new Blob([buildCsv(currentRows())], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(url);
      elements.saveStatus.textContent = "CSV descargado";
    }

    async function saveCsv() {
      const csvText = buildCsv(currentRows());
      if (window.showSaveFilePicker) {
        try {
          const handle = await window.showSaveFilePicker({
            suggestedName: "annotation_sheet.csv",
            types: [
              {
                description: "CSV",
                accept: { "text/csv": [".csv"] },
              },
            ],
          });
          const writable = await handle.createWritable();
          await writable.write(csvText);
          await writable.close();
          elements.saveStatus.textContent = "CSV guardado en disco";
          return;
        } catch (error) {
          if (error && error.name === "AbortError") {
            return;
          }
          console.warn("File picker save failed; falling back to download.", error);
        }
      }
      downloadCsv();
    }

    function updateRow(clipId, updates) {
      const base = annotationRows.find((row) => row.clip_id === clipId);
      if (!base) return;
      const next = { ...(draftState.get(clipId) || base), ...updates };
      draftState.set(clipId, next);
      serializeDrafts();
      renderRows();
    }

    function clearDrafts() {
      draftState.clear();
      localStorage.removeItem(STORAGE_KEY);
      elements.saveStatus.textContent = "Borrador local limpiado";
      renderRows();
    }

    function isDirty(row) {
      const base = annotationRows.find((item) => item.clip_id === row.clip_id);
      return JSON.stringify(row) !== JSON.stringify(base);
    }

    function filteredRows(rows) {
      const query = elements.searchInput.value.trim().toLowerCase();
      const split = elements.splitFilter.value;
      const status = elements.statusFilter.value;
      const label = elements.labelFilter.value;
      return rows.filter((row) => {
        if (split !== "all" && row.split !== split) return false;
        if (status === "labeled" && !row.gold_label) return false;
        if (status === "unlabeled" && row.gold_label) return false;
        if (label !== "all" && row.gold_label !== label) return false;
        if (query && !rowSearchText(row).includes(query)) return false;
        return true;
      });
    }

    function renderRows() {
      const rows = currentRows();
      const visibleRows = filteredRows(rows);
      elements.body.innerHTML = visibleRows.map((row) => {
        const dirty = isDirty(row);
        const excludeChecked = normalizeBool(row.exclude_from_gold) ? "checked" : "";
        const buttonGroup = [
          ...LABELS.map((label) => `
            <button
              type="button"
              class="label-btn ${row.gold_label === label ? "active" : ""}"
              data-action="set-label"
              data-clip-id="${escapeHtml(row.clip_id)}"
              data-label="${label}">
              ${label}
            </button>`),
          `<button
              type="button"
              class="label-btn ${!row.gold_label ? "active" : ""}"
              data-action="set-label"
              data-clip-id="${escapeHtml(row.clip_id)}"
              data-label="">
              clear
            </button>`,
        ].join("");
        return `
          <tr class="${dirty ? "dirty-row" : ""} ${row.exclude_from_gold === "true" ? "excluded-row" : ""}" data-clip-id="${escapeHtml(row.clip_id)}">
            <td class="clip-meta">
              <strong>${escapeHtml(row.clip_id)}</strong>
              <div class="helper">${escapeHtml(row.video_file)}</div>
            </td>
            <td>${escapeHtml(row.split)}</td>
            <td>${escapeHtml(row.meeting_id)}</td>
            <td>${escapeHtml(row.camera)}</td>
            <td>${escapeHtml(row.clip_start_s)} - ${escapeHtml(row.clip_end_s)}</td>
            <td class="control-cell">
              <div class="current-label ${labelClass(row.gold_label)}">${labelText(row.gold_label)}</div>
              <div class="segmented">${buttonGroup}</div>
              <div class="field-row">
                <label>
                  <input
                    type="checkbox"
                    data-action="toggle-exclude"
                    data-clip-id="${escapeHtml(row.clip_id)}"
                    ${excludeChecked}>
                  exclude_from_gold
                </label>
              </div>
              <div class="field-row">
                <label>annotator</label>
                <input
                  type="text"
                  value="${escapeHtml(row.annotator || "")}"
                  data-action="annotator"
                  data-clip-id="${escapeHtml(row.clip_id)}"
                  placeholder="nombre o iniciales">
              </div>
              <div class="field-row">
                <label>notes</label>
                <textarea
                  data-action="notes"
                  data-clip-id="${escapeHtml(row.clip_id)}"
                  placeholder="notas opcionales">${escapeHtml(row.notes || "")}</textarea>
              </div>
              <div class="inline-note">suggested: <strong>${escapeHtml(row.suggested_label || "none")}</strong></div>
            </td>
            <td><img src="${escapeHtml(row.thumbnail_path)}" alt="thumb" loading="lazy"></td>
            <td><video controls preload="metadata" src="${escapeHtml(row.video_rel_path)}"></video></td>
            <td>${escapeHtml(row.suggested_label)}</td>
            <td>${escapeHtml(row.suggested_confidence)}</td>
            <td>${escapeHtml(row.face_detected_ratio)}</td>
            <td>${escapeHtml(row.suggestion_summary)}</td>
          </tr>`;
      }).join("");

      elements.visibleCount.textContent = String(visibleRows.length);
      elements.labeledCount.textContent = String(rows.filter((row) => row.gold_label).length);
      elements.dirtyCount.textContent = String(rows.filter((row) => isDirty(row)).length);
    }

    function installFilters() {
      const splits = [...new Set(annotationRows.map((row) => row.split).filter(Boolean))].sort();
      splits.forEach((split) => {
        const option = document.createElement("option");
        option.value = split;
        option.textContent = split;
        elements.splitFilter.appendChild(option);
      });
      [elements.searchInput, elements.splitFilter, elements.statusFilter, elements.labelFilter].forEach((element) => {
        element.addEventListener("input", renderRows);
        element.addEventListener("change", renderRows);
      });
    }

    elements.body.addEventListener("click", (event) => {
      const button = event.target.closest("[data-action='set-label']");
      if (!button) return;
      updateRow(button.dataset.clipId, { gold_label: button.dataset.label });
      elements.saveStatus.textContent = "Cambios guardados localmente";
    });

    elements.body.addEventListener("change", (event) => {
      const target = event.target;
      const clipId = target.dataset.clipId;
      if (!clipId) return;
      if (target.dataset.action === "toggle-exclude") {
        updateRow(clipId, { exclude_from_gold: target.checked ? "true" : "false" });
        elements.saveStatus.textContent = "Cambios guardados localmente";
      }
    });

    elements.body.addEventListener("input", (event) => {
      const target = event.target;
      const clipId = target.dataset.clipId;
      if (!clipId) return;
      if (target.dataset.action === "annotator") {
        updateRow(clipId, { annotator: target.value });
      }
      if (target.dataset.action === "notes") {
        updateRow(clipId, { notes: target.value });
      }
      elements.saveStatus.textContent = "Cambios guardados localmente";
    });

    elements.saveFileBtn.addEventListener("click", saveCsv);
    elements.downloadBtn.addEventListener("click", () => downloadCsv("annotation_sheet.copy.csv"));
    elements.resetDraftBtn.addEventListener("click", () => {
      if (confirm("Esto borrará el borrador local no exportado. ¿Continuar?")) {
        clearDrafts();
      }
    });

    restoreDrafts();
    installFilters();
    renderRows();
  </script>
</body>
</html>
"""
    document = document.replace(
        "__FIELDNAMES_JSON__",
        json.dumps(
            [
                "clip_id",
                "split",
                "video_file",
                "meeting_id",
                "camera",
                "clip_start_s",
                "clip_end_s",
                "video_path",
                "thumbnail_path",
                "rater_1_label",
                "rater_2_label",
                "adjudicated_label",
                "gold_label",
                "annotator",
                "adjudicator",
                "exclude_from_gold",
                "agreement_status",
                "notes",
                "suggested_label",
                "suggested_confidence",
                "face_detected_ratio",
                "suggestion_summary",
            ]
        ),
    ).replace("__ROWS_JSON__", json.dumps(render_rows, ensure_ascii=False))
    ensure_parent(output_path)
    Path(output_path).write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir = output_dir / "thumbnails"
    labels_output = Path(args.labels_output) if args.labels_output else output_dir / "annotation_sheet.csv"

    manifest_rows = read_csv_rows(args.manifest)
    prediction_rows = read_csv_rows(args.predictions) if args.predictions else []
    prediction_index = build_prediction_index(prediction_rows)
    existing_labels = {}
    if labels_output.exists():
        existing_labels = {row["clip_id"]: row for row in read_csv_rows(labels_output)}
    annotation_rows = build_annotation_rows(
        manifest_rows,
        prediction_index,
        existing_labels,
        output_dir,
        thumbnails_dir,
        frames_per_clip=args.frames_per_clip,
        thumb_height=args.thumb_height,
    )

    fieldnames = [
        "clip_id",
        "split",
        "video_file",
        "meeting_id",
        "camera",
        "clip_start_s",
        "clip_end_s",
        "video_path",
        "thumbnail_path",
        "rater_1_label",
        "rater_2_label",
        "adjudicated_label",
        "gold_label",
        "annotator",
        "adjudicator",
        "exclude_from_gold",
        "agreement_status",
        "notes",
        "suggested_label",
        "suggested_confidence",
        "face_detected_ratio",
        "suggestion_summary",
    ]
    write_csv_rows(labels_output, annotation_rows, fieldnames)

    html_output = output_dir / "index.html"
    render_html(annotation_rows, output_dir, html_output)

    print(f"Wrote annotation sheet to {labels_output}")
    print(f"Wrote HTML review page to {html_output}")
    print(f"Wrote {len(annotation_rows)} thumbnail strips to {thumbnails_dir}")


if __name__ == "__main__":
    main()

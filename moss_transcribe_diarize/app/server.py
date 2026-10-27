import json
from pathlib import Path
from typing import Any

from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

from .ffmpeg import detect_ffmpeg
from .jobs import JobManager
from .model_runner import ModelRunner
from .vllm_runner import VllmRunner


def create_app(
    *,
    model_path: str | Path,
    runs_dir: str | Path = "runs",
    device: str = "auto",
    dtype: str = "bf16",
    prompt: str = DEFAULT_PROMPT,
    max_length: int = 131072,
    max_new_tokens: int = 2048,
    decoding: str = "greedy",
    temperature: float | None = None,
    backend: str = "hf",
    vllm_base_url: str | None = None,
    vllm_model: str | None = None,
    vllm_api_key: str | None = None,
    vllm_timeout: float = 600.0,
):
    try:
        from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    except ImportError as exc:
        raise RuntimeError("Install fastapi, uvicorn, and python-multipart to run the local web app.") from exc

    app = FastAPI(title="MOSS Subtitle Studio")
    if backend == "vllm":
        if not vllm_base_url:
            raise ValueError("--vllm-base-url is required when backend='vllm'.")
        runner = VllmRunner(
            base_url=vllm_base_url,
            model=vllm_model or str(model_path),
            api_key=vllm_api_key,
            timeout=vllm_timeout,
        )
    else:
        runner = ModelRunner(model_path, device=device, dtype=dtype)
    manager = JobManager(
        runs_dir,
        runner,
        prompt=prompt,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        decoding=decoding,
        temperature=temperature,
    )
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(INDEX_HTML, headers={"Cache-Control": "no-store"})

    @app.get("/favicon.svg")
    def favicon():
        return Response(FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    @app.get("/api/runtime")
    def runtime():
        return {
            "ffmpeg": detect_ffmpeg().to_dict(),
            "model": _runner_runtime_info(manager.model_runner),
            "inference": {
                "prompt": manager.prompt,
                "max_length": manager.max_length,
                "max_new_tokens": manager.max_new_tokens,
                "decoding": manager.decoding,
                "temperature": manager.temperature,
            },
        }

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [job.to_dict() for job in manager.list_jobs()]}

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        prompt: str | None = Form(None),
        max_new_tokens: int | None = Form(None),
        max_len: int | None = Form(None),
        decoding: str | None = Form(None),
        temperature: float | None = Form(None),
    ):
        try:
            job, input_path = manager.create_job_for_upload(
                file.filename or "input.media",
                prompt=prompt,
                max_length=max_len,
                max_new_tokens=max_new_tokens,
                decoding=decoding,
                temperature=temperature,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            with input_path.open("wb") as handle:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            manager.enqueue(job.id)
            return job.to_dict()
        except Exception as exc:
            manager._set_status(job, "failed", 1.0, error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return manager.get_job(job_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        try:
            manager.delete_job(job_id)
            return {"ok": True}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/rerun")
    async def rerun_job(job_id: str, request: Request):
        try:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            job = manager.rerun_job(
                job_id,
                prompt=payload.get("prompt"),
                max_length=payload.get("max_len") or payload.get("max_length"),
                max_new_tokens=payload.get("max_new_tokens"),
                decoding=payload.get("decoding"),
                temperature=payload.get("temperature"),
            )
            return job.to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Media file is missing.") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/media")
    def media(job_id: str):
        try:
            job = manager.get_job(job_id)
            path = Path(job.input_path)
            if not path.exists():
                raise FileNotFoundError(str(path))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Media file is missing.") from exc
        return FileResponse(path, filename=path.name)

    @app.get("/api/jobs/{job_id}/segments")
    def get_segments(job_id: str):
        try:
            return {"segments": manager.list_segments(job_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/jobs/{job_id}/segments")
    async def update_segments(job_id: str, request: Request):
        try:
            payload: Any = await request.json()
            segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
            style = payload.get("style") if isinstance(payload, dict) else None
            if not isinstance(segments, list):
                raise ValueError("Expected a JSON list or an object with a segments list.")
            return {"segments": manager.update_segments(job_id, segments, style)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/render")
    async def render(job_id: str, request: Request):
        try:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            job = manager.render(job_id, payload.get("style") if isinstance(payload, dict) else None)
            return job.to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str, kind: str):
        try:
            path = manager.download_path(job_id, kind)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"File is not ready: {kind}") from exc
        return FileResponse(path, filename=path.name)

    return app


def _read_processor_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path).expanduser() / "processor_config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    keys = [
        "audio_tokens_per_second",
        "audio_merge_size",
        "time_marker_every_seconds",
        "enable_time_marker",
    ]
    return {key: data[key] for key in keys if key in data}


def _runner_runtime_info(runner) -> dict[str, Any]:
    if hasattr(runner, "runtime_info"):
        info = dict(runner.runtime_info())
    else:
        info = {
            "backend": "hf",
            "path": runner.model_path,
            "device": runner.device_name,
            "dtype": runner.dtype_name,
        }
    if info.get("backend") == "hf":
        info["processor"] = _read_processor_config(info.get("path") or "")
    else:
        info.setdefault("processor", {})
    return info


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#007d77"/>
  <rect x="11" y="12" width="42" height="40" rx="8" fill="#fffdfa"/>
  <rect x="16" y="17" width="32" height="19" rx="4" fill="#1d1f22"/>
  <path d="M29 22v9.5l8.5-4.75z" fill="#c94b35"/>
  <rect x="17" y="41" width="30" height="4" rx="2" fill="#007d77"/>
  <rect x="17" y="48" width="21" height="3.5" rx="1.75" fill="#6d6a63"/>
</svg>"""


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MOSS 字幕工作台</title>
  <link rel="icon" type="image/svg+xml" href="favicon.svg" />
  <style>
    :root {
      --bg: #f7f5f0;
      --panel: #ffffff;
      --line: #d8d3c7;
      --text: #1d1f22;
      --muted: #6d6a63;
      --teal: #007d77;
      --coral: #c94b35;
      --green: #2f7d4f;
      --sidebar-width: 300px;
      --sidebar-collapsed-width: 12px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: #fffdfa;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 720; }
    main {
      height: calc(100vh - 56px);
      display: grid;
      grid-template-columns: var(--sidebar-width) minmax(0, 1fr);
      overflow: hidden;
    }
    body.sidebar-collapsed main {
      grid-template-columns: var(--sidebar-collapsed-width) minmax(0, 1fr);
    }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
    input, select, button, textarea {
      font: inherit;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: white;
      color: var(--text);
    }
    input[type="file"], input[type="number"], input[type="text"], select {
      width: 100%;
      padding: 8px;
    }
    button {
      min-height: 36px;
      padding: 8px 12px;
      cursor: pointer;
      background: #fff;
    }
    button.primary { background: var(--teal); border-color: var(--teal); color: white; }
    button.warn { background: var(--coral); border-color: var(--coral); color: white; }
    button.ghost { background: transparent; }
    button.saved { color: var(--muted); background: #f6f3ec; }
    button.small {
      min-height: 28px;
      padding: 4px 8px;
      font-size: 12px;
    }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      background: white;
      font-size: 12px;
      white-space: nowrap;
    }
    .pill.ok { color: var(--green); border-color: #9cc5aa; }
    .pill.bad { color: var(--coral); border-color: #d6a193; }
    .muted { color: var(--muted); font-size: 13px; }
    .meta { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .error { color: var(--coral); font-size: 13px; overflow-wrap: anywhere; }
    .warning { color: #9b5a00; font-size: 12px; overflow-wrap: anywhere; }
    .save-status {
      font-size: 12px;
      color: var(--muted);
      min-height: 18px;
      margin-top: 8px;
    }
    .save-status.dirty { color: #946b00; }
    .save-status.saving { color: var(--teal); }
    .save-status.saved { color: var(--green); }
    .save-status.error { color: var(--coral); }
    .is-hidden { display: none !important; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .task-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }
    .task-title {
      font-weight: 700;
      font-size: 18px;
      line-height: 1.25;
      overflow-wrap: anywhere;
      margin-bottom: 8px;
    }
    .task-meta {
      margin-top: 4px;
      line-height: 1.35;
    }
    .task-notice {
      margin-top: 10px;
      line-height: 1.35;
    }
    .primary-action {
      width: 100%;
      min-height: 42px;
      margin-top: 12px;
      font-weight: 650;
    }
    .secondary-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-top: 10px;
    }
    .secondary-actions > div {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      min-width: 0;
    }
    .secondary-actions button {
      min-height: 30px;
      padding: 5px 9px;
      font-size: 13px;
    }
    .secondary-actions .save-status {
      margin-top: 0;
      min-height: auto;
      white-space: nowrap;
    }
    .progress {
      height: 8px;
      background: #ebe6dc;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar { width: 0%; height: 100%; background: var(--teal); transition: width 160ms ease; }
    .sidebar {
      position: relative;
      border-right: 1px solid var(--line);
      background: #fbfaf7;
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 0;
      overflow: visible;
    }
    .sidebar-collapsed .sidebar {
      grid-template-rows: 1fr;
      border-right: 0;
      background: #f4f1e9;
    }
    .sidebar-head {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    .sidebar-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }
    .sidebar-title strong { font-size: 15px; }
    .sidebar-tools {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      margin-bottom: 10px;
    }
    .sidebar-primary {
      width: 100%;
      min-height: 34px;
    }
    .sidebar-toggle-zone {
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      width: 24px;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: flex-end;
    }
    .sidebar-toggle-zone::before {
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      right: 0;
      width: 1px;
      background: rgba(216, 211, 199, 0.8);
    }
    .sidebar-toggle-zone:hover::before,
    .sidebar-toggle-zone:focus-within::before {
      background: var(--teal);
    }
    .sidebar-toggle {
      width: 28px;
      height: 48px;
      min-height: 48px;
      padding: 0;
      border-radius: 999px;
      border-color: #cfc8ba;
      background: #fffdfa;
      color: var(--text);
      box-shadow: 0 4px 14px rgba(24, 25, 26, 0.12);
      opacity: 0;
      pointer-events: none;
      transform: translateX(10px);
      transition: opacity 120ms ease, transform 120ms ease, border-color 120ms ease;
    }
    .sidebar-toggle::before { content: "‹"; font-size: 18px; line-height: 1; }
    .sidebar-toggle-zone:hover .sidebar-toggle,
    .sidebar-toggle:focus-visible {
      opacity: 1;
      pointer-events: auto;
      transform: translateX(14px);
    }
    .sidebar-toggle:hover,
    .sidebar-toggle:focus-visible { border-color: var(--teal); }
    .sidebar-collapsed .sidebar-head,
    .sidebar-collapsed .task-list {
      display: none;
    }
    .sidebar-collapsed .sidebar-toggle-zone {
      left: 0;
      right: auto;
      width: 30px;
      justify-content: flex-start;
    }
    .sidebar-collapsed .sidebar-toggle-zone::before {
      right: auto;
      left: 0;
      background: #cfc8ba;
    }
    .sidebar-collapsed .sidebar-toggle::before { content: "›"; }
    .sidebar-collapsed .sidebar-toggle-zone:hover .sidebar-toggle,
    .sidebar-collapsed .sidebar-toggle:focus-visible {
      transform: translateX(8px);
    }
    .task-list {
      overflow: auto;
      padding: 8px;
    }
    .task-item {
      width: 100%;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 9px;
      background: transparent;
      cursor: pointer;
      text-align: left;
      margin-bottom: 6px;
    }
    .task-item:hover { background: #fffdfa; border-color: var(--line); }
    .task-item.active { background: white; border-color: #9cc5aa; }
    .task-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .task-name {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 640;
    }
    .task-id { margin-top: 4px; }
    .task-foot {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-top: 8px;
    }
    .task-progress {
      flex: 1 1 auto;
      min-width: 54px;
    }
    .content {
      min-width: 0;
      min-height: 0;
      overflow: hidden;
    }
    .view {
      height: 100%;
      min-height: 0;
      overflow: hidden;
    }
    .center-view {
      height: 100%;
      display: grid;
      align-items: start;
      justify-items: center;
      padding: 28px;
      overflow: auto;
    }
    .import-panel,
    .process-panel {
      width: min(680px, 100%);
    }
    .view-title {
      margin: 0 0 16px;
      font-size: 22px;
      font-weight: 760;
    }
    textarea.prompt-input {
      width: 100%;
      min-height: 112px;
      resize: vertical;
      padding: 8px;
    }
    details.advanced {
      width: 100%;
      margin: 10px 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fffdfa;
    }
    details.advanced summary {
      min-height: 38px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 10px;
      cursor: pointer;
      color: var(--text);
      font-weight: 650;
      font-size: 13px;
    }
    details.advanced summary::-webkit-details-marker { display: none; }
    details.advanced summary::marker { content: ""; }
    details.advanced summary::after {
      content: "+";
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      border: 1px solid var(--line);
      color: var(--teal);
      background: white;
      flex: 0 0 auto;
      font-weight: 800;
    }
    details.advanced[open] summary::after { content: "-"; }
    .advanced-title { display: block; }
    .advanced-hint {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }
    .advanced-body {
      padding: 0 10px 10px;
      border-top: 1px solid var(--line);
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .process-panel .progress { margin: 16px 0 8px; }
    .workbench {
      height: 100%;
      padding: 12px 14px 14px;
      overflow: hidden;
    }
    .editor-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 12px;
      height: 100%;
      min-height: 0;
    }
    .preview-column {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 10px;
      min-width: 0;
      min-height: 0;
    }
    .video-shell {
      width: 100%;
      min-height: 0;
      background: transparent;
      overflow: hidden;
      display: flex;
      align-items: flex-start;
      justify-content: center;
    }
    .video-stage {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      max-height: 48vh;
      max-width: 100%;
      background: #111;
      border-radius: 6px;
      overflow: hidden;
      flex: 0 0 auto;
    }
    video {
      width: 100%;
      height: 100%;
      background: #111;
      display: block;
      object-fit: contain;
    }
    .subtitle-overlay {
      position: absolute;
      left: 50%;
      bottom: 56px;
      width: max-content;
      max-width: 92%;
      padding: 0 2%;
      transform: translateX(-50%);
      display: none;
      justify-content: center;
      pointer-events: none;
      z-index: 2;
      text-align: center;
      color: white;
      font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", sans-serif;
      font-size: 48px;
      font-weight: 400;
      line-height: 1.448;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: normal;
      -webkit-text-stroke: 3px #000;
      paint-order: stroke fill;
      text-shadow: 0 2px 3px rgba(0, 0, 0, 0.65);
    }
    .subtitle-overlay.visible { display: flex; }
    .table-wrap {
      overflow: auto;
      min-width: 0;
      min-height: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      scrollbar-gutter: stable;
    }
    .subtitle-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      table-layout: fixed;
      font-size: 14px;
      line-height: 1.35;
    }
    .subtitle-table th,
    .subtitle-table td {
      border-bottom: 1px solid #ece7dc;
      padding: 4px 8px;
      vertical-align: middle;
    }
    .subtitle-table th {
      position: sticky;
      top: 0;
      z-index: 2;
      height: 30px;
      background: #fffdfa;
      box-shadow: 0 1px 0 var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-align: left;
    }
    .subtitle-table th.time { width: 76px; }
    .subtitle-table th.speaker { width: 68px; }
    .subtitle-table th.action { width: 210px; text-align: center; }
    .subtitle-table tbody tr {
      cursor: pointer;
      background: var(--speaker-row-bg, #fff);
      color: #20242c;
      transition: background 120ms ease;
    }
    .subtitle-table tbody tr:nth-child(even) { background: var(--speaker-row-bg, #fff); }
    .subtitle-table tbody tr:hover {
      background: color-mix(in srgb, var(--speaker-row-bg, #fff) 88%, #d8d3c7);
    }
    .subtitle-table tbody tr.active { background: var(--speaker-row-bg, #e1f1ee); }
    .subtitle-table tbody tr.active td {
      border-bottom-color: #a8d0ca;
      box-shadow: inset 3px 0 0 var(--teal);
    }
    .subtitle-table tbody tr.active td:first-child { box-shadow: inset 3px 0 0 var(--teal); }
    .subtitle-table tbody tr.active td:not(:first-child) { box-shadow: none; }
    .subtitle-table tbody tr:not(.active) td:first-child {
      box-shadow: inset 4px 0 0 var(--speaker-row-accent, transparent);
    }
    .subtitle-table input,
    .subtitle-table textarea {
      width: 100%;
      min-width: 0;
      border: 1px solid transparent;
      border-radius: 4px;
      background: transparent;
      color: var(--text);
      font: inherit;
      line-height: 1.35;
      transition: border-color 120ms ease, background 120ms ease, box-shadow 120ms ease;
    }
    .subtitle-table input {
      height: 30px;
      padding: 4px 5px;
      font-variant-numeric: tabular-nums;
    }
    .subtitle-table input.start,
    .subtitle-table input.end,
    .subtitle-table input.speaker {
      color: #313438;
    }
    .subtitle-table input.start,
    .subtitle-table input.end {
      text-align: right;
    }
    .subtitle-table input.speaker {
      text-align: center;
      font-weight: 600;
    }
    .subtitle-table textarea {
      display: block;
      min-height: 30px;
      max-height: 48px;
      padding: 5px 6px;
      resize: none;
      overflow: hidden;
      white-space: pre-wrap;
    }
    .subtitle-table tr.active textarea,
    .subtitle-table textarea:focus {
      max-height: 112px;
    }
    .subtitle-table input:focus,
    .subtitle-table textarea:focus {
      outline: none;
      border-color: #86bcb5;
      background: #fff;
      box-shadow: 0 0 0 2px rgba(0, 125, 119, 0.12);
    }
    .subtitle-table input::-webkit-outer-spin-button,
    .subtitle-table input::-webkit-inner-spin-button {
      margin: 0;
      -webkit-appearance: none;
    }
    .subtitle-table input[type="number"] { -moz-appearance: textfield; }
    .subtitle-table td.action { text-align: center; }
    .segment-actions {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
    }
    .segment-action {
      min-height: 26px;
      padding: 3px 7px;
      border-color: transparent;
      background: transparent;
      font-size: 12px;
      white-space: nowrap;
    }
    .segment-merge { color: var(--teal); }
    .segment-add { color: #456b9a; }
    .segment-delete { color: var(--coral); }
    .segment-merge:hover,
    .segment-merge:focus-visible {
      border-color: #86bcb5;
      background: #eff9f7;
      outline: none;
    }
    .segment-add:hover,
    .segment-add:focus-visible {
      border-color: #9eb4cf;
      background: #f1f6fb;
      outline: none;
    }
    .segment-delete:hover,
    .segment-delete:focus-visible {
      border-color: #d6a193;
      background: #fff4f1;
      outline: none;
    }
    .inspector {
      border-left: 1px solid var(--line);
      padding-left: 14px;
      min-width: 0;
      overflow: auto;
    }
    .group {
      border-bottom: 1px solid var(--line);
      padding: 0 0 16px;
      margin-bottom: 16px;
    }
    .group:last-child { border-bottom: 0; }
    .speaker-map {
      display: grid;
      gap: 8px;
    }
    .speaker-map-row {
      display: grid;
      grid-template-columns: 54px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
    }
    .offset-controls {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
    }
    .offset-controls button { white-space: nowrap; }
    .offset-quick-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .offset-quick-actions button {
      min-height: 30px;
      padding: 5px 8px;
      font-size: 12px;
    }
    .speaker-tag {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .downloads a {
      display: inline-flex;
      align-items: center;
      min-height: 32px;
      padding: 6px 10px;
      margin: 0 6px 6px 0;
      border-radius: 6px;
      color: var(--teal);
      background: white;
      border: 1px solid var(--line);
      text-decoration: none;
    }
    @media (max-width: 900px) {
      body { overflow: auto; }
      main { height: auto; grid-template-columns: 1fr; }
      body.sidebar-collapsed main { grid-template-columns: 1fr; }
      .sidebar { min-height: 260px; border-right: 0; border-bottom: 1px solid var(--line); }
      .sidebar-collapsed .sidebar { min-height: 42px; }
      .sidebar-toggle-zone { display: none; }
      .view, .workbench { height: auto; }
      .editor-grid { grid-template-columns: 1fr; }
      .inspector { border-left: 0; padding-left: 0; }
    }
  </style>
</head>
<body>
  <header>
    <h1>MOSS 字幕工作台</h1>
    <span id="runtime" class="pill">检测中</span>
  </header>
  <main>
    <aside class="sidebar">
      <div class="sidebar-head">
        <div class="sidebar-title">
          <strong>任务</strong>
        </div>
        <div class="sidebar-tools">
          <span id="jobCount" class="meta">0 个任务</span>
          <button id="refreshJobs" class="small ghost">刷新</button>
        </div>
        <button id="newTask" class="primary sidebar-primary">导入媒体</button>
      </div>
      <div class="sidebar-toggle-zone">
        <button id="sidebarToggle" class="sidebar-toggle" type="button" aria-label="收起任务栏" title="收起任务栏"></button>
      </div>
      <div id="jobList" class="task-list"></div>
    </aside>
    <section class="content">
      <section id="importView" class="view center-view">
        <div class="import-panel">
          <h2 id="importTitle" class="view-title">导入媒体</h2>
          <label for="file">媒体文件</label>
          <input id="file" type="file" accept="audio/*,video/*,.mp4,.mov,.mkv,.wav,.mp3,.m4a" />
          <div id="rerunSource" class="meta" style="margin-top:8px"></div>
          <details class="advanced">
            <summary>
              <span>
                <span class="advanced-title">推理参数</span>
                <span class="advanced-hint">默认不用改</span>
              </span>
            </summary>
            <div class="advanced-body">
              <label for="prompt" style="margin-top:10px">推理 Prompt</label>
              <textarea id="prompt" class="prompt-input"></textarea>
              <div class="row" style="margin-top:10px">
                <div>
                  <label for="maxNewTokens">输出 tokens</label>
                  <input id="maxNewTokens" type="number" min="1" step="1" value="2048" />
                </div>
                <div>
                  <label for="maxLen">上下文上限</label>
                  <input id="maxLen" type="number" min="1" step="1" value="131072" />
                </div>
              </div>
              <div class="row" style="margin-top:10px">
                <div>
                  <label for="decoding">解码</label>
                  <select id="decoding"><option value="greedy">greedy</option><option value="sample">sample</option></select>
                </div>
                <div>
                  <label for="temperature">温度</label>
                  <input id="temperature" type="number" min="0.01" step="0.05" value="1.0" />
                </div>
              </div>
              <div id="modelinfo" class="meta" style="margin-top:8px"></div>
            </div>
          </details>
          <button id="upload" class="primary">开始转写</button>
          <div id="importError" class="error" style="margin-top:10px"></div>
        </div>
      </section>
      <section id="processingView" class="view center-view is-hidden">
        <div class="process-panel">
          <h2 id="processTitle" class="view-title">转写中</h2>
          <div id="processName"></div>
          <div id="processMeta" class="meta" style="margin-top:8px"></div>
          <div class="progress"><div id="processBar" class="bar"></div></div>
          <div id="processError" class="error"></div>
          <div class="actions" style="margin-top:14px">
            <button id="deleteCurrent">删除任务</button>
            <button id="openNew">新建任务</button>
          </div>
        </div>
      </section>
      <section id="workbench" class="view workbench is-hidden">
        <div class="editor-grid">
          <div class="preview-column">
            <div class="video-shell">
              <div id="videoStage" class="video-stage">
                <video id="preview" controls></video>
                <div id="subtitleOverlay" class="subtitle-overlay"></div>
              </div>
            </div>
            <div class="table-wrap">
              <table class="subtitle-table">
                <thead>
                  <tr>
                    <th class="time">开始</th>
                    <th class="time">结束</th>
                    <th class="speaker">说话人</th>
                    <th>字幕</th>
                    <th class="action">操作</th>
                  </tr>
                </thead>
                <tbody id="segments"></tbody>
              </table>
            </div>
          </div>
          <div class="inspector">
            <div class="group">
              <div class="task-header">
                <label style="margin:0">当前任务</label>
                <span id="taskStatus" class="pill">待校对</span>
              </div>
              <div id="selectedName" class="task-title"></div>
              <div id="taskUsage" class="meta task-meta"></div>
              <div id="taskParams" class="meta task-meta"></div>
              <div id="taskNotice" class="task-notice is-hidden"></div>
              <button id="render" class="warn primary-action">烧录视频</button>
              <div class="secondary-actions">
                <div>
                  <button id="undo" class="ghost" type="button" disabled title="撤销上一步编辑">撤销</button>
                  <button id="save" class="primary is-hidden">保存修改</button>
                  <div id="saveStatus" class="save-status saved">已保存</div>
                </div>
                <button id="rerun" class="ghost">重新转写</button>
              </div>
            </div>
            <div class="group">
              <label>说话人名称</label>
              <div id="speakerMap" class="speaker-map"></div>
            </div>
            <div class="group">
              <label for="timelineOffset">整体时间偏移（秒）</label>
              <div class="offset-controls">
                <input id="timelineOffset" type="number" step="0.05" value="0" />
                <button id="applyTimelineOffset" type="button">应用偏移</button>
              </div>
              <div class="offset-quick-actions">
                <button type="button" data-timeline-offset="-0.1">提前 0.1 秒</button>
                <button type="button" data-timeline-offset="0.1">延后 0.1 秒</button>
              </div>
              <div class="meta" style="margin-top:6px">负数提前，正数延后；应用后需要保存。</div>
            </div>
            <div class="group">
              <div class="row">
                <div>
                  <label for="fontSize">字号</label>
                  <input id="fontSize" type="number" min="18" max="96" value="48" />
                </div>
                <div>
                  <label for="marginV">底边距</label>
                  <input id="marginV" type="number" min="12" max="220" value="56" />
                </div>
              </div>
              <div class="row" style="margin-top:10px">
                <div>
                  <label for="showSpeaker">说话人</label>
                  <select id="showSpeaker"><option value="true">显示</option><option value="false">隐藏</option></select>
                </div>
                <div>
                  <label for="speakerColors">颜色</label>
                  <select id="speakerColors"><option value="true">按说话人</option><option value="false">统一</option></select>
                </div>
              </div>
            </div>
            <div class="group">
              <label>输出</label>
              <div class="downloads" id="downloads"></div>
            </div>
          </div>
        </div>
      </section>
    </section>
  </main>
<script>
const RUNNING_STATES = new Set(['queued', 'loading_model', 'transcribing', 'postprocessing', 'rendering']);
const EDIT_STATES = new Set(['waiting_review', 'rendering', 'done']);
const TERMINAL_STATES = new Set(['waiting_review', 'done', 'failed', 'cancelled']);
const fileInput = document.querySelector('#file');
const importTitleEl = document.querySelector('#importTitle');
const rerunSourceEl = document.querySelector('#rerunSource');
const promptInput = document.querySelector('#prompt');
const advancedDetails = document.querySelector('.advanced');
const maxNewTokensInput = document.querySelector('#maxNewTokens');
const maxLenInput = document.querySelector('#maxLen');
const decodingSelect = document.querySelector('#decoding');
const temperatureInput = document.querySelector('#temperature');
const uploadBtn = document.querySelector('#upload');
const newTaskBtn = document.querySelector('#newTask');
const refreshJobsBtn = document.querySelector('#refreshJobs');
const sidebarToggleBtn = document.querySelector('#sidebarToggle');
const deleteCurrentBtn = document.querySelector('#deleteCurrent');
const openNewBtn = document.querySelector('#openNew');
const undoBtn = document.querySelector('#undo');
const saveBtn = document.querySelector('#save');
const renderBtn = document.querySelector('#render');
const rerunBtn = document.querySelector('#rerun');
const timelineOffsetInput = document.querySelector('#timelineOffset');
const applyTimelineOffsetBtn = document.querySelector('#applyTimelineOffset');
const saveStatusEl = document.querySelector('#saveStatus');
const importView = document.querySelector('#importView');
const processingView = document.querySelector('#processingView');
const workbench = document.querySelector('#workbench');
const runtimeEl = document.querySelector('#runtime');
const jobListEl = document.querySelector('#jobList');
const jobCountEl = document.querySelector('#jobCount');
const importErrorEl = document.querySelector('#importError');
const processTitleEl = document.querySelector('#processTitle');
const processNameEl = document.querySelector('#processName');
const processMetaEl = document.querySelector('#processMeta');
const processBarEl = document.querySelector('#processBar');
const processErrorEl = document.querySelector('#processError');
const selectedNameEl = document.querySelector('#selectedName');
const taskStatusEl = document.querySelector('#taskStatus');
const taskUsageEl = document.querySelector('#taskUsage');
const taskParamsEl = document.querySelector('#taskParams');
const taskNoticeEl = document.querySelector('#taskNotice');
const modelInfoEl = document.querySelector('#modelinfo');
const tbody = document.querySelector('#segments');
const speakerMapEl = document.querySelector('#speakerMap');
const videoStage = document.querySelector('#videoStage');
const videoShell = document.querySelector('.video-shell');
const preview = document.querySelector('#preview');
const subtitleOverlay = document.querySelector('#subtitleOverlay');
const downloads = document.querySelector('#downloads');
let jobs = [];
let currentJob = null;
let rerunDraftJob = null;
let pollTimer = null;
let ffmpegAvailable = false;
let activeSegmentIndex = -1;
let assPlayRes = { x: 1920, y: 1080 };
let layoutFitFrame = 0;
let editorDirty = false;
let saveStatusTimer = 0;
let speakerNameMap = {};
let playbackSyncHandle = 0;
let lastPlaybackSyncAt = 0;
let editorHistory = [];
let currentEditorSnapshot = '';
let savedEditorSnapshot = '';
let historyRestoring = false;
let lastHistoryInputKey = '';
let lastHistoryInputAt = 0;
let manualSegmentCounter = 0;
const assFontLineHeightFactor = 1.448;
const speakerPalette = ['#ffffff', '#ffe75b', '#8ff286', '#ffa7bb', '#ffd700', '#6bb5ff', '#db8eff', '#d8d8d8'];
const speakerRowPalette = ['#f1f3f5', '#fff7b8', '#dcf8d8', '#ffe1e9', '#fff0ad', '#dcecff', '#efddff', '#ececec'];

function apiUrl(path) {
  const clean = String(path).replace(/^\\/+/, '');
  const basePath = window.location.pathname.endsWith('/') ? window.location.pathname : window.location.pathname + '/';
  return new URL(clean, window.location.origin + basePath).toString();
}

async function refreshRuntime() {
  try {
    const res = await fetch(apiUrl('api/runtime'), { cache: 'no-store' });
    if (!res.ok) throw new Error('runtime status ' + res.status);
    const data = await res.json();
    ffmpegAvailable = !!(data.ffmpeg && data.ffmpeg.available);
    runtimeEl.textContent = ffmpegAvailable ? 'FFmpeg 可用' : 'FFmpeg 缺失';
    runtimeEl.className = 'pill ' + (ffmpegAvailable ? 'ok' : 'bad');
    renderBtn.disabled = !ffmpegAvailable;
    applyInferenceDefaults(data.inference || {});
    renderModelInfo(data.model || {});
  } catch (err) {
    ffmpegAvailable = false;
    runtimeEl.textContent = 'API 连接失败';
    runtimeEl.className = 'pill bad';
    renderBtn.disabled = true;
    importErrorEl.textContent = '无法连接 api/runtime，请确认页面来自 mtd-subtitle-web 服务。';
  }
}

function applyInferenceDefaults(defaults) {
  if (!promptInput.value && defaults.prompt) promptInput.value = defaults.prompt;
  if (defaults.max_new_tokens) maxNewTokensInput.value = defaults.max_new_tokens;
  if (defaults.max_length) maxLenInput.value = defaults.max_length;
  if (defaults.decoding) decodingSelect.value = defaults.decoding;
  if (defaults.temperature) temperatureInput.value = defaults.temperature;
  updateDecodingControls();
}

function renderModelInfo(model) {
  const parts = [];
  if (model.path) {
    const pathParts = String(model.path).split('/');
    parts.push(pathParts.slice(-2).join('/'));
  }
  if (model.device) parts.push(model.device);
  if (model.dtype) parts.push(model.dtype);
  const processor = model.processor || {};
  if (processor.time_marker_every_seconds) parts.push('time marker ' + processor.time_marker_every_seconds + 's');
  modelInfoEl.textContent = parts.join(' · ');
}

function updateDecodingControls() {
  temperatureInput.disabled = decodingSelect.value !== 'sample';
}

function scheduleLayoutFit() {
  if (layoutFitFrame) cancelAnimationFrame(layoutFitFrame);
  layoutFitFrame = requestAnimationFrame(() => {
    layoutFitFrame = 0;
    fitVideoStageToMedia();
  });
}

function setSidebarCollapsed(collapsed, persist = true) {
  document.body.classList.toggle('sidebar-collapsed', collapsed);
  sidebarToggleBtn.setAttribute('aria-label', collapsed ? '展开任务栏' : '收起任务栏');
  sidebarToggleBtn.title = collapsed ? '展开任务栏' : '收起任务栏';
  if (persist) {
    try {
      localStorage.setItem('mtdSidebarCollapsed', collapsed ? '1' : '0');
    } catch (err) {}
  }
  scheduleLayoutFit();
}

function restoreSidebarState() {
  try {
    setSidebarCollapsed(localStorage.getItem('mtdSidebarCollapsed') === '1', false);
  } catch (err) {
    setSidebarCollapsed(false);
  }
}

function setSaveState(state, message) {
  if (saveStatusTimer) {
    clearTimeout(saveStatusTimer);
    saveStatusTimer = 0;
  }
  saveStatusEl.className = 'save-status ' + state;
  saveStatusEl.textContent = message;
  const showButton = state === 'dirty' || state === 'saving' || state === 'error';
  saveBtn.classList.toggle('is-hidden', !showButton);
  saveBtn.classList.toggle('primary', showButton);
  saveBtn.classList.toggle('saved', false);
  saveBtn.disabled = state === 'saving' || !currentJob;
  if (state === 'dirty') saveBtn.textContent = '保存修改';
  else if (state === 'saving') saveBtn.textContent = '保存中...';
  else if (state === 'error') saveBtn.textContent = '重试保存';
  else saveBtn.textContent = '保存修改';
}

function setEditorDirty(dirty) {
  editorDirty = dirty;
  if (dirty) setSaveState('dirty', '有未保存修改');
  else setSaveState('saved', '已保存');
}

function markEditorDirty() {
  if (!currentJob) return;
  setEditorDirty(true);
}

function serializeEditorState() {
  return JSON.stringify({
    segments: collectSegments(),
    style: collectSubtitleStyle()
  });
}

function updateUndoAction() {
  undoBtn.disabled = !currentJob || editorHistory.length === 0;
  undoBtn.title = undoBtn.disabled ? '没有可撤销的修改' : '撤销上一步编辑';
}

function clearEditorHistory() {
  editorHistory = [];
  currentEditorSnapshot = '';
  savedEditorSnapshot = '';
  lastHistoryInputKey = '';
  lastHistoryInputAt = 0;
  updateUndoAction();
}

function resetEditorHistory() {
  editorHistory = [];
  currentEditorSnapshot = serializeEditorState();
  savedEditorSnapshot = currentEditorSnapshot;
  lastHistoryInputKey = '';
  lastHistoryInputAt = 0;
  updateUndoAction();
}

function pushEditorSnapshot(snapshot) {
  if (!snapshot || editorHistory[editorHistory.length - 1] === snapshot) return;
  editorHistory.push(snapshot);
  if (editorHistory.length > 50) editorHistory.shift();
}

function checkpointEditorState() {
  if (!currentJob || historyRestoring) return;
  const snapshot = serializeEditorState();
  pushEditorSnapshot(snapshot);
  currentEditorSnapshot = snapshot;
  lastHistoryInputKey = '';
  lastHistoryInputAt = 0;
  updateUndoAction();
}

function editorInputKey(element) {
  const row = element && element.closest ? element.closest('tr') : null;
  if (row) return (row.dataset.id || row.dataset.index || 'row') + ':' + element.className;
  if (element && element.dataset && element.dataset.speaker) return 'speaker-name:' + element.dataset.speaker;
  return element && (element.id || element.name || element.className) || 'editor';
}

function recordEditorInputChange(element) {
  if (!currentJob || historyRestoring) return;
  const nextSnapshot = serializeEditorState();
  if (!currentEditorSnapshot) currentEditorSnapshot = nextSnapshot;
  if (nextSnapshot === currentEditorSnapshot) return;
  const now = Date.now();
  const inputKey = editorInputKey(element);
  const grouped = inputKey === lastHistoryInputKey && now - lastHistoryInputAt < 800;
  if (!grouped) pushEditorSnapshot(currentEditorSnapshot);
  currentEditorSnapshot = nextSnapshot;
  lastHistoryInputKey = inputKey;
  lastHistoryInputAt = now;
  updateUndoAction();
}

function finalizeEditorMutation() {
  currentEditorSnapshot = serializeEditorState();
  updateUndoAction();
}

function undoLastEditorChange() {
  if (!currentJob || !editorHistory.length) return;
  const snapshot = editorHistory.pop();
  const state = JSON.parse(snapshot);
  historyRestoring = true;
  editorDirty = false;
  applySubtitleStyle(state.style || {});
  renderSegments(state.segments || []);
  historyRestoring = false;
  currentEditorSnapshot = serializeEditorState();
  lastHistoryInputKey = '';
  lastHistoryInputAt = 0;
  const dirty = currentEditorSnapshot !== savedEditorSnapshot;
  setEditorDirty(dirty);
  if (dirty) setSaveState('dirty', '已撤销上一步，待保存');
  else setSaveState('saved', '已撤销到已保存版本');
  updateUndoAction();
}

decodingSelect.addEventListener('change', updateDecodingControls);

newTaskBtn.addEventListener('click', () => showImportView({ clearDraft: true }));
openNewBtn.addEventListener('click', () => showImportView({ clearDraft: true }));
refreshJobsBtn.addEventListener('click', () => refreshJobs());
sidebarToggleBtn.addEventListener('click', () => {
  setSidebarCollapsed(!document.body.classList.contains('sidebar-collapsed'));
});
deleteCurrentBtn.addEventListener('click', async () => {
  if (currentJob) await deleteJob(currentJob.id);
});

jobListEl.addEventListener('click', async (event) => {
  const deleteButton = event.target.closest('[data-delete-id]');
  if (deleteButton) {
    event.stopPropagation();
    await deleteJob(deleteButton.dataset.deleteId);
    return;
  }
  const item = event.target.closest('[data-job-id]');
  if (item) await selectJob(item.dataset.jobId);
});

fileInput.addEventListener('change', () => {
  if (rerunDraftJob) resetImportMode();
  const file = fileInput.files[0];
  if (file) {
    preview.src = URL.createObjectURL(file);
    resetVideoStage();
  }
});

uploadBtn.addEventListener('click', async () => {
  if (rerunDraftJob) {
    await startRerunDraft();
    return;
  }
  const file = fileInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  form.append('prompt', promptInput.value);
  if (maxNewTokensInput.value) form.append('max_new_tokens', maxNewTokensInput.value);
  if (maxLenInput.value) form.append('max_len', maxLenInput.value);
  form.append('decoding', decodingSelect.value);
  if (temperatureInput.value) form.append('temperature', temperatureInput.value);
  uploadBtn.disabled = true;
  advancedDetails.open = false;
  importErrorEl.textContent = '';
  showProcessingPlaceholder(file.name);
  const res = await fetch(apiUrl('api/jobs'), { method: 'POST', body: form });
  const job = await res.json();
  uploadBtn.disabled = false;
  if (!res.ok) {
    importErrorEl.textContent = job.detail || '上传失败';
    showImportView({ preserveError: true });
    return;
  }
  currentJob = job;
  await refreshJobs({ keepSelection: true });
  await selectJob(job.id);
});

saveBtn.addEventListener('click', async () => {
  await saveSegments();
});

undoBtn.addEventListener('click', undoLastEditorChange);
document.addEventListener('keydown', (event) => {
  const target = event.target;
  const editingField = target && target.closest && target.closest('input, textarea, select, [contenteditable="true"]');
  if ((event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === 'z' && !editingField) {
    event.preventDefault();
    undoLastEditorChange();
  }
});

renderBtn.addEventListener('click', async () => {
  if (!currentJob || !ffmpegAvailable) return;
  const saved = await saveSegments();
  if (!saved) return;
  const style = collectSubtitleStyle();
  const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/render`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ style })
  });
  const data = await res.json();
  if (!res.ok) setTaskNotice(data.detail || '烧录失败', 'error');
  else {
    currentJob = data;
    renderCurrentJob(data);
    await refreshJobs({ keepSelection: true });
  }
});

rerunBtn.addEventListener('click', () => {
  if (currentJob) showRerunDraft(currentJob);
});

applyTimelineOffsetBtn.addEventListener('click', applyTimelineOffset);
timelineOffsetInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') applyTimelineOffset();
});
document.querySelectorAll('[data-timeline-offset]').forEach((button) => {
  button.addEventListener('click', () => applyTimelineOffset(Number(button.dataset.timelineOffset)));
});

preview.addEventListener('timeupdate', syncActiveSegment);
preview.addEventListener('seeked', syncActiveSegment);
preview.addEventListener('play', startPlaybackSync);
preview.addEventListener('pause', stopPlaybackSync);
preview.addEventListener('ended', stopPlaybackSync);
preview.addEventListener('loadedmetadata', () => {
  fitVideoStageToMedia();
  syncActiveSegment();
});
window.addEventListener('resize', () => {
  scheduleLayoutFit();
});
if ('ResizeObserver' in window) {
  const layoutObserver = new ResizeObserver(scheduleLayoutFit);
  for (const element of [videoShell, document.querySelector('.content'), document.querySelector('.editor-grid')]) {
    if (element) layoutObserver.observe(element);
  }
}
tbody.addEventListener('input', (event) => {
  recordEditorInputChange(event.target);
  markEditorDirty();
  if (event.target.classList.contains('text')) {
    const tr = event.target.closest('tr');
    resizeSegmentTextarea(event.target, tr && tr.classList.contains('active'));
  }
  if (event.target.classList.contains('start') || event.target.classList.contains('end')) syncActiveSegment();
  else {
    if (event.target.classList.contains('speaker')) {
      const segments = collectSegments();
      renderSpeakerMap(segments);
      applySpeakerRowColors(segments);
      refreshSegmentActions();
      finalizeEditorMutation();
    }
    updateSubtitlePreview();
  }
});
tbody.addEventListener('change', markEditorDirty);
tbody.addEventListener('click', (event) => {
  const addButton = event.target.closest('[data-add-after]');
  if (addButton) {
    const tr = addButton.closest('tr');
    if (tr) addSegmentAfter(tr);
    return;
  }
  const mergeButton = event.target.closest('[data-merge-next]');
  if (mergeButton) {
    const tr = mergeButton.closest('tr');
    if (tr) mergeSegmentWithNext(tr);
    return;
  }
  const deleteButton = event.target.closest('[data-delete-segment]');
  if (!deleteButton) return;
  const tr = deleteButton.closest('tr');
  if (tr) deleteSegmentRow(tr);
});
speakerMapEl.addEventListener('input', () => {
  recordEditorInputChange(document.activeElement);
  syncSpeakerNameInputs();
  markEditorDirty();
  updateSubtitlePreview();
});
tbody.addEventListener('focusin', (event) => {
  const tr = event.target.closest('tr');
  if (!tr) return;
  setActiveSegment(Number(tr.dataset.index), false);
  resizeSegmentRow(tr, true);
  updateSubtitlePreview();
});
for (const id of ['fontSize', 'marginV', 'showSpeaker', 'speakerColors']) {
  document.querySelector('#' + id).addEventListener('input', () => {
    recordEditorInputChange(document.querySelector('#' + id));
    markEditorDirty();
    updateSubtitlePreview();
  });
  document.querySelector('#' + id).addEventListener('change', () => {
    recordEditorInputChange(document.querySelector('#' + id));
    markEditorDirty();
    updateSubtitlePreview();
  });
}

async function refreshJobs(options = {}) {
  const res = await fetch(apiUrl('api/jobs'), { cache: 'no-store' });
  if (!res.ok) return;
  const data = await res.json();
  jobs = data.jobs || [];
  renderJobList();
  if (currentJob) {
    const fresh = jobs.find((job) => job.id === currentJob.id);
    if (fresh) {
      const wasEditable = EDIT_STATES.has(currentJob.status);
      currentJob = fresh;
      if (options.background && wasEditable && EDIT_STATES.has(fresh.status)) {
        updateEditorChrome(fresh);
      } else {
        renderCurrentJob(fresh, { skipSegments: options.skipSegments || editorDirty });
      }
    } else {
      currentJob = null;
      showImportView();
    }
  } else if (!options.keepSelection && jobs.length && options.selectLatest) {
    await selectJob(jobs[0].id);
  }
  ensurePolling();
}

function renderJobList() {
  jobCountEl.textContent = jobs.length + ' 个任务';
  if (!jobs.length) {
    jobListEl.innerHTML = '<div class="meta" style="padding:10px">还没有任务</div>';
    return;
  }
  jobListEl.innerHTML = jobs.map((job) => {
    const active = currentJob && currentJob.id === job.id ? ' active' : '';
    const canDelete = !RUNNING_STATES.has(job.status);
    const percent = Math.round((job.progress || 0) * 100);
    const warning = truncationWarning(job);
    return `
      <div class="task-item${active}" data-job-id="${escapeHtml(job.id)}">
        <div class="task-row">
          <div class="task-name">${escapeHtml(job.media_name || 'input.media')}</div>
          <span class="${statusClass(job.status)}">${statusLabel(job.status)}</span>
        </div>
        <div class="task-id meta">${escapeHtml(job.id)}</div>
        <div class="meta">${escapeHtml(tokenUsageSummary(job))}</div>
        ${warning ? `<div class="warning">${escapeHtml(warning)}</div>` : ''}
        <div class="task-foot">
          <div class="progress task-progress"><div class="bar" style="width:${percent}%"></div></div>
          ${canDelete ? `<button class="small ghost" data-delete-id="${escapeHtml(job.id)}">删除</button>` : ''}
        </div>
      </div>`;
  }).join('');
}

async function selectJob(jobId) {
  const local = jobs.find((job) => job.id === jobId);
  currentJob = local || currentJob;
  renderJobList();
  const res = await fetch(apiUrl(`api/jobs/${jobId}`), { cache: 'no-store' });
  if (!res.ok) {
    await refreshJobs();
    return;
  }
  currentJob = await res.json();
  renderCurrentJob(currentJob);
}

function renderCurrentJob(job, options = {}) {
  renderJobList();
  if (EDIT_STATES.has(job.status)) showEditor(job, options);
  else showProcessing(job);
}

function showImportView(options = {}) {
  if (options.clearDraft !== false) resetImportMode();
  currentJob = null;
  clearEditorHistory();
  setEditorDirty(false);
  fileInput.value = '';
  if (!options.preserveError) importErrorEl.textContent = '';
  setVisible(importView);
  renderJobList();
}

function resetImportMode() {
  rerunDraftJob = null;
  importTitleEl.textContent = '导入媒体';
  rerunSourceEl.textContent = '';
  fileInput.disabled = false;
  uploadBtn.textContent = '开始转写';
}

function showProcessingPlaceholder(name) {
  currentJob = null;
  processTitleEl.textContent = '创建任务';
  processNameEl.textContent = name;
  processMetaEl.textContent = '上传媒体并准备转写';
  processBarEl.style.width = '2%';
  processErrorEl.textContent = '';
  setVisible(processingView);
}

function showProcessing(job) {
  processTitleEl.textContent = job.status === 'failed' ? '任务失败' : '转写中';
  processNameEl.textContent = job.media_name || 'input.media';
  processMetaEl.textContent = jobSummary(job);
  processBarEl.style.width = `${Math.round((job.progress || 0) * 100)}%`;
  processErrorEl.textContent = job.error || truncationWarning(job);
  deleteCurrentBtn.disabled = RUNNING_STATES.has(job.status);
  setVisible(processingView);
}

async function showEditor(job, options = {}) {
  applySubtitleStyle(job.subtitle_style || {});
  updateEditorChrome(job);
  setVisible(workbench);
  const mediaUrl = apiUrl(`api/jobs/${job.id}/media`);
  if (preview.dataset.jobId !== job.id) {
    preview.dataset.jobId = job.id;
    preview.src = mediaUrl;
    resetVideoStage();
  }
  renderDownloads(job.status);
  if (!options.skipSegments) await loadSegments(job.id);
  fitVideoStageToMedia();
}

function updateEditorChrome(job) {
  selectedNameEl.textContent = job.media_name || 'input.media';
  taskStatusEl.textContent = statusLabel(job.status);
  taskStatusEl.className = statusClass(job.status);
  taskUsageEl.textContent = tokenUsageSummary(job);
  taskParamsEl.textContent = parameterSummary(job);
  if (job.error) setTaskNotice(job.error, 'error');
  else if (truncationWarning(job)) setTaskNotice('可能截断，建议提高输出 tokens 后重新转写。', 'warning');
  else setTaskNotice('', '');
  renderBtn.disabled = !ffmpegAvailable || job.status === 'rendering';
  renderBtn.textContent = job.status === 'rendering' ? '烧录中...' : ffmpegAvailable ? '烧录视频' : 'FFmpeg 不可用';
  updateRerunAction(job);
  setSaveState(editorDirty ? 'dirty' : 'saved', editorDirty ? '有未保存修改' : '已保存');
  renderDownloads(job.status);
}

function setTaskNotice(message, kind) {
  taskNoticeEl.textContent = message || '';
  taskNoticeEl.className = 'task-notice ' + (kind || '');
  taskNoticeEl.classList.toggle('is-hidden', !message);
}

function updateRerunAction(job) {
  rerunBtn.disabled = RUNNING_STATES.has(job.status);
  rerunBtn.textContent = '重新转写';
}

function showRerunDraft(job) {
  const usage = job.usage || {};
  const inference = job.inference || {};
  const currentMax = Number(usage.max_new_tokens || inference.max_new_tokens || 0);
  rerunDraftJob = job;
  currentJob = null;
  importTitleEl.textContent = '重跑转写';
  fileInput.value = '';
  fileInput.disabled = true;
  rerunSourceEl.textContent = '来源媒体：' + (job.media_name || 'input.media');
  promptInput.value = inference.prompt || '';
  maxNewTokensInput.value = usage.possibly_truncated && currentMax > 0
    ? Math.max(currentMax * 2, currentMax + 512)
    : currentMax || '';
  maxLenInput.value = inference.max_length || '';
  decodingSelect.value = inference.decoding || 'greedy';
  temperatureInput.value = inference.temperature == null ? '1.0' : inference.temperature;
  updateDecodingControls();
  advancedDetails.open = true;
  uploadBtn.textContent = '开始重跑';
  importErrorEl.textContent = '';
  setVisible(importView);
  renderJobList();
}

async function startRerunDraft() {
  if (!rerunDraftJob) return;
  const source = rerunDraftJob;
  const payload = {
    prompt: promptInput.value,
    max_new_tokens: Number(maxNewTokensInput.value || 0),
    max_len: Number(maxLenInput.value || 0),
    decoding: decodingSelect.value,
  };
  if (temperatureInput.value) payload.temperature = Number(temperatureInput.value);
  uploadBtn.disabled = true;
  advancedDetails.open = false;
  importErrorEl.textContent = '';
  showProcessingPlaceholder(source.media_name || 'input.media');
  const res = await fetch(apiUrl(`api/jobs/${source.id}/rerun`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  uploadBtn.disabled = false;
  if (!res.ok) {
    importErrorEl.textContent = data.detail || '重跑失败';
    showImportView({ clearDraft: false, preserveError: true });
    return;
  }
  resetImportMode();
  currentJob = data;
  await refreshJobs({ keepSelection: true });
  await selectJob(data.id);
}

function setVisible(view) {
  importView.classList.toggle('is-hidden', view !== importView);
  processingView.classList.toggle('is-hidden', view !== processingView);
  workbench.classList.toggle('is-hidden', view !== workbench);
}

async function deleteJob(jobId) {
  const job = jobs.find((item) => item.id === jobId);
  if (job && RUNNING_STATES.has(job.status)) return;
  const res = await fetch(apiUrl(`api/jobs/${jobId}`), { method: 'DELETE' });
  if (!res.ok) return;
  if (currentJob && currentJob.id === jobId) {
    currentJob = null;
    preview.removeAttribute('src');
    preview.removeAttribute('data-job-id');
    preview.load();
    tbody.innerHTML = '';
    downloads.innerHTML = '';
    clearEditorHistory();
    setEditorDirty(false);
    showImportView();
  }
  await refreshJobs({ keepSelection: true });
}

async function saveSegments() {
  if (!currentJob) return false;
  if (!editorDirty) return true;
  setSaveState('saving', '正在保存...');
  const segments = collectSegments();
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/segments`), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segments, style: collectSubtitleStyle() })
    });
    const data = await res.json();
    if (!res.ok) {
      setTaskNotice(data.detail || '保存失败', 'error');
      setSaveState('error', data.detail || '保存失败');
      saveBtn.disabled = false;
      return false;
    }
    setTaskNotice('', '');
    renderSegments(data.segments);
    resetEditorHistory();
    setEditorDirty(false);
    saveStatusEl.textContent = '已保存';
    saveStatusTimer = setTimeout(() => {
      if (!editorDirty) saveStatusEl.textContent = '已保存';
    }, 1200);
    await selectJob(currentJob.id);
    return true;
  } catch (err) {
    setTaskNotice('保存失败：' + err.message, 'error');
    setSaveState('error', '保存失败');
    saveBtn.disabled = false;
    return false;
  }
}

async function loadSegments(jobId) {
  const res = await fetch(apiUrl(`api/jobs/${jobId}/segments`));
  const data = await res.json();
  renderSegments(data.segments || []);
  resetEditorHistory();
  setEditorDirty(false);
}

function ensurePolling() {
  const shouldPoll = jobs.some((job) => RUNNING_STATES.has(job.status));
  if (shouldPoll && !pollTimer) pollTimer = setInterval(() => refreshJobs({ keepSelection: true, background: true }), 1500);
  if (!shouldPoll && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function collectSubtitleStyle() {
  return {
    font_size: Number(document.querySelector('#fontSize').value || 48),
    margin_v: Number(document.querySelector('#marginV').value || 56),
    show_speaker: document.querySelector('#showSpeaker').value === 'true',
    speaker_colors: document.querySelector('#speakerColors').value === 'true',
    speaker_names: collectSpeakerNames()
  };
}

function applySubtitleStyle(style) {
  if (!style || editorDirty) return;
  if (style.font_size != null) document.querySelector('#fontSize').value = style.font_size;
  if (style.margin_v != null) document.querySelector('#marginV').value = style.margin_v;
  if (style.show_speaker != null) document.querySelector('#showSpeaker').value = String(!!style.show_speaker);
  if (style.speaker_colors != null) document.querySelector('#speakerColors').value = String(!!style.speaker_colors);
  speakerNameMap = {};
  speakerMapEl.innerHTML = '';
  const names = style.speaker_names || {};
  for (const [speaker, name] of Object.entries(names)) {
    if (String(name).trim()) speakerNameMap[String(speaker)] = String(name).trim();
  }
}

function collectSpeakerNames() {
  const names = {};
  for (const input of speakerMapEl.querySelectorAll('input[data-speaker]')) {
    const speaker = input.dataset.speaker || '';
    const name = input.value.trim();
    if (speaker && name) names[speaker] = name;
  }
  return names;
}

function syncSpeakerNameInputs() {
  for (const input of speakerMapEl.querySelectorAll('input[data-speaker]')) {
    const speaker = input.dataset.speaker || '';
    if (!speaker) continue;
    const name = input.value.trim();
    if (name) speakerNameMap[speaker] = name;
    else delete speakerNameMap[speaker];
  }
}

function renderSpeakerMap(segments) {
  syncSpeakerNameInputs();
  const speakers = [...new Set(segments.map((segment) => segment.speaker).filter(Boolean))].sort();
  if (!speakers.length) {
    speakerMapEl.innerHTML = '<div class="meta">暂无说话人</div>';
    return;
  }
  speakerMapEl.innerHTML = speakers.map((speaker) => {
    const name = speakerNameMap[speaker] || '';
    return `
      <div class="speaker-map-row">
        <div class="speaker-tag">${escapeHtml(speaker)}</div>
        <input type="text" data-speaker="${escapeHtml(speaker)}" value="${escapeHtml(name)}" placeholder="显示名称">
      </div>`;
  }).join('');
}

function speakerDisplayName(speaker) {
  const names = collectSpeakerNames();
  return names[speaker] || speakerNameMap[speaker] || speaker;
}

function renderSegments(segments) {
  tbody.innerHTML = '';
  activeSegmentIndex = -1;
  for (const [index, segment] of segments.entries()) {
    const tr = document.createElement('tr');
    tr.dataset.id = segment.id;
    tr.dataset.index = String(index);
    tr.innerHTML = `
      <td><input class="start" type="number" min="0" step="0.01" value="${segment.start}"></td>
      <td><input class="end" type="number" min="0" step="0.01" value="${segment.end}"></td>
      <td><input class="speaker" type="text" value="${escapeHtml(segment.speaker)}"></td>
      <td><textarea class="text" rows="1">${escapeHtml(segment.text)}</textarea></td>
      <td class="action">
        <div class="segment-actions">
          <button type="button" class="segment-action segment-merge" data-merge-next>合并下行</button>
          <button type="button" class="segment-action segment-add" data-add-after>下方新增</button>
          <button type="button" class="segment-action segment-delete" data-delete-segment>删除</button>
        </div>
      </td>
    `;
    tr.addEventListener('click', (event) => {
      if (event.target.closest('input, textarea, button')) return;
      const rowIndex = Number(tr.dataset.index);
      const start = Number(tr.querySelector('.start').value);
      if (Number.isFinite(start)) preview.currentTime = Math.max(0, start);
      setActiveSegment(rowIndex, false);
      updateSubtitlePreview();
    });
    tbody.appendChild(tr);
    resizeSegmentRow(tr, false);
  }
  refreshSegmentActions();
  applySpeakerRowColors(segments);
  renderSpeakerMap(segments);
  syncActiveSegment();
}

function addSegmentAfter(tr) {
  const rows = [...tbody.querySelectorAll('tr')];
  const index = rows.indexOf(tr);
  if (index < 0) return;
  checkpointEditorState();
  const segments = collectSegments();
  const current = segments[index];
  const next = segments[index + 1];
  const start = Number.isFinite(Number(current.end)) ? Number(current.end) : 0;
  const nextStart = next && Number(next.start);
  const end = Number.isFinite(nextStart) && nextStart > start ? nextStart : start + 2;
  manualSegmentCounter += 1;
  segments.splice(index + 1, 0, {
    id: `seg_manual_${Date.now()}_${manualSegmentCounter}`,
    start: Number(formatTimestampValue(start)),
    end: Number(formatTimestampValue(end)),
    speaker: current.speaker || '',
    text: ''
  });
  renderSegments(segments);
  finalizeEditorMutation();
  markEditorDirty();
  setSaveState('dirty', `已在第 ${index + 1} 行下方新增字幕，待保存`);
  const newRow = tbody.querySelector(`tr[data-index="${index + 1}"]`);
  if (newRow) {
    setActiveSegment(index + 1, true);
    const textInput = newRow.querySelector('.text');
    if (textInput) textInput.focus();
  }
}

function deleteSegmentRow(tr) {
  checkpointEditorState();
  tr.remove();
  refreshSegmentActions();
  activeSegmentIndex = -1;
  const segments = collectSegments();
  renderSpeakerMap(segments);
  applySpeakerRowColors(segments);
  markEditorDirty();
  finalizeEditorMutation();
  syncActiveSegment();
}

function mergeSegmentWithNext(tr) {
  const rows = [...tbody.querySelectorAll('tr')];
  const index = rows.indexOf(tr);
  const next = rows[index + 1];
  if (index < 0 || !next) return;
  const speaker = tr.querySelector('.speaker').value.trim();
  const nextSpeaker = next.querySelector('.speaker').value.trim();
  if (speaker !== nextSpeaker) return;
  checkpointEditorState();
  const endInput = tr.querySelector('.end');
  const nextEnd = Number(next.querySelector('.end').value);
  const currentEnd = Number(endInput.value);
  if (Number.isFinite(nextEnd)) {
    endInput.value = formatTimestampValue(Number.isFinite(currentEnd) ? Math.max(currentEnd, nextEnd) : nextEnd);
  }
  const textInput = tr.querySelector('.text');
  textInput.value = joinSegmentText(textInput.value, next.querySelector('.text').value);
  next.remove();
  refreshSegmentActions();
  resizeSegmentRow(tr, true);
  activeSegmentIndex = -1;
  const segments = collectSegments();
  renderSpeakerMap(segments);
  applySpeakerRowColors(segments);
  markEditorDirty();
  finalizeEditorMutation();
  setSaveState('dirty', `已合并第 ${index + 1}、${index + 2} 行，待保存`);
  syncActiveSegment();
}

function joinSegmentText(firstText, secondText) {
  const first = String(firstText || '').trimEnd();
  const second = String(secondText || '').trimStart();
  if (!first) return second;
  if (!second) return first;
  const needsSpace = /[A-Za-z0-9]$/.test(first) && /^[A-Za-z0-9]/.test(second);
  return first + (needsSpace ? ' ' : '') + second;
}

function refreshSegmentActions() {
  const rows = [...tbody.querySelectorAll('tr')];
  rows.forEach((row, index) => {
    row.dataset.index = String(index);
    const deleteButton = row.querySelector('[data-delete-segment]');
    if (deleteButton) {
      deleteButton.setAttribute('aria-label', `删除第 ${index + 1} 行字幕`);
      deleteButton.title = '删除这一行';
    }
    const addButton = row.querySelector('[data-add-after]');
    if (addButton) {
      addButton.setAttribute('aria-label', `在第 ${index + 1} 行下方新增字幕`);
      addButton.title = '在这一行下方新增字幕';
    }
    const mergeButton = row.querySelector('[data-merge-next]');
    if (!mergeButton) return;
    const next = rows[index + 1];
    mergeButton.classList.toggle('is-hidden', !next);
    if (!next) return;
    const speaker = row.querySelector('.speaker').value.trim();
    const nextSpeaker = next.querySelector('.speaker').value.trim();
    const sameSpeaker = speaker === nextSpeaker;
    mergeButton.disabled = !sameSpeaker;
    mergeButton.setAttribute('aria-label', `合并第 ${index + 1} 行和第 ${index + 2} 行字幕`);
    mergeButton.title = sameSpeaker ? '与下一行合并' : '下一行说话人不同，请先统一说话人';
  });
}

function applyTimelineOffset(requestedOffset) {
  const offset = Number.isFinite(requestedOffset) ? requestedOffset : Number(timelineOffsetInput.value);
  const rows = [...tbody.querySelectorAll('tr')];
  if (!Number.isFinite(offset) || offset === 0 || !rows.length) return;
  checkpointEditorState();
  for (const row of rows) {
    const startInput = row.querySelector('.start');
    const endInput = row.querySelector('.end');
    const start = Number(startInput.value);
    const end = Number(endInput.value);
    if (Number.isFinite(start)) startInput.value = formatTimestampValue(Math.max(0, start + offset));
    if (Number.isFinite(end)) endInput.value = formatTimestampValue(Math.max(0, end + offset));
  }
  timelineOffsetInput.value = '0';
  finalizeEditorMutation();
  markEditorDirty();
  const direction = offset < 0 ? '提前' : '延后';
  setSaveState('dirty', `已整体${direction} ${Math.abs(offset).toFixed(2)} 秒，待保存`);
  syncActiveSegment();
}

function formatTimestampValue(value) {
  return String(Math.round(value * 1000) / 1000);
}

function resizeSegmentTextarea(textarea, expanded) {
  if (!textarea) return;
  const maxHeight = expanded ? 112 : 48;
  textarea.style.height = 'auto';
  const naturalHeight = textarea.scrollHeight;
  const nextHeight = Math.max(30, Math.min(naturalHeight, maxHeight));
  textarea.style.height = nextHeight + 'px';
  textarea.style.overflowY = naturalHeight > maxHeight ? 'auto' : 'hidden';
}

function resizeSegmentRow(tr, expanded) {
  resizeSegmentTextarea(tr && tr.querySelector('textarea.text'), expanded);
}

function collectSegments() {
  return [...tbody.querySelectorAll('tr')].map((tr, index) => ({
    id: tr.dataset.id || `seg_${String(index + 1).padStart(4, '0')}`,
    start: Number(tr.querySelector('.start').value),
    end: Number(tr.querySelector('.end').value),
    speaker: tr.querySelector('.speaker').value,
    text: tr.querySelector('.text').value
  }));
}

function resetVideoStage() {
  assPlayRes = { x: 1920, y: 1080 };
  videoStage.style.width = '';
  videoStage.style.height = '';
  videoStage.style.aspectRatio = assPlayRes.x + ' / ' + assPlayRes.y;
}

function fitVideoStageToMedia() {
  const videoWidth = Number(preview.videoWidth || 0);
  const videoHeight = Number(preview.videoHeight || 0);
  const shell = videoStage.parentElement;
  if (!shell || videoWidth <= 0 || videoHeight <= 0) {
    resetVideoStage();
    updateSubtitlePreview();
    return;
  }
  assPlayRes = { x: videoWidth, y: videoHeight };
  const maxWidth = shell.clientWidth || videoWidth;
  const maxHeight = Math.max(180, Math.floor(window.innerHeight * 0.48));
  const scale = Math.min(maxWidth / videoWidth, maxHeight / videoHeight);
  videoStage.style.width = Math.max(1, Math.floor(videoWidth * scale)) + 'px';
  videoStage.style.height = Math.max(1, Math.floor(videoHeight * scale)) + 'px';
  videoStage.style.aspectRatio = videoWidth + ' / ' + videoHeight;
  updateSubtitlePreview();
}

function assScriptScale() {
  const playResY = Number(assPlayRes.y || preview.videoHeight || 0);
  if (playResY <= 0) return 1;
  return (videoStage.clientHeight || playResY) / playResY;
}

function syncActiveSegment() {
  const time = Number(preview.currentTime || 0);
  const segments = collectSegments();
  const index = segments.findIndex((segment) => {
    const start = Number(segment.start);
    const end = Number(segment.end);
    return Number.isFinite(start) && Number.isFinite(end) && start <= time && time <= end;
  });
  setActiveSegment(index, true);
  updateSubtitlePreview(segments);
}

function startPlaybackSync() {
  stopPlaybackSync();
  lastPlaybackSyncAt = 0;
  const tick = (now) => {
    if (preview.paused || preview.ended) {
      playbackSyncHandle = 0;
      return;
    }
    if (!lastPlaybackSyncAt || now - lastPlaybackSyncAt >= 50) {
      lastPlaybackSyncAt = now;
      syncActiveSegment();
    }
    if (typeof preview.requestVideoFrameCallback === 'function') {
      playbackSyncHandle = preview.requestVideoFrameCallback(tick);
    } else {
      playbackSyncHandle = requestAnimationFrame(tick);
    }
  };
  if (typeof preview.requestVideoFrameCallback === 'function') {
    playbackSyncHandle = preview.requestVideoFrameCallback(tick);
  } else {
    playbackSyncHandle = requestAnimationFrame(tick);
  }
}

function stopPlaybackSync() {
  if (!playbackSyncHandle) return;
  if (typeof preview.cancelVideoFrameCallback === 'function') {
    preview.cancelVideoFrameCallback(playbackSyncHandle);
  } else {
    cancelAnimationFrame(playbackSyncHandle);
  }
  playbackSyncHandle = 0;
}

function setActiveSegment(index, shouldScroll) {
  if (index === activeSegmentIndex) return;
  activeSegmentIndex = index;
  for (const tr of tbody.querySelectorAll('tr')) {
    const active = Number(tr.dataset.index) === index;
    tr.classList.toggle('active', active);
    resizeSegmentRow(tr, active);
    if (active && shouldScroll) scrollSegmentRowIntoView(tr);
  }
}

function scrollSegmentRowIntoView(tr) {
  const container = tr.closest('.table-wrap');
  if (!container) return;
  const stickyHeaderHeight = 30;
  const rowTop = tr.offsetTop;
  const rowBottom = rowTop + tr.offsetHeight;
  const viewTop = container.scrollTop + stickyHeaderHeight;
  const viewBottom = container.scrollTop + container.clientHeight;
  if (rowTop < viewTop) {
    container.scrollTop = Math.max(0, rowTop - stickyHeaderHeight - 4);
  } else if (rowBottom > viewBottom) {
    container.scrollTop = rowBottom - container.clientHeight + 8;
  }
}

function updateSubtitlePreview(segments) {
  segments = segments || collectSegments();
  const segment = segments[activeSegmentIndex];
  if (!segment || !segment.text || activeSegmentIndex < 0) {
    subtitleOverlay.classList.remove('visible');
    subtitleOverlay.textContent = '';
    return;
  }
  const showSpeaker = document.querySelector('#showSpeaker').value === 'true';
  const useSpeakerColors = document.querySelector('#speakerColors').value === 'true';
  const fontSize = Math.max(12, Number(document.querySelector('#fontSize').value || 48));
  const marginV = Math.max(0, Number(document.querySelector('#marginV').value || 56));
  const text = showSpeaker && segment.speaker ? speakerDisplayName(segment.speaker) + ': ' + segment.text : segment.text;
  const scale = assScriptScale();
  subtitleOverlay.textContent = text;
  subtitleOverlay.style.fontSize = Math.max(22, fontSize * scale) + 'px';
  subtitleOverlay.style.lineHeight = String(assFontLineHeightFactor);
  subtitleOverlay.style.bottom = Math.max(marginV * scale, nativeVideoControlsSafeBottom()) + 'px';
  subtitleOverlay.style.webkitTextStroke = subtitleTextStroke(scale);
  subtitleOverlay.style.textShadow = subtitleTextShadow(scale);
  const color = useSpeakerColors ? speakerColor(segment.speaker, segments) : '#ffffff';
  subtitleOverlay.style.color = color;
  subtitleOverlay.style.webkitTextFillColor = color;
  subtitleOverlay.classList.add('visible');
}

function nativeVideoControlsSafeBottom() {
  const stageHeight = Number(videoStage.clientHeight || 0);
  if (stageHeight <= 0) return 80;
  return Math.min(140, Math.max(64, stageHeight * 0.18));
}

function subtitleTextStroke(scale) {
  return Math.max(1, 3 * scale) + 'px #000';
}

function subtitleTextShadow(scale) {
  const shadow = Math.max(0.5, 1 * scale);
  const blur = Math.max(1, 3 * scale);
  return `0 ${shadow}px ${blur}px rgba(0, 0, 0, 0.65)`;
}

function speakerColor(speaker, segments) {
  const speakers = [];
  for (const segment of segments) {
    if (segment.speaker && !speakers.includes(segment.speaker)) speakers.push(segment.speaker);
  }
  speakers.sort();
  const index = Math.max(0, speakers.indexOf(speaker || ''));
  return speakerPalette[index % speakerPalette.length];
}

function applySpeakerRowColors(segments) {
  const speakers = [...new Set(segments.map((segment) => segment.speaker).filter(Boolean))].sort();
  const rows = [...tbody.querySelectorAll('tr')];
  rows.forEach((row, index) => {
    const speaker = segments[index] && segments[index].speaker || '';
    const speakerIndex = Math.max(0, speakers.indexOf(speaker));
    row.style.setProperty('--speaker-row-bg', speakerRowPalette[speakerIndex % speakerRowPalette.length]);
    const accent = speakerPalette[speakerIndex % speakerPalette.length];
    row.style.setProperty('--speaker-row-accent', accent === '#ffffff' ? '#aeb7bf' : accent);
  });
}

function renderDownloads(status) {
  if (!currentJob) return;
  const links = [
    ['json', 'JSON'],
    ['srt', 'SRT'],
    ['ass', 'ASS'],
    ['transcript', '原文']
  ];
  if (status === 'done') links.push(['mp4', 'MP4']);
  downloads.innerHTML = links.map(([kind, label]) =>
    `<a href="${apiUrl(`api/jobs/${currentJob.id}/download?kind=${kind}`)}" target="_blank">${label}</a>`
  ).join('');
}

function jobSummary(job) {
  const inference = job.inference || {};
  const temp = inference.temperature ? (' · temp ' + inference.temperature) : '';
  return tokenUsageSummary(job) + ' · max_len ' + inference.max_length + ' · ' + inference.decoding + temp;
}

function parameterSummary(job) {
  const inference = job.inference || {};
  const temp = inference.temperature ? (' · temp ' + inference.temperature) : '';
  return 'max_len ' + inference.max_length + ' · ' + inference.decoding + temp;
}

function tokenUsageSummary(job) {
  const usage = job.usage || {};
  const inference = job.inference || {};
  const maxNewTokens = usage.max_new_tokens || inference.max_new_tokens || 0;
  if (usage.generated_tokens == null) return '生成 tokens ' + maxNewTokens;
  const prompt = usage.prompt_tokens == null ? '' : (' · prompt ' + usage.prompt_tokens);
  return '生成 ' + usage.generated_tokens + '/' + maxNewTokens + ' tokens' + prompt;
}

function truncationWarning(job) {
  const usage = job.usage || {};
  if (!usage.possibly_truncated) return '';
  return '可能截断：生成 token 已达到上限，请检查字幕末尾或提高输出 tokens 后重跑。';
}

function statusClass(status) {
  return 'pill ' + (status === 'failed' ? 'bad' : status === 'done' ? 'ok' : '');
}

function statusLabel(status) {
  const labels = {
    queued: '排队中',
    loading_model: '加载模型',
    transcribing: '转写中',
    postprocessing: '处理中',
    waiting_review: '待校对',
    rendering: '烧录中',
    done: '已完成',
    failed: '失败',
    cancelled: '已取消',
    idle: '空闲'
  };
  return labels[status] || status;
}

function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

restoreSidebarState();
refreshRuntime();
refreshJobs({ selectLatest: true });
</script>
</body>
</html>
"""

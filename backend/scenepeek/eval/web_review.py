"""Simple web UI for reviewing eval/real candidates."""

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from scenepeek.eval.real import CANDIDATES_PATH, REPO_ROOT, SOURCES_PATH, load_sources
from scenepeek.eval.review import (
    _add_to_dataset,
    _load_candidates,
    _load_dataset,
    _save_candidates,
    _save_dataset,
)

app = FastAPI()

VIDEOS_PATH = REPO_ROOT / "eval/videos/real"


class EditRequest(BaseModel):
    text: str | None = None
    start_s: float | None = None
    end_s: float | None = None


DATASET_PATH = REPO_ROOT / "eval/real/dataset.yaml"


@app.get("/api/candidates/pending")
def get_pending():
    """Return next pending candidate with video file info."""
    cand_data = _load_candidates(CANDIDATES_PATH)
    sources = {v["key"]: v for v in load_sources(SOURCES_PATH)}
    pending = [c for c in cand_data["candidates"] if c.get("status", "pending") == "pending"]

    if not pending:
        return {"total": 0, "remaining": 0, "candidate": None}

    c = pending[0]
    idx = next(i for i, x in enumerate(cand_data["candidates"]) if x["id"] == c["id"])

    video_info = sources.get(c["video"], {})
    video_file = video_info.get("file")

    return {
        "total": len(cand_data["candidates"]),
        "remaining": len(pending),
        "index": idx + 1,
        "candidate": c,
        "video_url": f"/api/video/{c['video']}" if video_file else None,
        "video_title": video_info.get("title", c["video"]),
    }


@app.post("/api/candidates/{cand_id}/approve")
def approve(cand_id: str):
    """Approve a candidate and add to dataset."""
    cand_data = _load_candidates(CANDIDATES_PATH)
    sources = {v["key"]: v for v in load_sources(SOURCES_PATH)}
    dataset = _load_dataset(DATASET_PATH)

    c = next((x for x in cand_data["candidates"] if x["id"] == cand_id), None)
    if not c:
        return {"error": "not found"}, 404

    c["status"] = "approved"
    _add_to_dataset(dataset, sources, c)
    _save_candidates(CANDIDATES_PATH, cand_data)
    _save_dataset(DATASET_PATH, dataset)

    return {"status": "approved"}


@app.post("/api/candidates/{cand_id}/reject")
def reject(cand_id: str):
    """Reject a candidate."""
    cand_data = _load_candidates(CANDIDATES_PATH)
    c = next((x for x in cand_data["candidates"] if x["id"] == cand_id), None)
    if not c:
        return {"error": "not found"}, 404

    c["status"] = "rejected"
    _save_candidates(CANDIDATES_PATH, cand_data)

    return {"status": "rejected"}


@app.patch("/api/candidates/{cand_id}")
def update(cand_id: str, data: EditRequest):
    """Update candidate text and/or times."""
    cand_data = _load_candidates(CANDIDATES_PATH)
    c = next((x for x in cand_data["candidates"] if x["id"] == cand_id), None)
    if not c:
        return {"error": "not found"}, 404

    if data.text is not None:
        c["text"] = data.text
    if data.start_s is not None:
        c["start_s"] = round(data.start_s, 2)
    if data.end_s is not None:
        c["end_s"] = round(data.end_s, 2)

    _save_candidates(CANDIDATES_PATH, cand_data)

    return c


@app.get("/api/video/{video_key}")
def get_video(video_key: str):
    """Serve video file for a candidate."""
    sources = {v["key"]: v for v in load_sources(SOURCES_PATH)}
    video_info = sources.get(video_key)

    if not video_info:
        return {"error": "video not found"}, 404

    video_path = REPO_ROOT / video_info["file"]
    if not video_path.exists():
        return {"error": "video file not found"}, 404

    return FileResponse(video_path, media_type="video/mp4")


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the review UI."""
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ScenePeek Candidate Review</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .progress {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        .progress-bar {
            height: 4px;
            background: #e0e0e0;
            border-radius: 2px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: #4CAF50;
            transition: width 0.3s;
        }
        .candidate {
            margin: 30px 0;
            padding: 20px;
            background: #f9f9f9;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            background: #2196F3;
            color: white;
            border-radius: 3px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
        }
        .badge.speech { background: #FF9800; }
        .badge.ocr { background: #9C27B0; }
        .badge.visual { background: #E91E63; }
        .badge.multi { background: #00BCD4; }
        .badge.semantic { background: #4CAF50; }

        .meta {
            font-size: 14px;
            color: #666;
            margin: 10px 0;
        }
        .query {
            margin: 20px 0;
            padding: 15px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        .context {
            font-size: 13px;
            color: #999;
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 3px;
            border-left: 2px solid #ddd;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        button {
            padding: 10px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-approve {
            background: #4CAF50;
            color: white;
        }
        .btn-approve:hover { background: #45a049; }

        .btn-reject {
            background: #f44336;
            color: white;
        }
        .btn-reject:hover { background: #da190b; }

        .btn-edit {
            background: #2196F3;
            color: white;
        }
        .btn-edit:hover { background: #0b7dda; }

        .btn-skip {
            background: #999;
            color: white;
        }
        .btn-skip:hover { background: #777; }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal.show { display: flex; }

        .modal-content {
            background: white;
            padding: 30px;
            border-radius: 8px;
            max-width: 500px;
            width: 90%;
        }
        .modal-content h3 { margin-top: 0; }
        .form-group {
            margin: 15px 0;
        }
        .form-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .form-group input,
        .form-group textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: inherit;
            font-size: 14px;
        }
        .form-group textarea {
            min-height: 100px;
            resize: vertical;
        }
        .modal-controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            justify-content: flex-end;
        }
        .empty {
            text-align: center;
            padding: 40px;
            color: #999;
        }
        .loading {
            text-align: center;
            padding: 20px;
        }
        .video-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 2000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .video-overlay.show {
            display: flex;
        }
        .video-player-wrapper {
            width: 90%;
            max-width: 1000px;
            aspect-ratio: 16/9;
            background: black;
            border-radius: 8px;
            overflow: hidden;
        }
        .video-player-wrapper video {
            width: 100%;
            height: 100%;
        }
        .video-controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .video-controls button {
            background: #2196F3;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .video-controls button:hover {
            background: #0b7dda;
        }
        .video-time {
            color: white;
            font-size: 14px;
        }
        .btn-play-video {
            background: #2196F3;
            color: white;
            margin-top: 20px;
        }
        .btn-play-video:hover {
            background: #0b7dda;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ScenePeek Candidate Review</h1>
            <div class="progress">
                <span id="progress-text">Loading...</span>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
            </div>
        </div>

        <div id="content"></div>
    </div>

    <div class="video-overlay" id="video-overlay">
        <div class="video-player-wrapper">
            <video id="video-player" controls>
                Your browser does not support the video tag.
            </video>
        </div>
        <div class="video-controls">
            <span class="video-time" id="video-time"></span>
            <button onclick="closeVideo()">Close Video</button>
        </div>
    </div>

    <div class="modal" id="edit-modal">
        <div class="modal-content">
            <h3>Edit Candidate</h3>
            <div class="form-group">
                <label>Query Text</label>
                <textarea id="edit-text"></textarea>
            </div>
            <div style="display: flex; gap: 20px;">
                <div class="form-group" style="flex: 1;">
                    <label>Start (s)</label>
                    <input type="number" id="edit-start" step="0.1">
                </div>
                <div class="form-group" style="flex: 1;">
                    <label>End (s)</label>
                    <input type="number" id="edit-end" step="0.1">
                </div>
            </div>
            <div class="modal-controls">
                <button class="btn-skip" onclick="closeModal()">Cancel</button>
                <button class="btn-approve" onclick="saveEdit()">Save</button>
            </div>
        </div>
    </div>

    <script>
        let current = null;

        async function load() {
            try {
                const res = await fetch('/api/candidates/pending');
                const data = await res.json();

                if (data.remaining === 0) {
                    document.getElementById('content').innerHTML = `
                        <div class="empty">
                            <h2>🎉 All done!</h2>
                            <p>${data.total} candidates reviewed.</p>
                        </div>
                    `;
                    return;
                }

                current = {
                    ...data.candidate,
                    video_url: data.video_url,
                    video_title: data.video_title
                };
                const pct = ((data.total - data.remaining) / data.total * 100).toFixed(0);

                document.getElementById('progress-text').textContent =
                    `${data.index} / ${data.total} — ${data.remaining} pending`;
                document.getElementById('progress-fill').style.width = pct + '%';

                const typeColor = current.type;
                const html = `
                    <div class="candidate">
                        <div>
                            <span class="badge ${typeColor}">${current.type.toUpperCase()}</span>
                            <span style="font-weight: 600;">${current.video}</span>
                        </div>
                        <div class="meta">
                            ${current.start_s.toFixed(1)}–${current.end_s.toFixed(1)}s
                        </div>
                        <div class="query">${escapeHtml(current.text)}</div>
                        ${current.context ? `<div class="context">${escapeHtml(current.context)}</div>` : ''}
                        <div class="controls">
                            <button class="btn-approve" onclick="doApprove()">✓ Approve</button>
                            <button class="btn-reject" onclick="doReject()">✗ Reject</button>
                            <button class="btn-edit" onclick="openEdit()">✎ Edit</button>
                            <button class="btn-skip" onclick="doSkip()">⇢ Skip</button>
                            <button class="btn-play-video" onclick="playVideo()">▶ Play Video</button>
                        </div>
                    </div>
                `;
                document.getElementById('content').innerHTML = html;
            } catch (e) {
                document.getElementById('content').innerHTML =
                    `<div class="empty">Error loading: ${e.message}</div>`;
            }
        }

        async function playVideo() {
            if (!current.video_url) {
                alert('Video not available');
                return;
            }
            document.getElementById('video-overlay').classList.add('show');
            const video = document.getElementById('video-player');
            video.src = current.video_url;
            video.currentTime = current.start_s;
            video.play();
        }

        function closeVideo() {
            const video = document.getElementById('video-player');
            video.pause();
            document.getElementById('video-overlay').classList.remove('show');
        }

        async function doApprove() {
            await fetch(`/api/candidates/${current.id}/approve`, { method: 'POST' });
            load();
        }

        async function doReject() {
            await fetch(`/api/candidates/${current.id}/reject`, { method: 'POST' });
            load();
        }

        function doSkip() {
            load();
        }

        function openEdit() {
            document.getElementById('edit-text').value = current.text;
            document.getElementById('edit-start').value = current.start_s;
            document.getElementById('edit-end').value = current.end_s;
            document.getElementById('edit-modal').classList.add('show');
        }

        function closeModal() {
            document.getElementById('edit-modal').classList.remove('show');
        }

        async function saveEdit() {
            const text = document.getElementById('edit-text').value.trim();
            const start = parseFloat(document.getElementById('edit-start').value);
            const end = parseFloat(document.getElementById('edit-end').value);

            await fetch(`/api/candidates/${current.id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, start_s: start, end_s: end })
            });

            closeModal();
            load();
        }

        function escapeHtml(text) {
            return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        load();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5173)

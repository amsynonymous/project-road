# R.O.A.D AI — Road Observation and Analysis for Damage System

A local, upload-first full-stack Flask application for road damage detection using your trained Roboflow RF-DETR model through a Hosted Workflow API. The database starts empty. It contains no sample roads, fabricated detections, seeded reports, or fake dashboard metrics.

## Features
- Dashboard starts with CCTV location selection and shows only recordings assigned to the selected location.
- Add/manage actual locations before uploading CCTV video or submitting a report.
- Reports require a location selection first and are filtered by that location.
- Dashboard counts backed by SQLite records only.
- Image upload and Roboflow Hosted Workflow inference with locally annotated image and detection metadata.
- Manually uploaded CCTV video library (MP4, AVI, MOV), with optional live AI bounding-box overlay while the uploaded video plays; no physical camera or RTSP connection.
- Video frame sampling with timestamps and annotated frames.
- Duplicate grouping across nearby sampled frames using class and approximate box overlap.
- Detection inbox with Discard audit history or Report action.
- Report submission ends at `Submitted`; no repair workflow.
- Detection history, configurable confidence threshold and frame interval.
- Empty states, server-side upload checks, unique stored filenames, and upload size limits.

## Requirements
- Python 3.10 or 3.11 recommended.
- A Roboflow Hosted Workflow published and accessible to your workspace API key.
- A rotated/new Roboflow API key (do not commit it to source control).
- Video codecs supported by OpenCV. Some AVI/MOV codecs may not be supported by every OpenCV build.

## Install on Windows
Open Command Prompt in this project folder:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then enter your Roboflow API key and verify the workspace/workflow ID. The API key must remain server-side:

```bat
python app.py
```

Open `http://127.0.0.1:5000`.

## Install on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your Roboflow API key, then run:
python app.py
```

## Roboflow Hosted Workflow configuration

The app calls your Roboflow Workflow using `inference-sdk` at `https://serverless.roboflow.com`. It does not load `weights/best.pt` and does not require Ultralytics for this RF-DETR workflow.

1. Copy `.env.example` to `.env`.
2. Set `ROBOFLOW_API_KEY` to a newly generated/rotated key from your Roboflow workspace settings.
3. Confirm `ROBOFLOW_WORKSPACE` and `ROBOFLOW_WORKFLOW_ID` match the Workflow's Deploy page.
4. Install dependencies and start Flask. The first real image/video analysis will verify API connectivity.

Environment variables:
- `ROBOFLOW_API_KEY`: secret workspace API key (keep private; never put in frontend code or commit `.env`).
- `ROBOFLOW_API_URL`: default `https://serverless.roboflow.com`.
- `ROBOFLOW_WORKSPACE`: default `amsyar-asttf`.
- `ROBOFLOW_WORKFLOW_ID`: default workflow ID shown in `.env.example`.
- `ROAD_DATABASE_PATH`: SQLite path (default `instance/road_ai.sqlite3`).
- `ROAD_MAX_UPLOAD_MB`: upload limit in MB (default 300).
- `ROAD_SECRET_KEY`: Flask session/flash secret.
- `ROAD_HOST`: local host (default `127.0.0.1`).
- `ROAD_PORT`: port (default `5000`).

The app accepts standard Roboflow Workflow prediction output with `predictions` entries containing class and bounding box coordinates (`x`, `y`, `width`, `height`, or `x1`, `y1`, `x2`, `y2`). If your Workflow returns a different output shape, adjust its output block or the response parser in `app.py`.

## Data and privacy
The SQLite database is created on first startup with settings only. Media, analysis, detection, audit, and report tables start empty. Uploaded files are stored under `static/uploads`, annotated images under `static/results`, and sampled video frames under `static/frames`. Do not expose this development app to the public internet. There is no authentication; it is explicitly a local single-user prototype. Back up the database and evidence files together if needed.

## Video processing notes
Video analysis runs synchronously in the request and can take time for long clips. Set a longer frame sampling interval for long videos. The app samples frames at the configured interval, not every frame. OpenCV must be able to decode the uploaded file. No FFmpeg binary is bundled.

## Tests
Run:
```bash
pytest -q
```
Tests use a temporary SQLite database and do not require model weights for the empty-state and report workflow checks. Inference integration should be tested with your configured Roboflow Hosted Workflow and a permitted sample image/video; hosted inference requires internet and may consume Roboflow credits.

## Scope boundaries
No live CCTV, RTSP streams, GPS auto-detection, external agency API, automatic email/SMS, authentication, repair assignment, or repair status. Reports stop at `Submitted`.

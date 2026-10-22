import os
import uuid
import mimetypes
import sqlite3
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, abort, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
INSTANCE_DIR = BASE_DIR / "instance"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
FRAME_DIR = BASE_DIR / "static" / "frames"
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com").strip()
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "amsyar-asttf").strip()
ROBOFLOW_WORKFLOW_ID = os.environ.get("ROBOFLOW_WORKFLOW_ID", "road-damage-detection-v2-vroad-damage-detection-v2-xg6by-ws6eo-1-rfdetr-small-t1-logic").strip()
DB_PATH = Path(os.environ.get("ROAD_DATABASE_PATH", str(INSTANCE_DIR / "road_ai.sqlite3")))
MAX_UPLOAD_MB = int(os.environ.get("ROAD_MAX_UPLOAD_MB", "300"))
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}

for folder in (INSTANCE_DIR, UPLOAD_DIR, RESULT_DIR, FRAME_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("ROAD_SECRET_KEY", "local-prototype-change-this-key")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

_workflow_client = None

def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

@app.template_filter("display_time")
def display_time(value):
    """Render ISO timestamps as a concise human-readable local-offset time."""
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.strftime("%d %b %Y, %I:%M %p")
    except (TypeError, ValueError):
        return str(value)

def db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            confidence REAL NOT NULL DEFAULT 0.25,
            frame_interval REAL NOT NULL DEFAULT 2.0,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS locations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL DEFAULT '',
            latitude REAL,
            longitude REAL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS media (
            id TEXT PRIMARY KEY,
            original_name TEXT NOT NULL,
            stored_name TEXT NOT NULL UNIQUE,
            media_type TEXT NOT NULL CHECK(media_type IN ('image','video')),
            mime_type TEXT,
            size_bytes INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Uploaded',
            duration_seconds REAL,
            width INTEGER,
            height INTEGER,
            error_message TEXT,
            location_id TEXT REFERENCES locations(id)
        );
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            media_id TEXT NOT NULL REFERENCES media(id),
            media_type TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            confidence REAL NOT NULL,
            frame_interval REAL,
            frames_processed INTEGER NOT NULL DEFAULT 0,
            detection_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT
        );
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            analysis_id TEXT NOT NULL REFERENCES analyses(id),
            media_id TEXT NOT NULL REFERENCES media(id),
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            frame_number INTEGER,
            timestamp_seconds REAL,
            evidence_path TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'Pending' CHECK(state IN ('Pending','Discarded','Reported')),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS detection_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id TEXT NOT NULL REFERENCES detections(id),
            action TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            detection_id TEXT NOT NULL REFERENCES detections(id),
            damage_type TEXT NOT NULL,
            location TEXT NOT NULL,
            priority TEXT NOT NULL,
            recipient TEXT NOT NULL,
            description TEXT NOT NULL,
            evidence_path TEXT NOT NULL,
            source_timestamp REAL,
            submitted_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Submitted' CHECK(status = 'Submitted'),
            location_id TEXT REFERENCES locations(id)
        );
        CREATE INDEX IF NOT EXISTS idx_media_uploaded ON media(uploaded_at DESC);
        CREATE INDEX IF NOT EXISTS idx_detections_state ON detections(state);
        CREATE INDEX IF NOT EXISTS idx_reports_submitted ON reports(submitted_at DESC);
        """)
        # Safe additive migrations for databases created by earlier versions.
        for table, column, definition in [
            ("media", "location_id", "TEXT REFERENCES locations(id)"),
            ("reports", "location_id", "TEXT REFERENCES locations(id)"),
            ("locations", "latitude", "REAL"),
            ("locations", "longitude", "REAL")
        ]:
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.execute(
            "INSERT OR IGNORE INTO settings(id, confidence, frame_interval, updated_at) VALUES(1, 0.25, 2.0, ?)",
            (now_iso(),)
        )

init_db()

def get_settings():
    with db() as conn:
        row = conn.execute("SELECT * FROM settings WHERE id=1").fetchone()
    return dict(row)

def save_settings(confidence, frame_interval):
    with db() as conn:
        conn.execute(
            "UPDATE settings SET confidence=?, frame_interval=?, updated_at=? WHERE id=1",
            (confidence, frame_interval, now_iso())
        )

def model_status():
    configured = bool(ROBOFLOW_API_KEY and ROBOFLOW_WORKSPACE and ROBOFLOW_WORKFLOW_ID)
    return {"available": configured, "path_configured": configured, "provider": "Roboflow Hosted Workflow"}


def get_workflow_client():
    global _workflow_client
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("Roboflow API key is missing. Add ROBOFLOW_API_KEY to your .env file.")
    if not ROBOFLOW_WORKSPACE or not ROBOFLOW_WORKFLOW_ID:
        raise RuntimeError("Roboflow workflow configuration is missing. Check ROBOFLOW_WORKSPACE and ROBOFLOW_WORKFLOW_ID.")
    if _workflow_client is None:
        from inference_sdk import InferenceHTTPClient
        _workflow_client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=ROBOFLOW_API_KEY,
        )
    return _workflow_client


def _find_predictions(value):
    """Find a predictions list in common Roboflow Workflow response shapes."""
    if isinstance(value, dict):
        preds = value.get("predictions")
        if isinstance(preds, list):
            return preds
        # Some workflows wrap model outputs inside named output objects.
        for child in value.values():
            found = _find_predictions(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_predictions(child)
            if found is not None:
                return found
    return None


def _workflow_detections(image_path, confidence=None, use_cache=True):
    """Run an image through the configured hosted Roboflow Workflow.

    Live playback passes use_cache=False so changing video frames are not served
    from a cached response.
    """
    result = get_workflow_client().run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_WORKFLOW_ID,
        images={"image": str(image_path)},
        use_cache=use_cache,
    )
    predictions = _find_predictions(result)
    if predictions is None:
        raise RuntimeError("Roboflow Workflow response did not contain a 'predictions' list. Check the workflow output block and its response format.")
    threshold = float(confidence if confidence is not None else get_settings()["confidence"])
    detections = []
    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        try:
            conf = float(pred.get("confidence", pred.get("score", 0.0)))
            if conf > 1.0:
                conf /= 100.0
            if conf < threshold:
                continue
            # Roboflow object-detection outputs normally provide center x/y and width/height.
            if all(k in pred for k in ("x", "y", "width", "height")):
                x, y = float(pred["x"]), float(pred["y"])
                w, h = float(pred["width"]), float(pred["height"])
                coords = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
            elif all(k in pred for k in ("x1", "y1", "x2", "y2")):
                coords = [float(pred[k]) for k in ("x1", "y1", "x2", "y2")]
            else:
                continue
            label = pred.get("class") or pred.get("class_name") or pred.get("label") or pred.get("class_id", "unknown")
            detections.append({"class_name": str(label), "confidence": conf, "coords": coords})
        except (TypeError, ValueError):
            continue
    return detections


def annotate_image(image_path, detections, output_path):
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Could not read image for annotation.")
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["coords"]]
        x1=max(0,x1); y1=max(0,y1); x2=min(image.shape[1]-1,x2); y2=min(image.shape[0]-1,y2)
        cv2.rectangle(image, (x1,y1), (x2,y2), (30, 190, 90), 2)
        text = f"{det['class_name']} {det['confidence']:.0%}"
        cv2.putText(image, text, (x1, max(18,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,190,90), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError("Could not save annotated image.")


def class_label(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and class_id < len(names):
        return str(names[class_id])
    return str(class_id)

def allowed_file(filename, allowed):
    return bool(filename and "." in filename and filename.rsplit(".", 1)[1].lower() in allowed)

def detected_mime(path):
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"

def image_magic_ok(path):
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def create_media(file, media_type):
    original = secure_filename(file.filename or "")
    if not original:
        raise ValueError("Please choose a file with a valid filename.")
    allowed = IMAGE_EXTENSIONS if media_type == "image" else VIDEO_EXTENSIONS
    if not allowed_file(original, allowed):
        extensions = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported file type. Allowed formats: {extensions}.")
    stored = f"{uuid.uuid4().hex}_{original}"
    target = UPLOAD_DIR / stored
    file.save(target)
    if media_type == "image" and not image_magic_ok(target):
        target.unlink(missing_ok=True)
        raise ValueError("The uploaded file is not a valid image.")
    size = target.stat().st_size
    mime = detected_mime(target)
    media_id = uuid.uuid4().hex
    width = height = None
    duration = None
    try:
        if media_type == "image":
            from PIL import Image
            with Image.open(target) as im:
                width, height = im.size
        else:
            import cv2
            cap = cv2.VideoCapture(str(target))
            if not cap.isOpened():
                cap.release()
                target.unlink(missing_ok=True)
                raise ValueError("Could not read this video. Check its format and codec.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = float(count / fps) if fps and fps > 0 and count > 0 else None
            cap.release()
    except ImportError as exc:
        target.unlink(missing_ok=True)
        raise ValueError("Required image/video library is unavailable. Install requirements.txt.") from exc
    except ValueError:
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise ValueError(f"Unable to inspect uploaded file: {exc}") from exc

    with db() as conn:
        conn.execute("""INSERT INTO media
            (id, original_name, stored_name, media_type, mime_type, size_bytes, uploaded_at,
             status, duration_seconds, width, height)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (media_id, original, stored, media_type, mime, size, now_iso(), "Uploaded",
             duration, width, height))
    return media_id

def media_row(media_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM media WHERE id=?", (media_id,)).fetchone()
    return row

def run_image_analysis(media_id, confidence=None):
    media = media_row(media_id)
    if not media or media["media_type"] != "image":
        abort(404)
    settings = get_settings()
    confidence = settings["confidence"] if confidence is None else confidence
    analysis_id = uuid.uuid4().hex
    started = now_iso()
    with db() as conn:
        conn.execute("""INSERT INTO analyses
        (id, media_id, media_type, started_at, status, confidence)
        VALUES(?,?,?,?,?,?)""", (analysis_id, media_id, "image", started, "Running", confidence))
        conn.execute("UPDATE media SET status='Processing', error_message=NULL WHERE id=?", (media_id,))
    try:
        import cv2
        source_path = UPLOAD_DIR / media["stored_name"]
        detections = _workflow_detections(source_path, confidence)
        output_name = f"{analysis_id}_annotated.jpg"
        output_path = RESULT_DIR / output_name
        annotate_image(source_path, detections, output_path)
        count = 0
        with db() as conn:
            for det in detections:
                coords = det["coords"]
                conn.execute("""INSERT INTO detections
                (id, analysis_id, media_id, class_name, confidence, x1,y1,x2,y2,
                 evidence_path, state, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (uuid.uuid4().hex, analysis_id, media_id, det["class_name"],
                 det["confidence"], *coords, f"results/{output_name}", "Pending", now_iso()))
                count += 1
            conn.execute("""UPDATE analyses SET completed_at=?, status='Completed',
                frames_processed=1, detection_count=? WHERE id=?""", (now_iso(), count, analysis_id))
            conn.execute("UPDATE media SET status='Analysed' WHERE id=?", (media_id,))
        return analysis_id
    except Exception as exc:
        message = str(exc)
        with db() as conn:
            conn.execute("UPDATE analyses SET completed_at=?, status='Failed', error_message=? WHERE id=?",
                         (now_iso(), message, analysis_id))
            conn.execute("UPDATE media SET status='Failed', error_message=? WHERE id=?", (message, media_id))
        raise

def video_frame_analysis(media_id, confidence=None, frame_interval=None):
    media = media_row(media_id)
    if not media or media["media_type"] != "video":
        abort(404)
    settings = get_settings()
    confidence = settings["confidence"] if confidence is None else confidence
    frame_interval = settings["frame_interval"] if frame_interval is None else frame_interval
    analysis_id = uuid.uuid4().hex
    started = now_iso()
    with db() as conn:
        conn.execute("""INSERT INTO analyses
        (id, media_id, media_type, started_at, status, confidence, frame_interval)
        VALUES(?,?,?,?,?,?,?)""",
        (analysis_id, media_id, "video", started, "Running", confidence, frame_interval))
        conn.execute("UPDATE media SET status='Processing', error_message=NULL WHERE id=?", (media_id,))
    cap = None
    try:
        import cv2
        cap = cv2.VideoCapture(str(UPLOAD_DIR / media["stored_name"]))
        if not cap.isOpened():
            raise ValueError("Could not open this video. The codec may not be supported by OpenCV.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = total_frames / fps if total_frames else media["duration_seconds"]
        frame_number = 0
        frames_processed = 0
        raw_detections = []
        next_sample = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_number / fps
            if timestamp + 1e-6 >= next_sample:
                frame_name = f"{analysis_id}_frame_{frame_number:08d}.jpg"
                frame_path = FRAME_DIR / frame_name
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError("Could not save sampled video frame for hosted inference.")
                frame_detections = _workflow_detections(frame_path, confidence)
                annotate_image(frame_path, frame_detections, frame_path)
                frames_processed += 1
                for det in frame_detections:
                    raw_detections.append({
                        "class_name": det["class_name"],
                        "confidence": det["confidence"], "coords": det["coords"],
                        "frame_number": frame_number, "timestamp": timestamp,
                        "evidence_path": f"frames/{frame_name}"
                    })
                next_sample += frame_interval
            frame_number += 1
        cap.release()
        cap = None

        # Group likely duplicate detections from adjacent sampled frames when class and normalized box overlap.
        grouped = []
        for det in raw_detections:
            x1,y1,x2,y2 = det["coords"]
            w = max(1, media["width"] or 1)
            h = max(1, media["height"] or 1)
            norm = (x1/w, y1/h, x2/w, y2/h)
            match = None
            for item in reversed(grouped[-100:]):
                if item["class_name"] != det["class_name"]:
                    continue
                # For a fixed-camera CCTV clip, the same physical damage may appear in many frames.
                # Compare the whole clip (not only nearby timestamps) to avoid duplicate captures.
                a = item["norm"]; ix1=max(a[0],norm[0]); iy1=max(a[1],norm[1]); ix2=min(a[2],norm[2]); iy2=min(a[3],norm[3])
                inter=max(0,ix2-ix1)*max(0,iy2-iy1)
                area_a=max(1e-9,(a[2]-a[0])*(a[3]-a[1]))
                area_b=max(1e-9,(norm[2]-norm[0])*(norm[3]-norm[1]))
                iou=inter/(area_a+area_b-inter+1e-9)
                if iou >= 0.25:
                    match=item
                    break
            if match:
                match["last_timestamp"] = det["timestamp"]
                if det["confidence"] > match["confidence"]:
                    match.update(det)
                    match["norm"] = norm
            else:
                grouped.append({**det, "last_timestamp": det["timestamp"], "norm": norm})

        with db() as conn:
            for det in grouped:
                x1,y1,x2,y2 = det["coords"]
                conn.execute("""INSERT INTO detections
                (id, analysis_id, media_id, class_name, confidence, x1,y1,x2,y2,
                 frame_number, timestamp_seconds, evidence_path, state, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (uuid.uuid4().hex, analysis_id, media_id, det["class_name"], det["confidence"],
                 x1,y1,x2,y2,det["frame_number"],det["timestamp"],det["evidence_path"],"Pending",now_iso()))
            conn.execute("""UPDATE analyses SET completed_at=?, status='Completed',
                frames_processed=?, detection_count=? WHERE id=?""",
                (now_iso(), frames_processed, len(grouped), analysis_id))
            conn.execute("UPDATE media SET status='Analysed', duration_seconds=COALESCE(?,duration_seconds) WHERE id=?",
                         (duration, media_id))
        return analysis_id
    except Exception as exc:
        if cap is not None:
            cap.release()
        message = str(exc)
        with db() as conn:
            conn.execute("UPDATE analyses SET completed_at=?, status='Failed', error_message=? WHERE id=?",
                         (now_iso(), message, analysis_id))
            conn.execute("UPDATE media SET status='Failed', error_message=? WHERE id=?", (message, media_id))
        raise

@app.context_processor
def inject_global():
    with db() as conn:
        pending_count = conn.execute("SELECT COUNT(*) FROM detections WHERE state='Pending'").fetchone()[0]
    return {"model_available": model_status()["available"], "max_upload_mb": MAX_UPLOAD_MB, "pending_count": pending_count}

@app.route("/api/realtime-detect/<media_id>", methods=["POST"])
def realtime_detect_frame(media_id):
    """Infer one browser-captured frame for the live playback overlay.

    This endpoint only returns frame predictions for the on-screen overlay.
    It does not create database detection records; use the normal video analysis
    workflow to save reviewed detections into Detection Inbox/History.
    """
    with db() as conn:
        media = conn.execute(
            "SELECT id, media_type FROM media WHERE id=?", (media_id,)
        ).fetchone()
    if not media or media["media_type"] != "video":
        return jsonify({"error": "Uploaded video not found."}), 404
    if not ROBOFLOW_API_KEY:
        return jsonify({"error": "Roboflow API is not configured."}), 503

    frame_file = request.files.get("frame")
    if not frame_file:
        return jsonify({"error": "No video frame was supplied."}), 400

    temp_path = FRAME_DIR / f"live_{uuid.uuid4().hex}.jpg"
    try:
        import cv2
        import numpy as np

        raw = frame_file.read()
        if not raw or len(raw) > 4 * 1024 * 1024:
            return jsonify({"error": "Frame is empty or exceeds the 4 MB limit."}), 413
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "The supplied frame is not a valid image."}), 400
        height, width = image.shape[:2]
        if width > 960:
            scale = 960 / width
            image = cv2.resize(image, (960, max(1, int(height * scale))))
            height, width = image.shape[:2]
        if not cv2.imwrite(str(temp_path), image):
            return jsonify({"error": "Could not prepare the frame for inference."}), 500

        detections = _workflow_detections(
            temp_path, confidence=get_settings()["confidence"], use_cache=False
        )
        return jsonify({
            "detections": detections,
            "image_width": width,
            "image_height": height,
            "time_detected": now_iso(),
        })
    except Exception as exc:
        app.logger.exception("Realtime frame inference failed")
        return jsonify({"error": f"Realtime inference failed: {str(exc)[:240]}"}), 502
    finally:
        temp_path.unlink(missing_ok=True)


@app.route("/")
def dashboard():
    selected_location = request.args.get("location_id", "").strip()
    with db() as conn:
        stats = {
            "images": conn.execute("SELECT COUNT(*) FROM analyses WHERE media_type='image' AND status='Completed'").fetchone()[0],
            "detections": conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0],
            "pending": conn.execute("SELECT COUNT(*) FROM detections WHERE state='Pending'").fetchone()[0],
            "reports": conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0],
        }
        locations = conn.execute("SELECT * FROM locations ORDER BY name").fetchall()
        if selected_location and not any(x["id"] == selected_location for x in locations):
            selected_location = ""
        selected = next((x for x in locations if x["id"] == selected_location), None)
        videos = []
        dashboard_video = None
        if selected:
            videos = conn.execute("SELECT * FROM media WHERE media_type='video' AND location_id=? ORDER BY uploaded_at DESC", (selected_location,)).fetchall()
            dashboard_video = next((v for v in videos if v["status"] == "Analysed"), videos[0] if videos else None)
        activity = conn.execute("""SELECT a.id, a.media_type,
            CASE
                WHEN a.status='Failed' THEN 'Failed'
                WHEN EXISTS (SELECT 1 FROM detections d WHERE d.analysis_id=a.id AND d.state='Pending') THEN 'Pending review'
                ELSE 'Completed'
            END AS display_status,
            a.started_at, a.detection_count, m.original_name,
            l.name AS location_name, l.id AS location_id
            FROM analyses a JOIN media m ON m.id=a.media_id
            LEFT JOIN locations l ON l.id=m.location_id
            ORDER BY a.started_at DESC LIMIT 8""").fetchall()
    return render_template("dashboard.html", stats=stats, activity=activity, locations=locations,
                           selected_location=selected_location, selected=selected, location_videos=videos,
                           dashboard_video=dashboard_video if selected else None)

@app.route("/locations", methods=["GET", "POST"])
def locations_page():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip()
        try:
            latitude = float(request.form.get("latitude", ""))
            longitude = float(request.form.get("longitude", ""))
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                raise ValueError
        except (TypeError, ValueError):
            latitude = longitude = None
        if not name:
            flash("Enter a location name.", "error")
        else:
            try:
                with db() as conn:
                    conn.execute("INSERT INTO locations(id,name,description,latitude,longitude,created_at) VALUES(?,?,?,?,?,?)",
                                 (uuid.uuid4().hex, name, description, latitude, longitude, now_iso()))
                flash("Location added.", "success")
            except sqlite3.IntegrityError:
                flash("That location already exists.", "error")
        return redirect(url_for("locations_page"))
    with db() as conn:
        locations = conn.execute("""SELECT l.*, COUNT(m.id) AS video_count FROM locations l
            LEFT JOIN media m ON m.location_id=l.id AND m.media_type='video'
            GROUP BY l.id ORDER BY l.name""").fetchall()
    return render_template("locations.html", locations=locations)

@app.route("/locations/<location_id>")
def location_detail(location_id):
    with db() as conn:
        location = conn.execute("SELECT * FROM locations WHERE id=?", (location_id,)).fetchone()
        if not location:
            abort(404)
        reports = conn.execute("SELECT * FROM reports WHERE location_id=? ORDER BY submitted_at DESC", (location_id,)).fetchall()
        # Only unresolved detections appear in Check Detection. Once reported or removed,
        # they remain in the audit/history but no longer clutter this action list.
        detections = conn.execute("""SELECT d.*, m.original_name, m.media_type, a.started_at AS analysed_at, l.description AS location_description, l.latitude AS location_latitude, l.longitude AS location_longitude, l.created_at AS location_created_at
            FROM detections d JOIN media m ON m.id=d.media_id JOIN analyses a ON a.id=d.analysis_id LEFT JOIN locations l ON l.id=m.location_id
            WHERE m.location_id=? AND d.state='Pending' ORDER BY d.created_at DESC LIMIT 100""", (location_id,)).fetchall()
        videos = conn.execute("SELECT * FROM media WHERE location_id=? AND media_type='video' ORDER BY uploaded_at DESC", (location_id,)).fetchall()
    return render_template("location_detail.html", location=location, reports=reports, detections=detections, videos=videos)

@app.route("/images", methods=["GET", "POST"])
def images():
    if request.method == "POST":
        try:
            location_id = request.form.get("location_id", "").strip()
            with db() as conn:
                location = conn.execute("SELECT id FROM locations WHERE id=?", (location_id,)).fetchone()
            if not location:
                raise ValueError("Create and select a registered location before uploading an image.")
            media_id = create_media(request.files.get("image"), "image")
            with db() as conn:
                conn.execute("UPDATE media SET location_id=? WHERE id=?", (location_id, media_id))
            confidence = get_settings()["confidence"]
            analysis_id = run_image_analysis(media_id, confidence)
            flash("Image analysis completed and assigned to the selected location.", "success")
            return redirect(url_for("analysis_detail", analysis_id=analysis_id))
        except Exception as exc:
            flash(str(exc), "error")
            return redirect(url_for("images"))
    with db() as conn:
        locations = conn.execute("SELECT * FROM locations ORDER BY name").fetchall()
        recent = conn.execute("""SELECT m.*, l.name AS location_name, a.id AS analysis_id, a.status AS analysis_status,
            a.detection_count FROM media m LEFT JOIN locations l ON l.id=m.location_id
            LEFT JOIN analyses a ON a.media_id=m.id
            WHERE m.media_type='image' ORDER BY m.uploaded_at DESC LIMIT 20""").fetchall()
    return render_template("images.html", recent=recent, locations=locations)

@app.route("/videos", methods=["GET", "POST"])
def videos():
    if request.method == "POST":
        try:
            location_id = request.form.get("location_id", "").strip()
            with db() as conn:
                location = conn.execute("SELECT id FROM locations WHERE id=?", (location_id,)).fetchone()
            if not location:
                raise ValueError("Create and select a location before uploading CCTV footage.")
            media_id = create_media(request.files.get("video"), "video")
            with db() as conn:
                conn.execute("UPDATE media SET location_id=? WHERE id=?", (location_id, media_id))
            flash("Video uploaded to the selected location. Run analysis when ready.", "success")
            return redirect(url_for("video_detail", media_id=media_id))
        except Exception as exc:
            flash(str(exc), "error")
            return redirect(url_for("videos"))
    selected_location = request.args.get("location_id", "").strip()
    with db() as conn:
        locations = conn.execute("SELECT * FROM locations ORDER BY name").fetchall()
        if selected_location and not any(x["id"] == selected_location for x in locations):
            selected_location = ""
        records = []
        if selected_location:
            records = conn.execute("""SELECT m.*, COALESCE((SELECT SUM(a.detection_count) FROM analyses a
            WHERE a.media_id=m.id AND a.status='Completed'),0) AS detections
            FROM media m WHERE m.media_type='video' AND m.location_id=? ORDER BY m.uploaded_at DESC""", (selected_location,)).fetchall()
    return render_template("videos.html", records=records, locations=locations, selected_location=selected_location)

@app.route("/videos/<media_id>/file")
def video_file(media_id):
    media = media_row(media_id)
    if not media or media["media_type"] != "video":
        abort(404)
    path = UPLOAD_DIR / media["stored_name"]
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype=media["mime_type"] or "video/mp4", conditional=True)

@app.route("/videos/<media_id>")
def video_detail(media_id):
    media = media_row(media_id)
    if not media or media["media_type"] != "video":
        abort(404)
    with db() as conn:
        analyses = conn.execute("SELECT * FROM analyses WHERE media_id=? ORDER BY started_at DESC", (media_id,)).fetchall()
        detections = conn.execute("""SELECT * FROM detections WHERE media_id=? ORDER BY timestamp_seconds, created_at""",
                                  (media_id,)).fetchall()
    settings = get_settings()
    return render_template("video_detail.html", media=media, analyses=analyses, detections=detections,
                           settings_confidence=settings["confidence"], settings_interval=settings["frame_interval"])

@app.route("/videos/<media_id>/analyse", methods=["POST"])
def analyse_video(media_id):
    try:
        settings = get_settings()
        confidence = settings["confidence"]
        interval = float(request.form.get("frame_interval", settings["frame_interval"]))
        if not 0.25 <= interval <= 60:
            raise ValueError("Frame interval must be between 0.25 and 60 seconds.")
        analysis_id = video_frame_analysis(media_id, confidence, interval)
        flash("Video analysis completed.", "success")
        return redirect(url_for("analysis_detail", analysis_id=analysis_id))
    except Exception as exc:
        flash(f"Video analysis failed: {exc}", "error")
        return redirect(url_for("video_detail", media_id=media_id))

@app.route("/analysis/<analysis_id>")
def analysis_detail(analysis_id):
    with db() as conn:
        analysis = conn.execute("""SELECT a.*, m.original_name, m.stored_name, m.media_type, m.id AS media_id
        FROM analyses a JOIN media m ON m.id=a.media_id WHERE a.id=?""", (analysis_id,)).fetchone()
        if not analysis:
            abort(404)
        detections = conn.execute("SELECT * FROM detections WHERE analysis_id=? ORDER BY timestamp_seconds, created_at",
                                  (analysis_id,)).fetchall()
    return render_template("analysis_detail.html", analysis=analysis, detections=detections)

@app.route("/inbox")
def inbox():
    state = request.args.get("state", "Pending")
    search = request.args.get("q", "").strip()
    location_id = request.args.get("location_id", "").strip()
    sort_by = request.args.get("sort", "latest").lower()
    if sort_by not in {"latest", "oldest"}: sort_by = "latest"
    valid_states = {"Pending", "Discarded", "Reported", "All"}
    if state not in valid_states:
        state = "Pending"
    with db() as conn:
        locations = conn.execute("SELECT * FROM locations ORDER BY name").fetchall()
        if location_id and location_id != "All" and not any(x["id"] == location_id for x in locations):
            location_id = ""
        query = """SELECT d.*, m.original_name, m.media_type, m.location_id,
            l.name AS location_name, a.started_at AS analysed_at
            FROM detections d JOIN media m ON m.id=d.media_id
            JOIN analyses a ON a.id=d.analysis_id LEFT JOIN locations l ON l.id=m.location_id WHERE 1=1"""
        params = []
        # Avoid a cluttered inbox: show detections only after a registered location is selected,
        # unless the user explicitly chooses All locations.
        if location_id and location_id != "All":
            query += " AND m.location_id=?"
            params.append(location_id)
        elif location_id != "All":
            query += " AND 1=0"
        if state != "All":
            query += " AND d.state=?"
            params.append(state)
        if search:
            query += " AND (d.class_name LIKE ? OR m.original_name LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        query += " ORDER BY d.created_at " + ("ASC" if sort_by == "oldest" else "DESC")
        detections = conn.execute(query, params).fetchall()
        registered_locations = conn.execute("""SELECT l.*,
            COUNT(DISTINCT CASE WHEN d.state='Pending' THEN d.id END) AS pending_report_count,
            COUNT(DISTINCT d.id) AS detection_count
            FROM locations l LEFT JOIN media m ON m.location_id=l.id
            LEFT JOIN detections d ON d.media_id=m.id
            GROUP BY l.id ORDER BY pending_report_count DESC, l.name ASC""").fetchall()
    return render_template("inbox.html", detections=detections, state=state, search=search,
                           registered_locations=registered_locations, locations=locations, location_id=location_id, sort_by=sort_by)

@app.route("/detections/<detection_id>/discard", methods=["POST"])
def discard_detection(detection_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM detections WHERE id=?", (detection_id,)).fetchone()
        if not row:
            abort(404)
        if row["state"] == "Pending":
            conn.execute("UPDATE detections SET state='Discarded' WHERE id=?", (detection_id,))
            conn.execute("INSERT INTO detection_audit(detection_id,action,note,created_at) VALUES(?,?,?,?)",
                         (detection_id, "Discarded", request.form.get("note", "").strip(), now_iso()))
    flash("Detection marked as discarded. Audit record retained.", "success")
    return redirect(url_for("inbox"))

@app.route("/detections/<detection_id>/report", methods=["GET", "POST"])
def report_detection(detection_id):
    with db() as conn:
        detection = conn.execute("""SELECT d.*, m.original_name, m.media_type, m.location_id AS source_location_id,
            l.name AS source_location_name, l.description AS source_location_description, l.latitude AS source_latitude, l.longitude AS source_longitude, l.created_at AS source_location_created_at
        FROM detections d JOIN media m ON m.id=d.media_id
        LEFT JOIN locations l ON l.id=m.location_id WHERE d.id=?""", (detection_id,)).fetchone()
        if not detection:
            abort(404)
        existing = conn.execute("SELECT id FROM reports WHERE detection_id=?", (detection_id,)).fetchone()
    if existing:
        flash("A report has already been submitted for this detection.", "error")
        return redirect(url_for("report_detail", report_id=existing["id"]))
    if detection["state"] == "Discarded":
        flash("Discarded detections cannot be reported. Review the detection first.", "error")
        return redirect(url_for("inbox"))
    # Report location is inherited from the CCTV/image source and cannot be changed here.
    with db() as conn:
        locations = conn.execute("SELECT * FROM locations ORDER BY name").fetchall()
    if request.method == "POST":
        damage_type = request.form.get("damage_type", "").strip()
        location_id = detection["source_location_id"] or ""
        with db() as conn:
            chosen_location = conn.execute("SELECT * FROM locations WHERE id=?", (location_id,)).fetchone() if location_id else None
        location = chosen_location["name"] if chosen_location else ""
        priority = request.form.get("priority", "").strip()
        recipient = request.form.get("recipient", "").strip()
        description = request.form.get("description", "").strip()
        if not location_id or not chosen_location:
            flash("This detection has no registered location. Assign the recording to a registered location and analyse it again before submitting a report.", "error")
        elif not all([damage_type, location, priority, recipient, description]):
            flash("Complete all required fields before submitting.", "error")
        elif priority not in {"Low", "Medium", "High", "Urgent"}:
            flash("Select a valid priority.", "error")
        else:
            report_id = "ROAD-" + uuid.uuid4().hex[:10].upper()
            with db() as conn:
                conn.execute("""INSERT INTO reports
                (id,detection_id,damage_type,location,priority,recipient,description,evidence_path,
                 source_timestamp,submitted_at,status,location_id)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (report_id,detection_id,damage_type,location,priority,recipient,description,
                 detection["evidence_path"],detection["timestamp_seconds"],now_iso(),"Submitted",location_id))
                conn.execute("UPDATE detections SET state='Reported' WHERE id=?", (detection_id,))
                conn.execute("INSERT INTO detection_audit(detection_id,action,note,created_at) VALUES(?,?,?,?)",
                             (detection_id, "Reported", f"Report {report_id} submitted", now_iso()))
            return redirect(url_for("report_detail", report_id=report_id))
    return render_template("report_form.html", detection=detection, locations=locations)

@app.route("/reports")
def reports():
    search = request.args.get("q", "").strip()
    location_id = request.args.get("location_id", "").strip()
    sort_by = request.args.get("sort", "latest").lower()
    if sort_by not in {"latest", "oldest"}: sort_by = "latest"
    with db() as conn:
        locations = conn.execute("SELECT * FROM locations ORDER BY name").fetchall()
        if location_id and not any(x["id"] == location_id for x in locations):
            location_id = ""
        rows = []
        if location_id:
            query = "SELECT * FROM reports WHERE location_id=?"
            params = [location_id]
            if search:
                query += " AND (id LIKE ? OR damage_type LIKE ? OR location LIKE ? OR recipient LIKE ?)"
                params.extend([f"%{search}%"] * 4)
            query += " ORDER BY submitted_at " + ("ASC" if sort_by == "oldest" else "DESC")
            rows = conn.execute(query, params).fetchall()
    return render_template("reports.html", reports=rows, search=search, locations=locations,
                           selected_location=location_id, sort_by=sort_by)

@app.route("/reports/<report_id>")
def report_detail(report_id):
    with db() as conn:
        report = conn.execute("""SELECT r.*, l.latitude AS latitude, l.longitude AS longitude,
            l.name AS pinned_location_name FROM reports r
            LEFT JOIN locations l ON l.id=r.location_id WHERE r.id=?""", (report_id,)).fetchone()
    if not report:
        abort(404)
    return render_template("report_detail.html", report=report)

@app.route("/history")
def history():
    status = request.args.get("status", "All")
    location_id = request.args.get("location_id", "").strip()
    sort_by = request.args.get("sort", "latest").lower()
    if sort_by not in {"latest", "oldest"}: sort_by = "latest"
    with db() as conn:
        locations = conn.execute("SELECT id, name FROM locations ORDER BY name").fetchall()
        sql = "SELECT a.*, m.original_name, m.location_id, l.name AS location_name FROM analyses a JOIN media m ON m.id=a.media_id LEFT JOIN locations l ON l.id=m.location_id WHERE 1=1"
        params = []
        if status != "All":
            sql += " AND a.status=?"
            params.append(status)
        if location_id and location_id != "All":
            sql += " AND m.location_id=?"
            params.append(location_id)
        sql += " ORDER BY a.started_at " + ("ASC" if sort_by == "oldest" else "DESC")
        rows = conn.execute(sql, params).fetchall()
    return render_template("history.html", rows=rows, status=status, locations=locations, location_id=location_id, sort_by=sort_by)

def _delete_media_tree(media_id):
    """Delete one uploaded image/video and all dependent analysis records and files."""
    with db() as conn:
        media = conn.execute("SELECT * FROM media WHERE id=?", (media_id,)).fetchone()
        if not media:
            abort(404)
        analysis_ids = [r[0] for r in conn.execute("SELECT id FROM analyses WHERE media_id=?", (media_id,)).fetchall()]
        detection_ids = [r[0] for r in conn.execute("SELECT id FROM detections WHERE media_id=?", (media_id,)).fetchall()]
        if detection_ids:
            marks = ",".join("?" for _ in detection_ids)
            conn.execute(f"DELETE FROM reports WHERE detection_id IN ({marks})", detection_ids)
            conn.execute(f"DELETE FROM detection_audit WHERE detection_id IN ({marks})", detection_ids)
            conn.execute(f"DELETE FROM detections WHERE id IN ({marks})", detection_ids)
        conn.execute("DELETE FROM analyses WHERE media_id=?", (media_id,))
        conn.execute("DELETE FROM media WHERE id=?", (media_id,))
    for folder in (UPLOAD_DIR, RESULT_DIR, FRAME_DIR):
        for file in folder.glob(f"*{media_id}*"):
            try: file.unlink()
            except OSError: pass
    # Remove associated media source by stored filename.
    try: (UPLOAD_DIR / media["stored_name"]).unlink(missing_ok=True)
    except OSError: pass

@app.route("/media/<media_id>/delete", methods=["POST"])
def delete_media(media_id):
    media = media_row(media_id)
    if not media: abort(404)
    kind = media["media_type"]
    _delete_media_tree(media_id)
    flash("Image detection record removed." if kind == "image" else "CCTV recording and its related analysis removed.", "success")
    return redirect(url_for("images" if kind == "image" else "videos"))

@app.route("/locations/<location_id>/delete", methods=["POST"])
def delete_location(location_id):
    with db() as conn:
        loc = conn.execute("SELECT id FROM locations WHERE id=?", (location_id,)).fetchone()
        if not loc: abort(404)
        conn.execute("UPDATE media SET location_id=NULL WHERE location_id=?", (location_id,))
        conn.execute("UPDATE reports SET location_id=NULL WHERE location_id=?", (location_id,))
        conn.execute("DELETE FROM locations WHERE id=?", (location_id,))
    flash("Location removed. Existing media and reports were kept, but are no longer assigned to that location.", "success")
    return redirect(url_for("locations_page"))

@app.route("/reports/<report_id>/delete", methods=["POST"])
def delete_report(report_id):
    with db() as conn:
        row = conn.execute("SELECT id FROM reports WHERE id=?", (report_id,)).fetchone()
        if not row: abort(404)
        conn.execute("DELETE FROM reports WHERE id=?", (report_id,))
    flash("Report removed.", "success")
    return redirect(url_for("reports"))

@app.route("/history/<analysis_id>/delete", methods=["POST"])
def delete_history_item(analysis_id):
    with db() as conn:
        row = conn.execute("SELECT media_id FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        if not row: abort(404)
        media_id = row["media_id"]
    _delete_media_tree(media_id)
    flash("Detection history and associated media removed.", "success")
    return redirect(url_for("history"))

@app.route("/detections/<detection_id>/delete", methods=["POST"])
def delete_detection(detection_id):
    with db() as conn:
        row = conn.execute("SELECT id FROM detections WHERE id=?", (detection_id,)).fetchone()
        if not row: abort(404)
        conn.execute("DELETE FROM reports WHERE detection_id=?", (detection_id,))
        conn.execute("DELETE FROM detection_audit WHERE detection_id=?", (detection_id,))
        conn.execute("DELETE FROM detections WHERE id=?", (detection_id,))
    flash("Detection record removed.", "success")
    return redirect(request.referrer or url_for("inbox"))

@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    if request.method == "POST":
        try:
            confidence = float(request.form.get("confidence", "0.25"))
            interval = float(request.form.get("frame_interval", "2"))
            if not 0.01 <= confidence <= 0.99:
                raise ValueError("Confidence must be between 0.01 and 0.99.")
            if not 0.25 <= interval <= 60:
                raise ValueError("Frame interval must be between 0.25 and 60 seconds.")
            save_settings(confidence, interval)
            flash("Settings saved.", "success")
        except Exception as exc:
            flash(str(exc), "error")
        return redirect(url_for("settings_page"))
    settings = get_settings()
    # Workflow class labels are returned by inference; do not load local YOLO weights.
    classes = None
    return render_template("settings.html", settings=settings, classes=classes)

@app.route("/health")
def health():
    return jsonify({"app": "R.O.A.D AI", "model_available": model_status()["available"]})

@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, message="The requested page or record was not found."), 404

@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    flash(f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.", "error")
    return redirect(request.referrer or url_for("dashboard"))

if __name__ == "__main__":
    app.run(host=os.environ.get("ROAD_HOST", "127.0.0.1"),
            port=int(os.environ.get("ROAD_PORT", "5000")), debug=False)

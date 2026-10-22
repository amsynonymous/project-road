import io
import sqlite3
from pathlib import Path

def test_dashboard_starts_empty(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"No activity yet" in response.data
    assert b"Uploaded videos" in response.data

def test_upload_rejects_unsupported_image(client):
    response = client.post("/images", data={
        "image": (io.BytesIO(b"not an image"), "roads.exe")
    }, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Unsupported file type" in response.data

def test_upload_rejects_invalid_image_content(client):
    response = client.post("/images", data={
        "image": (io.BytesIO(b"not really a png"), "road.png")
    }, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"not a valid image" in response.data

def test_missing_model_is_reported(client):
    response = client.get("/settings")
    assert response.status_code == 200
    assert b"Model weights missing" in response.data

def test_report_submission_and_discard_action(client, app):
    road_app = app
    with road_app.app_context():
        with road_app.test_request_context():
            from PIL import Image
            img = Image.new("RGB", (20, 20), "white")
            path = road_app.UPLOAD_DIR / "test_source.png"
            img.save(path)
        with road_app.db() as conn:
            conn.execute("""INSERT INTO media
                (id,original_name,stored_name,media_type,mime_type,size_bytes,uploaded_at,status,width,height)
                VALUES('m1','road.png','test_source.png','image','image/png',100,'2026-01-01T00:00:00+00:00','Uploaded',20,20)""")
            conn.execute("""INSERT INTO analyses
                (id,media_id,media_type,started_at,completed_at,status,confidence,frames_processed,detection_count)
                VALUES('a1','m1','image','2026-01-01T00:00:00+00:00','2026-01-01T00:00:01+00:00','Completed',0.25,1,1)""")
            conn.execute("""INSERT INTO detections
                (id,analysis_id,media_id,class_name,confidence,x1,y1,x2,y2,evidence_path,state,created_at)
                VALUES('d1','a1','m1','pothole',0.8,1,1,10,10,'uploads/test_source.png','Pending','2026-01-01T00:00:01+00:00')""")
    response = client.post("/detections/d1/discard", data={"note":"test"}, follow_redirects=True)
    assert response.status_code == 200
    with road_app.app_context():
        with road_app.db() as conn:
            assert conn.execute("SELECT state FROM detections WHERE id='d1'").fetchone()[0] == "Discarded"
            assert conn.execute("SELECT COUNT(*) FROM detection_audit WHERE detection_id='d1'").fetchone()[0] == 1

def test_report_requires_fields_and_can_submit(client, app):
    road_app = app
    with road_app.app_context():
        with road_app.db() as conn:
            conn.execute("""INSERT INTO media
                (id,original_name,stored_name,media_type,mime_type,size_bytes,uploaded_at,status,width,height)
                VALUES('m2','road.png','source.png','image','image/png',100,'2026-01-01T00:00:00+00:00','Analysed',20,20)""")
            conn.execute("""INSERT INTO analyses
                (id,media_id,media_type,started_at,completed_at,status,confidence,frames_processed,detection_count)
                VALUES('a2','m2','image','2026-01-01T00:00:00+00:00','2026-01-01T00:00:01+00:00','Completed',0.25,1,1)""")
            conn.execute("""INSERT INTO detections
                (id,analysis_id,media_id,class_name,confidence,x1,y1,x2,y2,evidence_path,state,created_at)
                VALUES('d2','a2','m2','crack',0.75,1,1,10,10,'uploads/source.png','Pending','2026-01-01T00:00:01+00:00')""")
    response = client.post("/detections/d2/report", data={
        "damage_type":"crack","location":"Jalan Ujian","priority":"High",
        "recipient":"Road owner","description":"Visible road crack"
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b"Report submitted" in response.data
    with road_app.app_context():
        with road_app.db() as conn:
            report = conn.execute("SELECT * FROM reports WHERE detection_id='d2'").fetchone()
            assert report is not None
            assert report["status"] == "Submitted"
            assert conn.execute("SELECT state FROM detections WHERE id='d2'").fetchone()[0] == "Reported"

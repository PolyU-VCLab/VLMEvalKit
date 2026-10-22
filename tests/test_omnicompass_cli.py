"""Public CLI smoke with tiny generated media and a local HTTP fixture."""

import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import av
from PIL import Image


def _json(path, value):
    path.write_text(json.dumps(value))


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _release(root):
    (root / "videos").mkdir(parents=True)
    (root / "references").mkdir()
    video = root / "videos/fixture.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for i in range(8):
            frame = av.VideoFrame.from_image(Image.new("RGB", (96, 64), (i * 20, 80, 120)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    reference = root / "references/fixture.png"
    Image.new("RGB", (96, 64), (40, 80, 120)).save(reference)
    _jsonl(root / "videos.jsonl", [{
        "video_id": "fixture",
        "video_path": "videos/fixture.mp4",
        "duration_seconds": 2.0,
        "width": 96,
        "height": 64,
        "fps": {
            "num": 4,
            "den": 1
        },
        "has_audio": False,
        "tracks": ["VL"],
        "domain": "fixture",
        "sub_category": "fixture",
    }])
    questions = []
    for task, answer in [
        ("TG", {
            "start_time": 0.5,
            "end_time": 1.5
        }),
        ("SG", {
            "start_bbox": [100, 100, 800, 800],
            "end_bbox": [100, 100, 800, 800]
        }),
        ("QA", {
            "answer": "A"
        }),
    ]:
        row = {
            "question_id": f"fixture_{task}",
            "video_id": "fixture",
            "track": "VL",
            "task": task,
            "question": {
                setting: f"{task} fixture question"
                for setting in ("text", "text_image")
            },
            "reference_image": "references/fixture.png",
            "ground_truth": answer,
        }
        if task == "SG":
            row["given_interval"] = [0.5, 1.5]
        if task == "QA":
            row["options"] = {key: f"choice {key}" for key in "ABCDE"}
        questions.append(row)
    _jsonl(root / "questions.jsonl", questions)
    manifest = [{
        "public_path": str(p.relative_to(root)),
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    } for p in (reference, video)]
    _jsonl(root / "media_manifest.jsonl", manifest)
    _json(root / "archives.json", [{
        "path": "media/fixture.tar",
        "first_path": "references/fixture.png",
        "last_path": "videos/fixture.mp4",
        "bytes": 0,
        "sha256": "",
        "files": 2,
    }])
    for name in ("schema.json", "taxonomy.json"):
        _json(root / name, {})
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "videos.jsonl",
            "questions.jsonl",
            "schema.json",
            "taxonomy.json",
            "archives.json",
            "media_manifest.jsonl",
        )
    }
    _json(root / "release.json", {
        "schema_version": "1.1",
        "audio_delivery": "embedded_in_original_video",
        "files": files,
    })


def test_public_cli_json_fresh_and_resume(tmp_path):
    root = tmp_path / "data"
    _release(root)
    received = []

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            text = json.dumps(payload["messages"])
            answer = "A"
            if "TG fixture" in text:
                answer = "[0.5, 1.5]"
            elif "SG fixture" in text:
                answer = json.dumps({
                    "start_time": 0.5,
                    "end_time": 1.5,
                    "start_bbox": [100, 100, 800, 800],
                    "end_bbox": [100, 100, 800, 800],
                })
            body = json.dumps({
                "id": f"fixture-{len(received)}",
                "choices": [{
                    "message": {
                        "content": answer
                    },
                    "finish_reason": "stop"
                }],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    config = tmp_path / "config.json"
    _json(
        config, {
            "model": {
                "fixture": {
                    "class": "GPT4V",
                    "model": "fixture",
                    "key": "offline-fixture",
                    "api_base": f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    "max_tokens": 32768,
                    "token_limit_field": "max_tokens",
                    "preserve_generation_config": True,
                    "return_response_json": True,
                    "retry": 1,
                    "timeout": 10,
                    "wait": 0,
                }
            },
            "data": {
                "custom-track": {
                    "class": "OmniCompass",
                    "dataset": "OmniCompass_VL",
                    "nframe": 2,
                }
            },
        })
    repo = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNICOMPASS_")}
    env.update({
        "PYTHONPATH": str(repo),
        "OMNICOMPASS_DATA": str(root),
        "OMNICOMPASS_CACHE": str(tmp_path / "cache"),
        "OMNICOMPASS_MAX_DOWNLOAD_BYTES": "0",
        "LMUData": str(tmp_path / "lmudata"),
        "PRED_FORMAT": "json",
        "NO_PROXY": "127.0.0.1,localhost",
    })
    cmd = [
        sys.executable,
        "run.py",
        "--config",
        str(config),
        "--work-dir",
        str(tmp_path / "output"),
        "--api-nproc",
        "1",
    ]
    try:
        first = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=60)
        assert first.returncode == 0, first.stdout + first.stderr
        assert len(received) == 6
        assert all(row["max_tokens"] == 32768 for row in received)
        assert all(
            any(block["type"] == "image_url" for message in row["messages"] for block in message["content"]
                if isinstance(block, dict)) for row in received)
        reports = list((tmp_path / "output").rglob("*.omnicompass.json"))
        assert reports
        score = json.loads(reports[0].read_text())
        assert score["tracks"]["VL"]["paired_cs"]["overall"] == 1.0
        # Any accidental model request on resume now receives HTTP 503.
        Handler.do_POST = lambda self: (received.append(None), self.send_error(503))
        resumed = subprocess.run(
            cmd + ["--reuse", "--reuse-aux", "all"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert len(received) == 6
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)

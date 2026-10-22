"""Real process termination across the native video checkpoint entrypoint."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from vlmeval.dataset.omnicompass import OmniCompass
from vlmeval.inference_video import infer_data_job_video
from vlmeval.smp import load


class FixtureDataset(OmniCompass):

    def __init__(self):
        self.dataset_name = "OmniCompass_VL"
        self.protocol = SimpleNamespace(protocol_id="interruption-test")
        self.nframe, self.fps = 32, -1
        self.docs = [{
            "index": i,
            "question_id": f"q{i}",
            "setting": "text",
            "protocol_id": self.protocol.protocol_id,
            "answer": ""
        } for i in range(3)]
        self.data = pd.DataFrame(self.docs)

    def build_prompt(self, line, video_llm=False):
        return [{"type": "text", "value": str(line)}]


class FixtureModel:
    INTERLEAVE = True
    model_path = "fixture/qwen"
    max_new_tokens = 32768

    def __init__(self, root):
        self.root = root
        self.model = object()

    def generate(self, message, dataset=None):
        index = int(message[0]["value"])
        with (self.root / "calls.jsonl").open("a") as file:
            file.write(json.dumps(index) + "\n")
            file.flush()
            os.fsync(file.fileno())
        if index == 1 and os.getenv("INTERRUPT_FIXTURE") == "1":
            (self.root / "ready").touch()
            time.sleep(60)
        return f"answer-{index}"


def test_kill_and_resume_native_entrypoint(tmp_path):
    command = [sys.executable, __file__, "--worker", str(tmp_path)]
    env = {**os.environ, "INTERRUPT_FIXTURE": "1"}
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while not (tmp_path / "ready").exists():
                assert process.poll() is None, (tmp_path / "worker.log").read_text()
                assert time.monotonic() < deadline, "Worker did not reach interruption point"
                time.sleep(0.05)
        finally:
            process.kill()
            process.wait(timeout=10)
    resumed = subprocess.run(command,
                             capture_output=True,
                             text=True,
                             timeout=30,
                             env={
                                 **os.environ, "INTERRUPT_FIXTURE": "0"
                             })
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (tmp_path / "calls.jsonl").read_text().splitlines() == ["0", "1", "1", "2"]
    saved = load(str(tmp_path / "result.tsv"))
    assert saved.prediction.tolist() == ["answer-0", "answer-1", "answer-2"]


if __name__ == "__main__":
    root = Path(sys.argv[2])
    infer_data_job_video(FixtureModel(root),
                         str(root),
                         "fixture",
                         FixtureDataset(),
                         result_file=str(root / "result.tsv"))

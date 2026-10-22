"""OmniCompass with automatic Hub download and official scoring."""

import hashlib
import json
import math
import os
from numbers import Integral
from pathlib import Path

import pandas as pd
import portalocker

from vlmeval.dataset.video_base import VideoBaseDataset
from .utils.omnicompass.bridge import Protocol, evaluate, prediction
from .utils.omnicompass.run_identity import model_identity, runtime_identity


class OmniCompass(VideoBaseDataset):
    TYPE = "Video-VQA"
    DEFAULT_JUDGE_MODEL = None

    def __init__(self,
                 dataset="OmniCompass_VL",
                 nframe=32,
                 fps=-1,
                 data_root=None,
                 cache=None,
                 video_ids=None,
                 settings=("text", "text_image")):
        if dataset not in self.supported_datasets():
            raise ValueError(f"Unknown dataset: {dataset}")
        self.settings = tuple(settings)
        self.protocol = Protocol(
            data_root or os.getenv("OMNICOMPASS_DATA"),
            cache or os.getenv("OMNICOMPASS_CACHE"),
            nframe=nframe if nframe > 0 else None,
            fps=fps if fps > 0 else None,
        )
        track = dataset.rsplit("_", 1)[1]
        if video_ids is None and os.getenv("OMNICOMPASS_VIDEO_IDS"):
            requested = set(os.environ["OMNICOMPASS_VIDEO_IDS"].split(","))
            known = set(self.protocol.videos)
            if not requested <= known:
                raise ValueError("Unknown OMNICOMPASS_VIDEO_IDS")
            video_ids = sorted(requested & {q["video_id"] for q in self.protocol.questions if q["track"] == track})
        self.docs = self.protocol.documents(track, settings=settings, video_ids=video_ids)
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ["OmniCompass_VL", "OmniCompass_AV"]

    def prepare_dataset(self, dataset):
        import hashlib
        fingerprint = hashlib.sha256(json.dumps(self.docs, sort_keys=True).encode()).hexdigest()
        target = self.protocol.cache / "tables" / f"{dataset}-{fingerprint}.tsv"
        target.parent.mkdir(parents=True, exist_ok=True)
        records = [{
            **doc, "index": i,
            "question": json.loads(doc["question_json"])["question"][doc["setting"]]
        } for i, doc in enumerate(self.docs)]
        if not target.exists():
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, suffix=".tsv", delete=False) as file:
                pd.DataFrame(records).to_csv(file, sep="\t", index=False)
                temp = Path(file.name)
            temp.replace(target)
        return {"root": str(self.protocol.root), "data_file": str(target)}

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, Integral):
            line = self.data.iloc[int(line)]
        # Even video-capable models receive canonical frame images. Raw-video
        # forwarding would let a backend resample and change the reference budget.
        return self.protocol.content(dict(line))

    def validate_model(self, model):
        if not getattr(model, "INTERLEAVE", False):
            raise ValueError("OmniCompass requires an interleaved multi-image backend")
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(self.dataset_name):
            raise ValueError("Disable model custom prompts for OmniCompass")
        if self.dataset_name.endswith("_AV") and "audio" not in getattr(model, "allowed_types", []):
            raise ValueError("OmniCompass-AV requires explicit audio-block support")
        if self.dataset_name.endswith("_AV") and any(
                getattr(model, flag, False) for flag in ("use_vllm", "use_lmdeploy", "use_audio_in_video")):
            raise ValueError("Use the separate-audio Transformers path for this AV prototype")

    def is_reusable_prediction(self, output):
        if isinstance(output, float) and math.isnan(output):
            output = ""
        row = prediction({"question_id": "", "setting": "text"}, output)
        return row.get("error_kind") != "transport"

    @staticmethod
    def inference_model_config(model):
        """Fingerprint generation settings without serializing credentials or weights."""
        if isinstance(model, str):
            raise TypeError("Resolve the model before computing its inference configuration")
        model_class = type(model)
        fields = (
            "model",
            "model_path",
            "api_base",
            "max_tokens",
            "max_new_tokens",
            "enable_thinking",
            "thinking_parameter",
            "context_length",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "generate_kwargs",
            "min_pixels",
            "max_pixels",
            "total_pixels",
            "system_prompt",
            "post_process",
            "return_response_json",
            "img_detail",
            "image_format",
            "audio_format",
            "token_limit_field",
            "preserve_generation_config",
            "use_vllm",
            "use_lmdeploy",
            "revision",
            "attn_implementation",
            "do_sample",
            "img_size",
            "total_img_size",
            "max_file_size",
            "stream",
            "response_charset",
            "use_audio_in_video",
            "limit_mm_per_prompt",
        )
        values = {name: getattr(model, name) for name in fields if hasattr(model, name)}
        # Native wrappers hold loaded weights in model; model_path identifies them.
        identity = {
            name: values[name]
            for name in fields if name in values and (name != "model" or isinstance(values[name], str))
        }
        # Include all passthrough parameters without writing potential credentials
        # from custom provider options into the sidecar.
        provider = getattr(model, "default_kwargs", {})
        identity["provider_parameters_sha256"] = hashlib.sha256(json.dumps(provider,
                                                                           sort_keys=True).encode()).hexdigest()
        identity["cache_version"] = 3
        identity["class"] = f"{model_class.__module__}.{model_class.__qualname__}"
        identity["artifacts"] = model_identity(values.get("model_path"), values.get("revision"))
        identity["runtime"] = runtime_identity()
        config = getattr(values.get("model"), "config", None)
        if config is not None:
            identity["effective_dtype"] = str(getattr(values["model"], "dtype", None))
            identity["effective_attention"] = getattr(config, "_attn_implementation", None)
            identity["resolved_revision"] = getattr(config, "_commit_hash", None)
        return json.loads(json.dumps(identity, sort_keys=True))

    @staticmethod
    def inference_cache_files(model_name, dataset_alias_name):
        name = hashlib.sha256(f"{model_name}:{dataset_alias_name}".encode()).hexdigest()[:20]
        return {f".omnicompass-cache-{name}.json"}

    def validate_inference_cache(self, work_dir, model_name, result_file, dataset_alias_name, generation_config=None):
        """Refuse old checkpoints before their predictions can acquire new metadata."""
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        identity = {
            "protocol_id": self.protocol.protocol_id,
            "documents_sha256": hashlib.sha256(json.dumps(self.docs, sort_keys=True).encode()).hexdigest(),
            "model_name": model_name,
            "dataset_alias_name": dataset_alias_name,
        }
        if generation_config is not None:
            identity["generation"] = generation_config
            if Path(result_file).suffix == ".xlsx":
                raise ValueError("Use PRED_FORMAT=tsv to preserve full generation records")
        sidecar = root / next(iter(self.inference_cache_files(model_name, dataset_alias_name)))
        with portalocker.Lock(str(sidecar) + ".lock", timeout=60):
            if sidecar.exists():
                if json.loads(sidecar.read_text()) != identity:
                    raise ValueError("Inference cache configuration changed; use a fresh output directory")
            else:
                previous = list(root.glob(f"{model_name}_{dataset_alias_name}*.pkl"))
                previous += list(root.glob(f"*_{model_name}_{dataset_alias_name}*.pkl"))
                if previous:
                    raise ValueError("Unverified legacy checkpoints; use a fresh output directory")
                if Path(result_file).exists():
                    if generation_config is not None:
                        raise ValueError("Saved predictions have no verified generation configuration; "
                                         "use a fresh output directory")
                    from vlmeval.smp import load
                    saved = pd.DataFrame(load(str(result_file)))
                    if "protocol_id" not in saved or set(saved["protocol_id"]) != {identity["protocol_id"]}:
                        raise ValueError("Inference cache configuration changed; use a fresh output directory")
                    expected = {(d["question_id"], d["setting"]): d for d in self.docs}
                    records = saved.fillna("").to_dict(orient="records")
                    if len(records) != len(expected):
                        raise ValueError("Inference cache selection changed")
                    seen = set()
                    for row in records:
                        key = (row["question_id"], row["setting"])
                        if key in seen or key not in expected or any(
                                row.get(field) != value for field, value in expected[key].items()):
                            raise ValueError("Inference cache configuration changed")
                        seen.add(key)
                sidecar.write_text(json.dumps(identity, sort_keys=True))

    def evaluate(self, eval_file, **judge_kwargs):
        from vlmeval.smp import load
        data = load(str(eval_file))
        actual = data.to_dict(orient="records") if hasattr(data, "to_dict") else data
        expected = {(d["question_id"], d["setting"]): d for d in self.docs}
        seen, docs, outputs = set(), [], []
        for row in actual:
            key = (row["question_id"], row["setting"])
            if key not in expected or key in seen:
                raise ValueError(f"Unknown or duplicate output row: {key}")
            if row.get("protocol_id") != expected[key]["protocol_id"]:
                raise ValueError("Protocol mismatch in saved results; use a fresh output directory")
            seen.add(key)
            docs.append(expected[key])
            output = row["prediction"]
            outputs.append("" if isinstance(output, float) and math.isnan(output) else output)
        if seen != set(expected):
            raise ValueError("Missing result rows")
        result = evaluate(docs, outputs, self.settings)
        destination = Path(eval_file).with_suffix(".omnicompass.json")
        destination.write_text(json.dumps(result, indent=2))
        return result

    def validate_completed_predictions(self, result_file):
        from vlmeval.smp import load_prediction_file
        rows = load_prediction_file(str(result_file)).fillna("").to_dict(orient="records")
        expected = {(d["question_id"], d["setting"]) for d in self.docs}
        actual = [(row["question_id"], row["setting"]) for row in rows]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Incomplete prediction selection")
        if any(not self.is_reusable_prediction(row["prediction"]) for row in rows):
            raise ValueError("Incomplete transport results; resume to retry failed requests")

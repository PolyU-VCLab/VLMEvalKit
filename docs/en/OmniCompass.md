# OmniCompass

OmniCompass evaluates temporal grounding, spatial grounding and video question
answering with paired Text and Text+Image references. VL uses video frames;
AV adds the complete audio track. Both reference settings are required for CS.

[Paper](https://arxiv.org/abs/2610.18829) |
[Official code](https://github.com/PolyU-VCLab/OmniCompass) |
[Dataset](https://huggingface.co/datasets/VCLab-PolyU/OmniCompass) |
[Project page](https://omnicompass.netlify.app/)

## Data

Install VLMEvalKit and its requirements. Install FFmpeg for AV evaluation and
place it on PATH, or set `FFMPEG_BINARY` to its executable.

The adapter downloads pinned metadata and the media shards needed by the selected
examples. Shards contain multiple videos, so even a small selection can require
several GB. Set `OMNICOMPASS_MAX_DOWNLOAD_BYTES` to cap the size of each media
archive; `0` prevents media downloads.

To reuse an extracted release, set `OMNICOMPASS_DATA` to the directory containing
`release.json`, the metadata JSONL files, `videos/` and `references/`.
The adapter verifies checksums and keeps this directory read-only.
`OMNICOMPASS_CACHE` selects the separate frame, audio and download cache.

## API evaluation

Edit the model ID and full Chat Completions URL in
[`scripts/omnicompass_api_32k.json`](../../scripts/omnicompass_api_32k.json).
Choose the token-limit field supported by your provider: `max_tokens` or
`max_completion_tokens`. Keep `preserve_generation_config` and
`return_response_json` enabled.

Set the credential environment variable used by the wrapper for your provider.
The generic path uses `OPENAI_API_KEY`; named providers can use their dedicated
variables, such as `GOOGLE_API_KEY` for Gemini preview models.
Run from the repository root:

```bash
export OMNICOMPASS_VIDEO_IDS=oc_004
export PRED_FORMAT=tsv
python run.py --config scripts/omnicompass_api_32k.json \
  --work-dir outputs/omnicompass-vl --api-nproc 1
```

This selection runs six requests: TG, SG and QA under both reference settings.
Unset `OMNICOMPASS_VIDEO_IDS` for the full track. Keep model, sampling and
generation settings fixed when resuming:

```bash
python run.py --config scripts/omnicompass_api_32k.json \
  --work-dir outputs/omnicompass-vl --api-nproc 1 --reuse --reuse-aux all
```

Use the regular config path above. `--api-mode` bypasses the video integration
and is rejected for OmniCompass.

For AV, change the data entry to `dataset: "OmniCompass_AV"` and give it a distinct
alias such as `OmniCompass_AV_32frame`. Select an AV video, for example `oc_595`,
and use a separate work directory. The endpoint must support multiple images and
`input_audio` containing base64 MP3. Providers requiring an audio URL or a data URI
need a compatible backend.

## Local models

Replace the model entry with a supported interleaved backend. For example:

```json
{
  "class": "Qwen3VLChat",
  "model_path": "/path/to/Qwen3-VL-model",
  "use_custom_prompt": false,
  "use_vllm": false,
  "max_new_tokens": 32768,
  "return_response_json": true
}
```

For Qwen3-Omni AV, use its model path and also set `use_audio_in_video: false`.
This integration uses separate audio through Transformers. AV rejects vLLM,
LMDeploy and embedded-video audio modes. Local weight loading and GPU capacity
must match the selected model; CPU fixtures do not establish GPU compatibility.

For the Qwen3 wrapper, `enable_thinking` is optional and follows the actual chat
template. `context_length` optionally sets a smaller total context limit.
OmniCompass checks that input tokens plus the output budget fit the model.
Sampling and generation budgets are not reduced automatically.

## Outputs and recovery

Predictions use TSV by default for `OmniCompass_` aliases. JSON is also supported.
Set `PRED_FORMAT=tsv` or `json` for custom aliases. XLSX is rejected because it
can truncate long generation records.

The adjacent `.omnicompass.json` report contains task scores and paired CS.
Completed responses without a parseable final answer score zero and remain
completed on resume. Transport failures remain incomplete and are retried.
Original generation text is retained in structured responses.

The cache checks the question selection, protocol, resolved model settings,
local artifacts and runtime dependencies. Changed configurations or older cache
identities require a new work directory. Resuming a local model loads its weights
before validating the cache. A run containing OmniCompass exits nonzero if any
evaluation combination fails.

## Citation

```bibtex
@article{liu2026omnicompass,
  title={OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding},
  author={Liu, Xuyang and Wang, Shihao and Wei, Xinyu and Li, Minghan and Wang, Zeqing and Yang, Zuhao and Wen, Zichen and Zhang, Lei},
  journal={arXiv preprint arXiv:2610.18829},
  year={2026}
}
```

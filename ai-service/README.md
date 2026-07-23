# TCHS Defensive AI Service Foundation

This directory is the Python service foundation for a defense-first computer-vision system that will eventually run beside the existing Node/Express application. It is intentionally isolated: this phase does not connect to the Node application, load a model, train a model, or change existing prediction behavior.

## Architecture

YOLO will detect concrete objects visible in a frame: defensive players by position, the football, and officials. It should not directly determine a defensive front or coverage because those concepts depend on relationships among detections, player alignment, movement over time, and football context. Higher-level classifiers will derive those football labels later and a coach will verify the results.

The locked V1 object classes are:

| Index | Object class |
| ---: | --- |
| 0 | `defensive_end` |
| 1 | `defensive_tackle` |
| 2 | `middle_linebacker` |
| 3 | `inside_linebacker` |
| 4 | `outside_linebacker` |
| 5 | `cornerback` |
| 6 | `safety` |
| 7 | `football` |
| 8 | `official` |

For linebacker annotations, use `middle_linebacker` only for the single linebacker aligned as the defense's central Mike. Use `inside_linebacker` for other interior linebackers when the defense has two or more inside linebackers. Use `outside_linebacker` for linebackers outside the interior box or aligned on the edge. Assign each player exactly one linebacker class on a frame.

The six labels derived in later phases are `defensive_front`, `box_count`, `coverage_shell`, `blitz_look`, `corner_leverage`, and `safety_rotation`. They are not YOLO object classes.

The pipeline is:

1. Frame extraction
2. Object detection
3. Player alignment features
4. Coach-initiated temporal tracking (Stage 1 available)
5. Defensive-front classifier
6. Blitz classifier
7. Coverage classifier
8. Coach verification

No trained TCHS model exists in this phase. Optional local YOLO inference can suggest people and football on one paused frame, but exact defensive positions require coach classification unless a future custom model uses the locked class names. Stage 1 tracking uses classical OpenCV CSRT, KCF, or MOSSE only after a coach accepts and selects a box. Neural tracking, training, evaluation, model versioning, and model promotion remain future work.

## Setup

Run commands from the `ai-service` directory. Python 3.12 is required.

### macOS

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

The optional environment variables are `AI_SERVICE_HOST`, `AI_SERVICE_PORT`, `AI_DATASET_DIR`, and `AI_MODEL_DIR`. Relative dataset and model paths are resolved from this directory on macOS, Windows, and Linux.

## Optional current-frame vision

The foundation service runs without vision packages. In that state `/health` and `/model-status` still work, while `/detect/frame` returns a clean `503`. To prepare a Python 3.12 vision environment later:

```sh
python -m pip install -r requirements-vision.txt
```

This task does not install those packages or download weights. Configure:

- `AI_DETECTOR_MODEL` — local filename/path or an Ultralytics model identifier; default `yolo11n.pt`
- `AI_DETECTOR_CONFIDENCE` — default `0.35`
- `AI_DETECTOR_IOU` — default `0.50`
- `AI_DETECTOR_DEVICE` — default `auto`
- `AI_DETECTOR_MAX_DETECTIONS` — bounded to 1–500
- `AI_ALLOW_MODEL_DOWNLOAD` — default `false`

With downloads disabled, the model must exist under `ai-service`, under `ai-service/models`, or at an explicit absolute path. Missing weights return `503` and never trigger an automatic download. Setting `AI_ALLOW_MODEL_DOWNLOAD=true` explicitly permits Ultralytics to resolve or download the configured identifier on first detection.

Device `auto` selects CUDA when available, then Apple Silicon MPS when PyTorch reports it available, then CPU. The model loads lazily on the first detection request under a process lock; it is not loaded during import or startup.

`POST /detect/frame` accepts a single JPEG, PNG, or WebP multipart field named `image`, with optional `confidence` and `iou`. Generic models return only people and sports balls. People have no suggested defensive class and require coach review; sports balls map provisionally to `football`. Exact locked class names from a custom model map directly. Uploaded frames remain in memory and are not written to permanent disk.

`POST /track/frames` accepts an `initial_image`, 1–60 ordered `frames`, normalized accepted `boxes` JSON, and matching `frame_times` JSON. It selects the best available tracker in CSRT → KCF → MOSSE order. Each result retains the supplied player identity and class. A failed tracker is removed immediately and returns a failure record rather than an invented location. Uploaded frames remain memory-only.

## Run and test

Start the development service from `ai-service`:

```sh
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Then inspect:

- `http://127.0.0.1:8000/health` for service status and confirmation that `model_connected` is `false`.
- `http://127.0.0.1:8000/classes` for the nine indexed object classes.
- `http://127.0.0.1:8000/config` for the public dataset and class configuration.
- `http://127.0.0.1:8000/model-status` for honest dependency, weight, model, and device state.

Start both local services in separate terminals:

```sh
# ai-service/
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# repository root
DATABASE_URL=postgresql://localhost:5432/tchs_film \
DATA_DIR="$PWD/.local-data" \
AI_SERVICE_URL=http://127.0.0.1:8000 \
PORT=8080 node server.js
```

Do not commit football film, annotations, model weights, training runs, `.env` files, or credentials. Placeholder files keep the empty dataset, model, and run directories in version control while their future contents remain ignored.

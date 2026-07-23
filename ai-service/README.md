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

The planned pipeline is:

1. Frame extraction
2. Object detection
3. Player alignment features
4. Temporal tracking
5. Defensive-front classifier
6. Blitz classifier
7. Coverage classifier
8. Coach verification

No trained model exists in this phase. Ultralytics, PyTorch, torchvision, OpenCV, pretrained weights, training, evaluation, model versioning, and model promotion belong to a later model-integration phase.

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

## Run and test

Start the development service from `ai-service`:

```sh
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Then inspect:

- `http://127.0.0.1:8000/health` for service status and confirmation that `model_connected` is `false`.
- `http://127.0.0.1:8000/classes` for the nine indexed object classes.
- `http://127.0.0.1:8000/config` for the public dataset and class configuration.

Do not commit football film, annotations, model weights, training runs, `.env` files, or credentials. Placeholder files keep the empty dataset, model, and run directories in version control while their future contents remain ignored.

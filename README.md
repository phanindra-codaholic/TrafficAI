# TrafficAI

TrafficAI is a Python application that detects vehicles in intersection-camera images and produces a priority-ordered traffic-signal plan.

## Technology

- Python 3
- FastAPI
- YOLOv8 ONNX model
- ONNX Runtime
- OpenCV

## Features

- Detects bicycles, cars, motorcycles, buses, and trucks.
- Processes one to four camera images per request.
- Filters detections to the road region of interest.
- Ranks lanes by detected vehicle count and waiting-time weight.
- Calculates a green-light time of five seconds plus two seconds per vehicle, capped at 30 seconds.
- Supports emergency preemption in the scheduling engine.

## Run the command-line demo

```powershell
cd src
python main.py
```

## Run the API

```powershell
cd src
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the API interactively.

## API endpoints

- `GET /health` checks whether the detection model is ready.
- `POST /detect` accepts one to four images in the `cameras` form field and returns the traffic-signal plan.

# 🚦 TrafficAI – AI-Based Intelligent Traffic Signal Optimization

**TrafficAI** is an AI-powered traffic management system that analyzes vehicle density from intersection camera images and generates an optimized traffic signal schedule. It combines YOLOv8-based vehicle detection with priority-based scheduling algorithms to improve traffic flow and reduce unnecessary waiting time.

The system processes multiple road-camera images, identifies vehicles, calculates traffic density, and assigns green-light durations based on the number of detected vehicles and waiting-time priority.

## ✨ Key Features

- **AI-Based Vehicle Detection:** Uses a YOLOv8 ONNX model to detect cars, buses, trucks, motorcycles, and bicycles.
- **Multi-Camera Processing:** Supports up to four intersection camera images simultaneously.
- **Region of Interest (ROI):** Filters vehicle detections to focus on the relevant road area.
- **Traffic Density Analysis:** Calculates the number of detected vehicles for each road.
- **Priority-Based Scheduling:** Ranks roads based on vehicle count and waiting-time weighting.
- **Dynamic Green Signal Timing:** Allocates green-light durations according to traffic density, with a configurable maximum limit.
- **Emergency Priority Logic:** Includes emergency preemption support in the scheduling engine.
- **REST API Integration:** Provides a FastAPI interface for submitting camera images and retrieving optimized signal plans.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application development |
| YOLOv8 | Vehicle detection model |
| ONNX Runtime | AI model inference |
| OpenCV | Image processing and visualization |
| NumPy | Numerical operations |
| FastAPI | REST API development |
| Dijkstra's Algorithm | Not currently implemented |

## 🏗️ System Architecture

The project follows a modular processing pipeline:

1. **Camera Input:** Receives images from up to four road cameras.
2. **Vehicle Detection:** YOLOv8 processes images and identifies different vehicle classes.
3. **ROI Filtering:** Filters detections based on the predefined road region.
4. **Traffic Analysis:** Calculates vehicle density for each road.
5. **Priority Calculation:** Evaluates traffic density and waiting-time weights.
6. **Signal Optimization:** Generates a priority-based signal schedule with dynamically calculated green-light durations.

## ⚙️ Signal Optimization Logic

The scheduling engine uses the following logic:

**1. Priority Weight**

\[
W_i=V_i+\left\lfloor\frac{T_i}{10}\right\rfloor
\]

Where:
- \(V_i\) = Number of vehicles on road \(i\)
- \(T_i\) = Waiting time assigned to road \(i\)

**2. Green Signal Duration**

\[
G_i=\min(5+2V_i,\ 30)
\]

Where:
- \(G_i\) = Green signal duration in seconds
- \(V_i\) = Number of detected vehicles
- 5 seconds = Base green-light duration
- 2 seconds = Additional time per vehicle
- 30 seconds = Maximum green-light duration

**Note:** These are configurable scheduling parameters, not timings validated through real-world traffic deployment.

## 📁 Project Structure

```text
TrafficAI/
│
├── data/
│   ├── cam1.png
│   ├── cam2.png
│   ├── cam3.png
│   └── cam4.png
│
├── models/
│   └── yolov8n_traffic.onnx
│
├── src/
│   ├── ai/
│   │   └── detector.py
│   │
│   ├── control/
│   │   ├── optimization_engine.py
│   │   └── traffic_manager.py
│   │
│   ├── api.py
│   └── main.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/TrafficAI.git
cd TrafficAI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Command-Line Demo

```bash
cd src
python main.py
```

The application processes the four sample camera images and displays the generated traffic signal schedule.

### 5. Run the FastAPI Server

From the `src` directory:

```bash
uvicorn api:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks application and AI model status |
| POST | `/detect` | Processes camera images and returns an optimized signal plan |

The `/detect` endpoint accepts between one and four images through the `cameras` form field.

### Sample API Response

```json
{
  "total_cameras": 4,
  "total_vehicles": 35,
  "cycle_time": 90,
  "signal_plan": [
    {
      "road_id": 101,
      "vehicle_count": 12,
      "green_time": 29,
      "priority_rank": 1,
      "is_emergency": false
    }
  ]
}
```

*Illustrative response structure; values are examples.*

## 🔮 Future Improvements

- Integrate real-time CCTV camera feeds.
- Implement automatic emergency vehicle detection and priority handling.
- Improve traffic prediction using historical traffic data.
- Introduce adaptive signal timing based on real-time traffic conditions.
- Extend the system to coordinate multiple intersections.
- Develop a graphical dashboard for traffic monitoring.

## 🎯 Project Objective

The primary objective of TrafficAI is to demonstrate how computer vision and intelligent scheduling algorithms can be combined to develop a smarter traffic management system.

By moving beyond fixed-time traffic signals and considering actual vehicle density, the system provides a foundation for adaptive and more efficient urban traffic management.

---

**Developed as an AI + Data Structures and Algorithms project.**

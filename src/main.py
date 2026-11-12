"""TrafficAI command-line demo."""

import sys
import cv2
from control.traffic_manager import TrafficManager


def main():
    print("--- TrafficAI Booting Up (Smart Cycle Scheduling) ---")

    try:
        manager = TrafficManager(model_path="../models/yolov8n_traffic.onnx")

        camera_paths = [
            "../data/cam1.png",
            "../data/cam2.png",
            "../data/cam3.png",
            "../data/cam4.png",
        ]

        cameras = []
        for i, path in enumerate(camera_paths):
            img = cv2.imread(path)
            if img is None:
                print(f"FATAL ERROR: Camera {i + 1} is offline!")
                return 1
            cameras.append(img)

        print("All cameras online. AI Processing intersection...")

        cycle_plan = manager.process_intersection(cameras)

        print()
        print("=" * 56)
        print("           TRAFFIC OPTIMIZATION CYCLE PLAN")
        print("=" * 56)
        print(
            f"{'Priority':<12}{'Road ID':<12}{'Vehicles':<15}"
            f"{'Green Time':<15}Status"
        )
        print("-" * 56)

        for dec in cycle_plan:
            status = "[EMERGENCY OVERRIDE]" if dec.is_emergency else "Normal"
            print(
                f"{'#' + str(dec.priority_rank):<12}"
                f"{'Road ' + str(dec.lane_id):<12}"
                f"{dec.vehicle_count:<15}"
                f"{str(dec.green_time) + 's':<15}"
                f"{status}"
            )

        print("=" * 56)
        print("Max Green Time Constraint: 30s per lane.\n")

        try:
            cv2.imshow("Camera 1 (Road 101)", cameras[0])
            cv2.imshow("Camera 2 (Road 102)", cameras[1])
            cv2.imshow("Camera 3 (Road 103)", cameras[2])
            cv2.imshow("Camera 4 (Road 104)", cameras[3])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            for i, cam in enumerate(cameras):
                out_path = f"../data/result_cam{i+1}.png"
                cv2.imwrite(out_path, cam)
            print("(GUI not available — saved annotated images to data/result_cam*.png)")


    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

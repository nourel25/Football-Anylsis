import pickle
import cv2
import numpy as np
import os

class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = first_frame_grayscale.shape

        mask_features = np.zeros_like(first_frame_grayscale)

        left_w = int(0.05 * w)
        right_w = int(0.05 * w)
        mask_features[:, :left_w] = 1
        mask_features[:, w - right_w:] = 1

        top_h = int(0.05 * h)
        bottom_h = int(0.05 * h)
        mask_features[:top_h, :] = 1
        mask_features[h - bottom_h:, :] = 1

        self.features = dict(
            maxCorners=200,
            qualityLevel=0.3,
            minDistance=5,
            blockSize=7,
            mask=mask_features,
        )

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            try:
                with open(stub_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                print("⚠️ Stub corrupted, recomputing camera movement")

        camera_movement = [[0, 0]] * len(frames)
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) < 10:
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

            if old_features is None:
                continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is None or status is None:
                continue

            valid_old = old_features[status.flatten() == 1].reshape(-1, 2)
            valid_new = new_features[status.flatten() == 1].reshape(-1, 2)

            if valid_old.shape[0] == 0 or valid_new.shape[0] == 0:
                continue

            dx = valid_new[:, 0] - valid_old[:, 0]
            dy = valid_new[:, 1] - valid_old[:, 1]

            median_dx = float(np.median(dx))
            median_dy = float(np.median(dy))

            if (
                abs(median_dx) > self.minimum_distance
                or abs(median_dy) > self.minimum_distance
            ):
                camera_movement[frame_num] = [median_dx, median_dy]
            else:
                camera_movement[frame_num] = camera_movement[frame_num - 1]

            old_gray = frame_gray.copy()
            old_features = valid_new.reshape(-1, 1, 2)

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            cv2.putText(frame, f"Camera X: {x_movement:.2f}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            cv2.putText(frame, f"Camera Y: {y_movement:.2f}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            output_frames.append(frame)
        return output_frames

    def add_adjusted_positions(self, tracks, camera_movement_per_frame):
        for object_name, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    if frame_num == 0:
                        adjusted_position = track["position"]
                    else:
                        dx, dy = camera_movement_per_frame[frame_num]
                        adjusted_position = (
                            track["position"][0] - dx,
                            track["position"][1] - dy,
                        )
                    tracks[object_name][frame_num][track_id]["adjusted_position"] = adjusted_position

import pickle
import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
import supervision as sv

from utils.bbox_utils import (
    get_bbox_center,
    get_bbox_width,
    get_foot_position
)

class Tracker:
    def __init__(self, model_path):
        # YOLO model for detections
        self.model = YOLO(model_path)

        # ByteTrack (supervision wrapper)
        self.tracker = sv.ByteTrack(
           track_activation_threshold=0.25,
           minimum_matching_threshold=0.8,
           lost_track_buffer=30
        )

    def interpolate_ball_positions(self, ball_positions, preserve_size=True, default_box_size=8):
        """
        - ball_positions: list of dicts like [{1: {'bbox': [x1,y1,x2,y2]}}] (one per frame)
        - returns same format, interpolated, length preserved
        - preserve_size: if True, uses median bbox size from available detections to reconstruct boxes
        """
        # Extract bboxes and compute centers & sizes (width, height)
        centers = []
        widths = []
        heights = []
        for fr in ball_positions:
            bbox = fr.get(1, {}).get('bbox', [])
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                centers.append([cx, cy])
                widths.append(w)
                heights.append(h)
            else:
                centers.append([np.nan, np.nan])
                widths.append(np.nan)
                heights.append(np.nan)

        df = pd.DataFrame(centers, columns=["cx", "cy"])
        df_w = pd.Series(widths, name="w")
        df_h = pd.Series(heights, name="h")

        # Interpolate centers and sizes with linear then forward/backward fill
        df[['cx', 'cy']] = df[['cx', 'cy']].interpolate(method='linear', limit_direction='both')
        df_w = df_w.interpolate(method='linear', limit_direction='both')
        df_h = df_h.interpolate(method='linear', limit_direction='both')

        # If still NaN (all frames missing), use defaults
        if df[['cx','cy']].isna().any().any():
            # put zeros to avoid failing downstream
            df[['cx','cy']] = df[['cx','cy']].fillna(0.0)
        if df_w.isna().all():
            df_w = df_w.fillna(default_box_size * 2)
        if df_h.isna().all():
            df_h = df_h.fillna(default_box_size * 2)

        # If preserve_size -> use median of known sizes for frames where interpolation produced NaN or tiny sizes
        known_w = df_w[~np.isnan(df_w)]
        known_h = df_h[~np.isnan(df_h)]
        median_w = float(np.median(known_w)) if len(known_w) > 0 else default_box_size * 2
        median_h = float(np.median(known_h)) if len(known_h) > 0 else default_box_size * 2

        # Reconstruct boxes
        output = []
        for idx, row in df.iterrows():
            cx = float(row["cx"])
            cy = float(row["cy"])
            w = float(df_w.iloc[idx]) if not np.isnan(df_w.iloc[idx]) else median_w
            h = float(df_h.iloc[idx]) if not np.isnan(df_h.iloc[idx]) else median_h

            # Ensure positive sizes
            w = max(w, 1.0)
            h = max(h, 1.0)

            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            bbox = [x1, y1, x2, y2]
            output.append({1: {"bbox": bbox}})

        return output

    def detect_frames(self, frames, batch_size=20, conf=0.1):
        """
        Predict in batches. Returns list of YOLO Result objects (one per frame).
        Defensive: catches exceptions and returns empty detections for failed batches.
        """
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            try:
                results = self.model.predict(batch, conf=conf, verbose=False)
                # results may be a list-like with length == batch_size
                for r in results:
                    detections.append(r)
            except Exception as e:
                # If detection fails, append empty placeholder results
                print(f"[Tracker] YOLO predict failed on batch starting at {i}: {e}")
                for _ in batch:
                    # create an empty ultralytics-compatible object using dict interface if possible
                    detections.append(sv.Detections.from_ultralytics(None) if hasattr(sv.Detections, 'from_ultralytics') else None)
        # pad if lengths differ
        if len(detections) < len(frames):
            detections.extend([None] * (len(frames) - len(detections)))
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Returns tracks dict with keys: 'players', 'referees', 'ball'
        Each is a list of length len(frames), each list element is a dict mapping track_id->properties
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections_per_frame = self.detect_frames(frames)

        # initialize empty structure
        tracks = {
            'players': [{} for _ in range(len(frames))],
            'referees': [{} for _ in range(len(frames))],
            'ball': [{} for _ in range(len(frames))]
        }

        for frame_num, detection in enumerate(detections_per_frame):
            if detection is None:
                # no detection available for this frame
                continue

            # Convert YOLO → Supervision format
            try:
                detection_supervision = sv.Detections.from_ultralytics(detection)
            except Exception:
                # fallback: skip frame
                print(f"[Tracker] Failed to convert detection for frame {frame_num}")
                continue

            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Reassign goalkeeper -> player if present
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                try:
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[object_idx] = cls_names_inv.get("player", class_id)
                except Exception:
                    # skip if mapping fails
                    pass

            # Update tracker; guard exceptions
            try:
                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            except Exception as e:
                print(f"[Tracker] tracker.update_with_detections failed on frame {frame_num}: {e}")
                detection_with_tracks = []

            # parse tracked detections (supervision output format)
            # detection_with_tracks: iterable of tuples/arrays in format used by supervision
            for frame_detection in detection_with_tracks:
                try:
                    bbox = list(map(float, frame_detection[0].tolist()))
                    cls_id = int(frame_detection[3])
                    track_id = int(frame_detection[4])
                except Exception:
                    continue

                # sanitize bbox: x1,y1,x2,y2
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                # correct possible inverted coords
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                bbox_clean = [x1, y1, x2, y2]

                # Map class ids to names safely
                cls_name = cls_names.get(cls_id, None)
                if cls_name == "player" or cls_name == "goalkeeper":
                    tracks['players'][frame_num][track_id] = {'bbox': bbox_clean}
                elif cls_name == "referee":
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox_clean}
                else:
                    # unknown class tracked; ignore
                    pass

            # Also collect detected (not tracked) ball(s) from detection_supervision
            try:
                for det in detection_supervision:
                    bbox = det[0].tolist() if hasattr(det[0], 'tolist') else det[0]
                    cls_id = int(det[3])
                    cls_name = cls_names.get(cls_id, None)
                    if cls_name == 'ball':
                        # use the canonical ball id 1
                        x1, y1, x2, y2 = map(float, bbox)
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        tracks['ball'][frame_num][1] = {'bbox': [x1, y1, x2, y2]}
            except Exception:
                # detection_supervision iteration may fail; skip
                pass

        # final safety: ensure each frame has dicts (already initialized), and optionally fill missing ball frames with empty dict
        for i in range(len(frames)):
            if tracks['players'][i] is None:
                tracks['players'][i] = {}
            if tracks['referees'][i] is None:
                tracks['referees'][i] = {}
            if tracks['ball'][i] is None:
                tracks['ball'][i] = {}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            return frame

        # clamp coordinates inside frame
        h, w = frame.shape[:2]
        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1))
        y2 = max(0, min(h-1, y2))

        y2_bottom = y2
        x_center, _ = get_bbox_center([x1, y1, x2, y2])
        width = max(3, int(get_bbox_width([x1, y1, x2, y2])))

        cv2.ellipse(
            frame,
            center=(int(x_center), int(y2_bottom)),
            axes=(int(width), int(max(2, 0.35 * width))),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=tuple(map(int, color)),
            thickness=2,
            lineType=cv2.LINE_4
        )

        # draw id rectangle if asked
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = int(x_center - rectangle_width // 2)
            x2_rect = int(x_center + rectangle_width // 2)
            y1_rect = int((y2_bottom - rectangle_height // 2) + 15)
            y2_rect = int((y2_bottom + rectangle_height // 2) + 15)

            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), tuple(map(int, color)), cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_traingle(self, frame, bbox, color):
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            return frame

        # clamp
        h, w = frame.shape[:2]
        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1))
        y2 = max(0, min(h-1, y2))

        x, _ = get_bbox_center([x1, y1, x2, y2])
        y = int(y1)

        triangle_points = np.array([
            [int(x), int(y)],
            [int(x-10), int(y-20)],
            [int(x+10), int(y-20)],
        ])
        cv2.drawContours(frame, [triangle_points], 0, tuple(map(int, color)), cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # defensively position box so it will be on-screen for any resolution
        box_x1 = int(0.7 * w)
        box_x2 = int(0.98 * w)
        box_y1 = int(0.78 * h)
        box_y2 = int(0.92 * h)

        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # defensive: ensure team_ball_control is numpy array-like
        try:
            tbc = np.asarray(team_ball_control)
        except Exception:
            tbc = np.array([])

        # Extract up to this frame (safe slicing)
        if frame_num < 0:
            frame_idx = 0
        else:
            frame_idx = min(frame_num, len(tbc) - 1) if len(tbc) > 0 else -1

        if frame_idx == -1:
            # nothing assigned yet
            team_1, team_2 = 0.0, 0.0
        else:
            team_ball_control_till_frame = tbc[:frame_idx + 1]
            # drop unassigned (-1)
            valid = team_ball_control_till_frame[team_ball_control_till_frame != -1]
            if valid.size == 0:
                team_1, team_2 = 0.0, 0.0
            else:
                team_1 = float(np.sum(valid == 1)) / float(valid.size)
                team_2 = float(np.sum(valid == 2)) / float(valid.size)

        # Draw text (positions scaled)
        text_x = box_x1 + 20
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",
                    (text_x, box_y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",
                    (text_x, box_y1 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return frame

    def draw_team_pass_count(self, frame, team_pass_count):
        overlay = frame.copy()
        h, w = frame.shape[:2]

        box_x1 = int(0.02 * w)
        box_x2 = int(0.35 * w)
        box_y1 = int(0.78 * h)
        box_y2 = int(0.92 * h)

        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # team_pass_count can be a timeline list or a single dict - handle both
        if isinstance(team_pass_count, dict):
            passes = team_pass_count
        elif isinstance(team_pass_count, (list, tuple, np.ndarray)):
            # expect dict per frame
            # if empty or short, default to zeros
            try:
                # if it's an array of dicts
                passes = team_pass_count if isinstance(team_pass_count, dict) else team_pass_count
            except Exception:
                passes = {1: 0, 2: 0}
        else:
            passes = {1: 0, 2: 0}

        # If passes is a list-like, caller should pass the correct frame dict in draw_annotations.
        # Here we gracefully access expected keys
        count1 = passes.get(1, 0) if isinstance(passes, dict) else 0
        count2 = passes.get(2, 0) if isinstance(passes, dict) else 0

        cv2.putText(frame, f"Team 1 Passes: {count1}", (box_x1 + 20, box_y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2 Passes: {count2}", (box_x1 + 20, box_y1 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return frame

    def draw_annotations(self, frames, tracks, team_ball_control, team_pass_count):
        output_frames = []
        num_frames = len(frames)

        # Ensure tracks structure shaped correctly
        players_list = tracks.get('players', [{} for _ in range(num_frames)])
        refs_list = tracks.get('referees', [{} for _ in range(num_frames)])
        ball_list = tracks.get('ball', [{} for _ in range(num_frames)])

        # Ensure team_pass_count is indexable per frame if it is a timeline
        # If team_pass_count is a single dict, we'll pass it directly each frame
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = players_list[frame_num] if frame_num < len(players_list) else {}
            referee_dict = refs_list[frame_num] if frame_num < len(refs_list) else {}
            ball_dict = ball_list[frame_num] if frame_num < len(ball_list) else {}

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))
                bbox = player.get('bbox', None)
                if bbox:
                    frame = self.draw_ellipse(frame, bbox, color, track_id=track_id)
                if player.get('has_ball', False) and bbox:
                    frame = self.draw_traingle(frame, bbox, (0, 0, 255))

            # Draw Referees
            for _, referee in referee_dict.items():
                bbox = referee.get('bbox', None)
                if bbox:
                    frame = self.draw_ellipse(frame, bbox, (0, 255, 255))

            # Draw Ball
            for _, ball in ball_dict.items():
                bbox = ball.get('bbox', None)
                if bbox:
                    frame = self.draw_traingle(frame, bbox, (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            # Draw Pass Counts
            # if team_pass_count is a timeline list-like, index it per frame; otherwise pass dict directly
            if isinstance(team_pass_count, (list, tuple, np.ndarray)):
                # guard index
                if frame_num < len(team_pass_count):
                    per_frame_pass = team_pass_count[frame_num]
                    if isinstance(per_frame_pass, dict):
                        frame = self.draw_team_pass_count(frame, per_frame_pass)
                    else:
                        # unexpected structure - draw zeros
                        frame = self.draw_team_pass_count(frame, {1: 0, 2: 0})
                else:
                    frame = self.draw_team_pass_count(frame, {1: 0, 2: 0})
            elif isinstance(team_pass_count, dict):
                frame = self.draw_team_pass_count(frame, team_pass_count)
            else:
                frame = self.draw_team_pass_count(frame, {1: 0, 2: 0})

            output_frames.append(frame)

        return output_frames

    def add_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    bbox = track.get('bbox')
                    if not bbox or len(bbox) != 4:
                        # skip invalid bbox
                        continue
                    if object_name == 'ball':
                        position = get_bbox_center(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_name][frame_num][track_id]['position'] = position

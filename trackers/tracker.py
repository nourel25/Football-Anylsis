from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd
from utils.bbox_utils import ( 
    get_bbox_center, 
    get_bbox_width,     
    get_foot_position
)

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
           track_activation_threshold=0.25,  # confidence threshold
           minimum_matching_threshold=0.8,   # IoU matching threshold
           lost_track_buffer=30              # how long to keep lost tracks
        )

    def interpolate_ball_positions(self, ball_positions):
        # extract ball bounding boxes (or empty list if missing)
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,
                                        columns=['x1', 'y1', 'x2', 'y2'])
        
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            yolo_results = self.model.predict(frames[i:i+batch_size], conf=0.1, verbose=False)
            detections += yolo_results

        return detections
    
    def get_object_tracks(self, frames,  read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections_per_frame = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': []
        }

        for frame_num, detection in enumerate(detections_per_frame):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert YOLO → Supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Reassign goalkeeper -> player
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_idx] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # Initialize per-frame dicts
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            # players & referees 
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                '''
                    tracks["players"][0] = {
                        7: {"bbox": [100, 200, 150, 300]},   # player ID 7
                        12: {"bbox": [400, 220, 460, 330]}   # player ID 12
                    }
                '''

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}

                elif cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}


            # ball (no track_id ball is often too small or fast for tracking)
            # record it by frame 
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # bottom of the box (near the feet)
        y2 = int(bbox[3])
        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height= 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
        
    def draw_traingle(self,frame,bbox,color):
        y = int(bbox[1])
        x, _ = get_bbox_center(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                      (1350, 850), (1900, 970),
                      (255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Extract ball control up to this frame
        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        # Count possession
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # Calculate possession percentages
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)
        
        # Draw text
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", 
                    (1400,900), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", 
                    (1400,950), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,0), 3)
        
        return frame

    def draw_team_pass_count(self, frame, team_pass_count):
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 850), (600, 970), (255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, f"Team 1 Passes: {team_pass_count[1]}", 
                    (70, 900), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Passes: {team_pass_count[2]}", 
                    (70, 950), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,0), 3)

        return frame

    def draw_annotations(self, frames, tracks, team_ball_control, team_pass_count):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id=track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player['bbox'], (0,0,255))


            # Draw Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0,255,255))

            # Draw Ball 
            for _, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0,255,0))

            # Drew Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            # Draw Pass Counts (static, doesn’t change frame by frame)
            frame = self.draw_team_pass_count(frame, team_pass_count[frame_num])
            
            output_frames.append(frame)

        return output_frames

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    bbox = track['bbox']
                    if object == 'ball':
                        position = get_bbox_center(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
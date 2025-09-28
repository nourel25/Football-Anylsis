import pickle
import cv2
import numpy as np
from utils.bbox_utils import measure_distance, measure_xy_distance
import os

'''

If we track players without correcting for camera movement, 
it looks like players move more than they really do.

To fix this → detect stable points (field edges, lines, ads)
→ track how they move frame to frame → that is camera motion.

Then subtract that motion from players/ball 
so their movement is relative to the pitch, not the camera.

'''


class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners = 100,     # detect up to 100 corners (features)
            qualityLevel = 0.3,   # discarding teh weakest 30%
            minDistance = 3,      # min distance between detected corners
            blockSize = 7,        # neighborhood size to check for corner quality
            mask = mask_features  # only look in certain areas of the frame
        )

        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2, 
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                10, 0.03
            )
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):

        # read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)
            

        camera_movement = [[0, 0]] * len(frames)

        # detects strong corner points in first frame
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # tracking the detectd corners (features)
            # calcOpticalFlowPyrLK plays “spot the difference” in the next frame to see where those dots moved
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray,           # prev frame
                frame_gray,         # current frame
                old_features,       # points we want to track       
                None,
                **self.lk_params 
            )

            # compare the old vs new
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # loop through tracked features
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    # store that as the estimated camera movement
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)
        
        return camera_movement
        
    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames
    
    def add_adjusted_positions(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    if frame_num == 0:
                        adjusted_position = track['position']
                    else:
                        x_movement, y_movement = camera_movement_per_frame[frame_num]
                        adjusted_position = (
                            track['position'][0] - x_movement,
                            track['position'][1] - y_movement
                        )
                    tracks[object][frame_num][track_id]['adjusted_position'] = adjusted_position

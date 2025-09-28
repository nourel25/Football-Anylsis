import numpy as np
import cv2

"""
ViewTransformer
---------------
Maps video pixel coordinates into real-world pitch coordinates (in meters).

Uses a perspective transform (homography) from 4+ reference points.
You should calibrate by selecting corner points (or known landmarks like penalty box corners)
in the video frame and matching them to their true positions on the field.
"""

class ViewTransformerModified:
    def __init__(self):
        # Standard football pitch dimensions (in meters)
        self.pitch_length = 105.0   # meters (goal to goal)
        self.pitch_width = 68.0     # meters (sideline to sideline)

        # Example: four corner points of the visible pitch in video (x,y in pixels).
        # You should replace with actual detected/calibrated points.
        self.pixel_vertices = np.array([
            [100, 1000],   # bottom-left
            [300, 200],    # top-left
            [1600, 200],   # top-right
            [1900, 1000],  # bottom-right
        ], dtype=np.float32)

        # Corresponding real-world coordinates (x,y in meters).
        # Convention: (0,0) = top-left corner of pitch
        self.target_vertices = np.array([
            [0, self.pitch_width],            # bottom-left
            [0, 0],                           # top-left
            [self.pitch_length, 0],           # top-right
            [self.pitch_length, self.pitch_width], # bottom-right
        ], dtype=np.float32)

        # Compute homography (perspective transform)
        self.perspective_transform = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        """
        Transform a single (x,y) pixel coordinate into real-world (meters).
        Returns None if the point lies outside the calibration area.
        """
        if point is None:
            return None

        p = (int(point[0]), int(point[1]))
        is_point_in_view = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_point_in_view:
            return None

        reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped_point, self.perspective_transform)
        return transformed.reshape(-1, 2)[0]  # return (x,y) in meters

    def add_transformed_positions(self, tracks):
        """
        For each tracked object (players/ball), add its real-world position
        in meters to the dictionary under key: 'transformed_position'.
        """
        for object_name, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    position = track.get('adjusted_position', None)
                    if position is None:
                        continue

                    position = np.array(position)
                    transformed_position = self.transform_point(position)
                    if transformed_position is not None:
                        transformed_position = transformed_position.tolist()

                    tracks[object_name][frame_num][track_id]['transformed_position'] = transformed_position
        return tracks

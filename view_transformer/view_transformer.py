import numpy as np
import cv2

'''
The idea is to take player/ball positions in pixel coordinates (from the video) 
and map them into real-world coordinates (meters on the football pitch)
using a perspective transformation (homography)
'''

class ViewTransformer:
    def __init__(self):
        # Detect it manually for now
        court_width = 68
        court_length = 23.32

        # Detect it manually for now
        self.pixel_vertices = np.array(
            [[110, 1035], 
             [265, 275], 
             [910, 260], 
             [1640, 915]]
        )

        self.target_vertices = np.array(
            [[0, court_length],
             [0, 0],
             [court_width, 0],
             [court_width, court_length]]
        )

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transform = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_point_in_view = cv2.pointPolygonTest(
            self.pixel_vertices, p, False
        ) >= 0 
        if not is_point_in_view:
            return None 
        
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(
            reshaped_point, self.perspective_transform
        )
        return transform_point.reshape(-1, 2)
    
    
    def add_transformed_positions(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    position = track['adjusted_position']
                    position = np.array(position)
                    transformed_position = self.transform_point(position)
                    if transformed_position is not None:
                        transformed_position = transformed_position.squeeze().tolist()
                    tracks[object][frame_num][track_id]['transformed_position'] = transformed_position

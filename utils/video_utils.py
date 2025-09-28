import cv2

def read_video(video_path):
    # Create video capture object to read frames
    cap = cv2.VideoCapture(video_path)
    frames =[]
    # Read frames per second
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    while True:
        ret, frame = cap.read()
        if not ret: # ret is False if reached the end
            break
        # each frame is numpy array representing an image
        frames.append(frame) 
    cap.release()
    return frames, fps


def save_video(output_video_frames, output_video_path, fps=24):
    h, w, _ = output_video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    # Creates a VideoWriter object that will save frames to output_video_path 
    # using: FPS and h & w
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    # Loops over each frame in the list and writes it into the output video
    for frame in output_video_frames:
        out.write(frame)
    out.release()

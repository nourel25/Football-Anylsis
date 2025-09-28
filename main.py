import numpy as np
import cv2

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer_modified import ViewTransformerModified
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from heatmap_of_positions import (
    compute_player_heatmap,
    compute_team_heatmap,
    plot_heatmap_on_pitch
)
from pass_network import (
    build_pass_network, 
    build_pass_network_graph,
    compute_player_average_positions,
    plot_pass_network
)


def get_team_pass_count(tracks):
    """Count the number of successful passes per team with 2-frame confirmation."""
    team_pass_count = {1: 0, 2: 0}
    pass_timeline = []

    prev_player, prev_team = None, None
    last_candidate, candidate_team = None, None
    hold_counter = 0
    hold_threshold = 3  # must hold ball for 3 frames

    for frame_players in tracks["players"]:
        current_player, current_team = None, None

        # find who has the ball
        for player_id, pdata in frame_players.items():
            if pdata.get("has_ball", False):
                current_player, current_team = player_id, pdata["team"]
                break

        if current_player is not None:
            if last_candidate == current_player:
                hold_counter += 1
            else:
                last_candidate = current_player
                candidate_team = current_team
                hold_counter = 1

            if hold_counter >= hold_threshold:
                # Only count as possession change after threshold
                if (
                    prev_player is not None
                    and prev_player != current_player
                    and prev_team == current_team
                ):
                    team_pass_count[current_team] += 1

                prev_player, prev_team = current_player, current_team

        pass_timeline.append(team_pass_count.copy())

    return pass_timeline


def assign_teams(video_frames, tracks):
    team_assigner = TeamAssigner()

    # find first frame with enough players
    for f_idx, player_track in enumerate(tracks['players']):
        if len(player_track) >= 2:
            team_assigner.assign_team_color(video_frames[f_idx], player_track)
            break

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            if team in team_assigner.team_colors:
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    return tracks


def assign_ball_possession(tracks):
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    last_team = -1

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox", None)
        assigned_player = player_assigner.assign_ball(player_track, ball_bbox) if ball_bbox else -1

        if assigned_player != -1:
            team = tracks["players"][frame_num][assigned_player].get("team", -1)
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(team)
            last_team = team
        else:
            team_ball_control.append(last_team)

    return np.array(team_ball_control)


def estimate_camera_movement(video_frames, tracks):
    """Estimate and adjust for camera movement across frames."""
    estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path='stubs_video_1/camera_movement_stub.pkl'
    )
    estimator.add_adjusted_positions(tracks, camera_movement_per_frame)
    return camera_movement_per_frame, tracks, estimator


def transform_positions(tracks):
    """Transform player and ball positions from pixel to real-world coordinates."""
    transformer = ViewTransformerModified()
    transformer.add_transformed_positions(tracks)
    return tracks


def estimate_speed_and_distance(tracks):
    """Add speed and distance metrics to tracks."""
    estimator = SpeedAndDistanceEstimator()
    estimator.add_speed_and_distance_to_tracks(tracks)
    return tracks, estimator


def main():
    # --- 1. Read video ---
    video_frames, fps = read_video('input_videos/08fd33_4.mp4')

    # --- 2. Tracking ---
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path='stubs_video_1/track_stubs.pkl'
    )
    tracker.add_position_to_tracks(tracks)

    # --- 3. Camera movement ---
    camera_movement_per_frame, tracks, camera_estimator = estimate_camera_movement(video_frames, tracks)

    # --- 4. Transform positions ---
    tracks = transform_positions(tracks)

    # --- 5. Interpolate ball trajectory ---
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # --- 6. Speed and distance ---
    tracks, speed_distance_estimator = estimate_speed_and_distance(tracks)

    # --- 7. Assign teams ---
    tracks = assign_teams(video_frames, tracks)

    # --- 8. Ball possession ---
    team_ball_control = assign_ball_possession(tracks)

    # --- 9. Count passes ---
    team_pass_count = get_team_pass_count(tracks)

    # --- 10. Visualization ---
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control, team_pass_count)
    output_video_frames = camera_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # --- 11. Heatmaps ---
    # 1) player heatmap for player id 6
    H6 = compute_player_heatmap(tracks, player_id=51)
    plot_heatmap_on_pitch(H6, title="Player 51 heatmap (m)", savepath='heatmap_images/heatmap_player_51.png')
    H114 = compute_player_heatmap(tracks, player_id=114)
    plot_heatmap_on_pitch(H114, title="Player 144 heatmap (m)", savepath='heatmap_images/heatmap_player_144.png')

    ## 2) team heatmap
    H_team1 = compute_team_heatmap(tracks, team_id=1)
    plot_heatmap_on_pitch(H_team1, cmap="Reds", title="Team 1 heatmap (m)", savepath='heatmap_images/team_1.png')
    H_team2 = compute_team_heatmap(tracks, team_id=2)
    plot_heatmap_on_pitch(H_team2, cmap="Reds", title="Team 2 heatmap (m)", savepath='heatmap_images/team_2.png')

    
    # --- 12. Save video ---
    save_video(output_video_frames, 'output_videos/output_video.avi', fps=fps)
    cv2.imwrite("output_images/output_frame_2.jpg", output_video_frames[200])

if __name__ == '__main__':
    main()

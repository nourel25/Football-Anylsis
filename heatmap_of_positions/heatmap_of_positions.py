import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def collect_positions(tracks):
    positions = {}
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            position = track.get('transformed_position', None)
            if position is not None:
                if player_id not in positions:
                    positions[player_id] = []
                x, y = float(position[0]), float(position[1])
                positions[player_id].append((x, 68.0 - y))
    
    return positions

def compute_player_heatmap(tracks, player_id):
    pitch_length, pitch_width = (105.0, 68.0)
    x_bins, y_bins = (105, 68)
    smoth_sigma = 2.0

    positions = collect_positions(tracks).get(player_id, [])
    if not positions:
        return np.zeros((y_bins, x_bins), dtype=float)
    
    x_coords, y_coords = zip(*positions)
    H, xedges, yedges = np.histogram2d(
        x_coords, y_coords, 
        bins=[x_bins, y_bins],
        range=[[0, pitch_length], [0, pitch_width]]
    )

    H = H.T  # Transpose to match (y, x) format
    H_smoothed = gaussian_filter(H, sigma=smoth_sigma)
    return H_smoothed

def compute_team_heatmap(tracks, team_id):
    pitch_length, pitch_width = (105.0, 68.0)
    x_bins, y_bins = (105, 68)
    smoth_sigma = 3.0

    positions = []
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            if track.get('team', None) == team_id:
                position = track.get('transformed_position', None)
                if position is not None:
                    x, y = float(position[0]), float(position[1])
                    positions.append((x, y))
    
    if not positions:
        return np.zeros((y_bins, x_bins), dtype=float)
    
    x_coords, y_coords = zip(*positions)
    H, xedges, yedges = np.histogram2d(
        x_coords, y_coords, 
        bins=[x_bins, y_bins],
        range=[[0, pitch_length], [0, pitch_width]]
    )

    H = H.T  # Transpose to match (y, x) format
    H_smoothed = gaussian_filter(H, sigma=smoth_sigma)   
    return H_smoothed

def plot_heatmap_on_pitch(H,
                          cmap="hot_r",
                          vmin=None,
                          vmax=None,
                          title=None,
                          savepath=None,
                          pitch_img_path="soccer-football-pitch-grass-background.webp"):
    """
    Plot heatmap H on a football pitch background image.
    H shape is (y_bins, x_bins).
    """
    pitch_length, pitch_width = (105.0, 68.0)
    y_bins, x_bins = H.shape

    margin_x, margin_y = 25, 0
    extent = [-margin_x, pitch_length + margin_x,
              0, pitch_width + margin_y]   

    plt.figure(figsize=(12, 8))

    # Load pitch background image
    if pitch_img_path and os.path.exists(pitch_img_path):
        pitch_img = mpimg.imread(pitch_img_path)
        plt.imshow(pitch_img, extent=extent, origin="lower", alpha=1.0)

    # Overlay heatmap
    plt.imshow(H, cmap=cmap, extent=extent, origin="lower",
               alpha=0.6, vmin=vmin, vmax=vmax)

    plt.colorbar(label="Position density")
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.xlabel("Pitch length (m)")
    plt.ylabel("Pitch width (m)")
    if title:
        plt.title(title)
    plt.gca().set_aspect(pitch_length / pitch_width)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
import matplotlib.pyplot as plt
import networkx as nx
from heatmap_of_positions import collect_positions
from collections import Counter

def build_pass_network(tracks):
    # (frame_idx, from_player, to_player, from_team)
    events = []
    prev_player, prev_team = None, None
    prev_frame_idx = None

    for frame_idx, frame_players in enumerate(tracks["players"]):
        current_player, current_team = None, None

        for player_id, pdata in frame_players.items():
            if pdata.get("has_ball", False):
                current_player, current_team = player_id, pdata["team"]
                break

        if current_player is not None:
            if (
                prev_player is not None
                and prev_player != current_player
                and prev_team == current_team
            ):
                # Only register a pass if it happens within 5 seconds (assuming 30 fps)
                if prev_frame_idx is not None and (frame_idx - prev_frame_idx) <= 150:
                    events.append((frame_idx, prev_player, current_player, current_team))

            prev_player, prev_team = current_player, current_team
            prev_frame_idx = frame_idx
        else:
            prev_player, prev_team = None, None
            prev_frame_idx = None   
    
    return events

def build_pass_network_graph(pass_events, team_filter=None):
    edge_counts = Counter()
    for (frame_idx, frm, to, team_id) in pass_events:
        if team_filter is None or team_id == team_filter:
            edge_counts[(frm, to)] += 1
    return edge_counts


def compute_player_average_positions(tracks):
    positions = collect_positions(tracks)
    avg_positions = {}
    for player_id, pos_list in positions.items():
        if pos_list:
            avg_x = sum(p[0] for p in pos_list) / len(pos_list)
            avg_y = sum(p[1] for p in pos_list) / len(pos_list)
            avg_positions[player_id] = (avg_x, avg_y)
    return avg_positions

def plot_pass_network(edge_counts,
                      avg_positions,
                      min_edge_weight = 1,
                      title = None,
                      savepath = None,
                      linewidth_scale = 0.5):
    """
    Draw pass network on pitch using average positions as node coordinates.
    Uses networkx for layout and drawing.
    """
    G = nx.DiGraph()
    for (frm, to), cnt in edge_counts.items():
        if cnt < min_edge_weight:
            continue
        G.add_edge(frm, to, weight=cnt)

    plt.figure(figsize=(12, 8))
    # Draw pitch base
    pitch_length = 105.0  # meters
    pitch_width =  68.0   # meters
    plt.xlim(0, pitch_length)
    plt.ylim(0, pitch_width)
    plt.gca().set_aspect((pitch_length / pitch_width))

    # Draw edges as arrows with widths proportional to counts
    for u, v, data in G.edges(data=True):
        if u not in avg_positions or v not in avg_positions:
            continue
        x1, y1 = avg_positions[u]
        x2, y2 = avg_positions[v]
        cnt = data.get("weight", 1)
        lw = linewidth_scale * cnt
        plt.arrow(x1, y1, x2-x1, y2-y1, width=0.001*lw, head_width=0.6, length_includes_head=True, alpha=0.8, color='yellow')

    # Draw nodes sized by total degree
    degrees = dict(G.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1
    for node, pos in avg_positions.items():
        if node not in degrees:
            continue
        x, y = pos
        size = 300 * (degrees[node] / max_deg)
        plt.scatter([x], [y], s=size, c='black', alpha=0.9)
        plt.text(x+0.3, y+0.3, str(node), color='blue', fontsize=9, weight='bold')

    if title:
        plt.title(title)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
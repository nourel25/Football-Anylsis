from utils.bbox_utils import get_bbox_center, measure_distance

class PlayerBallAssigner:
    def __init__(self, max_player_ball_distance=70):
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball(self, players, ball_bbox):
        if not ball_bbox or len(ball_bbox) != 4:
            return -1

        ball_position = get_bbox_center(ball_bbox)
        min_dist = float("inf")
        assigned = -1

        for pid, pdata in players.items():
            bbox = pdata["bbox"]
            d_left = measure_distance((bbox[0], bbox[-1]), ball_position)
            d_right = measure_distance((bbox[2], bbox[-1]), ball_position)
            d = min(d_left, d_right)

            if d < self.max_player_ball_distance and d < min_dist:
                min_dist = d
                assigned = pid

        return assigned

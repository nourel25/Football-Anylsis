from utils.bbox_utils import get_bbox_center, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        # threshold 
        # (if ball is farther than 70px from any player it wont be assigned)
        self.max_player_ball_distance = 70

    def assign_ball(self, players, ball_bbox):
        ball_position = get_bbox_center(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            # bottom-left corner
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            # bottom-right corner
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player
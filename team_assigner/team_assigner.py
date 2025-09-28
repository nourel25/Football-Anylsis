from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=5)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            return None
        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return None
        top_half = image[: image.shape[0] // 2, :]
        if top_half.size == 0:
            return None

        try:
            kmeans = self.get_clustering_model(top_half)
        except Exception:
            return None

        labels = kmeans.labels_
        clustered = labels.reshape(top_half.shape[0], top_half.shape[1])
        corner_clusters = [clustered[0,0], clustered[0,-1],
                           clustered[-1,0], clustered[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, pdata in player_detections.items():
            color = self.get_player_color(frame, pdata["bbox"])
            if color is not None:
                player_colors.append(color)

        if len(player_colors) < 2:
            print("⚠️ Not enough player colors, fallback to default")
            self.team_colors = {1: np.array([255,0,0]), 2: np.array([0,0,255])}
            return

        player_colors = np.array(player_colors)
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        self.kmeans.fit(player_colors)

        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

    def get_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        color = self.get_player_color(frame, bbox)
        if color is None or self.kmeans is None:
            return -1
        team_id = self.kmeans.predict(color.reshape(1,-1))[0] + 1
        self.player_team_dict[player_id] = team_id
        return team_id

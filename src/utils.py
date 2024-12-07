import pickle
import os

class Data:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.k = 0
        self._dist = {}
        self.neighbors = {}
        self.allDists = []

    def dist(self, u, v):
        if (u == v):
            return 0
        return self._dist.get((u, v), float('inf'))

    def save(self, preprocessed_path):
        with open(preprocessed_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(preprocessed_path):
        with open(preprocessed_path, 'rb') as f:
            return pickle.load(f)

def load_data(data_path):
    preprocessed_path = data_path + '.preprocessed'
    if os.path.exists(preprocessed_path):
        data = Data.load(preprocessed_path)
        return data
    data = Data()
    with open(data_path, 'r') as f:
        n, m, k = map(int, f.readline().split())
        data.n = n
        data.m = m
        data.k = k
        for _ in range(data.m):
            u, v, d = map(int, f.readline().split())
            data._dist[(u, v)] = d
            data._dist[(v, u)] = d
            data.neighbors.setdefault(u, []).append(v)
            data.neighbors.setdefault(v, []).append(u)
    
    # Initialize distance matrix
    dist = [[float('inf')] * data.n for _ in range(data.n)]
    for u in range(1, data.n + 1):
        dist[u - 1][u - 1] = 0
    for (u, v), d in data._dist.items():
        dist[u - 1][v - 1] = d

    print("Computing all pair shortest path...")
    # Floyd-Warshall algorithm
    for k in range(data.n):
        for i in range(data.n):
            for j in range(data.n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    for i in range(data.n):
        for j in range(data.n):
            if data.dist(i + 1, j + 1) == float('inf'):
                data._dist[(i + 1, j + 1)] = dist[i][j]
                data._dist[(j + 1, i + 1)] = dist[i][j]
    
    # Store distances
    data.allDists = sorted(set(dist[i][j] for i in range(data.n) for j in range(i+1, data.n)))
    
    print(f"Saving preprocessed data to {preprocessed_path}...")
    data.save(preprocessed_path)
    return data

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

def load_data(data_path):
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
            data.allDists.append(d)
        data.allDists.sort()
    return data

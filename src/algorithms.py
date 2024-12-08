import os
import random
from . import utils
from typing import List, Tuple
import time
import pickle

def eval(algorithm, data_path, p=1):
    data = utils.load_data(data_path)
    algorithm_class = globals()[algorithm]
    if algorithm == 'Scr_p':
        algo_instance = algorithm_class(data, p)
    else:
        algo_instance = algorithm_class(data)
    start = time.time()
    result = algo_instance.run()
    end = time.time()
    print(f"Evaluating {algorithm} on {data_path}: make-span {result[1]} in {end - start:.8f} seconds")
    return result, end - start

class Algorithm:
    def __init__(self, data):
        self.n = data.n
        self.m = len(data.allDists)
        self.k = data.k
        self.data = data
        random.seed(time.time())
    
    def setDist(self, v, C):
        if C == []:
            return 0
        return min([self.data.dist(v, c) for c in C])

    def makespan(self, C):
        return max([self.setDist(i, C) for i in range(1, self.n + 1)])

    def run(self) -> Tuple[List[int], float]:
        pass

class Gon(Algorithm):
    def __init__(self, data):
        super().__init__(data)

    def run(self):
        C = []
        rest = set(range(1, self.n + 1))
        C.append(random.randint(1, self.n))
        rest.remove(C[0])
        for i in range(self.k - 1):
            ci = max(rest, key=lambda x: self.setDist(x, C))
            C.append(ci)
            rest.remove(ci)
        return C, self.makespan(C)

class HS(Algorithm):
    def __init__(self, data):
        super().__init__(data)
    
    def check(self, r):
        rest = set(range(1, self.n + 1))
        C = []
        while rest != set():
            c = random.choice(list(rest))
            C.append(c)
            rmv = set()
            for i in rest:
                if self.data.dist(i, c) <= 2 * r:
                    rmv.add(i)
            rest -= rmv
        if len(C) <= self.k:
            return C
        else:
            return None

    def run(self):
        low, high = 0, self.m - 1
        res = None
        while low < high:
            mid = (low + high) // 2
            C = self.check(self.data.allDists[mid])
            if C is None:
                low = mid + 1
            else:
                high = mid
                res = C
        return res, self.makespan(res)

class Gr(Algorithm):
    def __init__(self, data):
        super().__init__(data)

    def run(self):
        C = []
        for _ in range(self.k - 1):
            applicable = [i for i in range(1, self.n + 1) if i not in C]
            ci = max(applicable, key=lambda x: self.makespan(C + [x]))
            C.append(ci)
        return C, self.makespan(C)

class CDS(Algorithm):
    def __init__(self, data):
        super().__init__(data)
    
    def getNeighbors(self, v, r):
        return [u for u in range(1, self.n + 1) if self.data.dist(v, u) <= r]

    def check(self, r):
        C = []
        D = set()
        score = [0] * (self.n + 1)
        neighbor = [[]] * (self.n + 1)
        for i in range(1, self.n + 1):
            neighbor[i] = self.getNeighbors(i, r)
            score[i] = len(neighbor[i])
        for i in range(self.k):
            f = max(range(1, self.n + 1), key=lambda x: self.setDist(x, C))
            if i == 0:
                f = random.randint(1, self.n)
            vf = max(neighbor[f], key=lambda x: score[x])
            S = [v for v in neighbor[vf] if v not in D]
            for v in S:
                D.add(v)
                for u in neighbor[v]:
                    score[u] -= 1
            C.append(vf)
        return C
    
    def run(self):
        C = min([self.check(self.data.allDists[i]) for i in range(self.m)], key=lambda x: self.makespan(x))
        return C, self.makespan(C)

class CDSh(CDS):
    def __init__(self, data):
        super().__init__(data)

    def run(self):
        low, high = 0, self.m - 1
        res = None
        minMakespan = float('inf')
        while low < high:
            mid = (low + high) // 2
            C = self.check(self.data.allDists[mid])
            if self.makespan(C) < minMakespan:
                high = mid
                res = C
            else:
                low = mid + 1
        return res, self.makespan(res)

class CDSh_p(CDS):
    def __init__(self, data):
        super().__init__(data)

    def run(self):
        low, high = 0, self.m - 1
        res = None
        minMakespan = float('inf')
        while low < high:
            mid = (low + high) // 2
            C = min([self.check(self.data.allDists[mid]) for _ in range(self.n)], key=lambda x: self.makespan(x))
            if self.makespan(C) < minMakespan:
                high = mid
                res = C
            else:
                low = mid + 1
        return res, self.makespan(res)

class Scr(Algorithm):
    def __init__(self, data):
        super().__init__(data)
    
    def getNeighbors(self, v, r):
        return [u for u in range(1, self.n + 1) if self.data.dist(v, u) <= r]

    def check(self, r):
        covcnt = [0] * (self.n + 1)
        D = set()
        score = [0] * (self.n + 1)
        neighbor = [[]] * (self.n + 1)
        for i in range(1, self.n + 1):
            neighbor[i] = self.getNeighbors(i, r)
            covcnt[i] = len(neighbor[i])
            score[i] = covcnt[i]
        for _ in range(self.n):
            x = min(range(1, self.n + 1), key=lambda x: score[x])
            y0 = 0
            for y in neighbor[x]:
                if covcnt[y] == 1:
                    y0 = y
                    break
            if y0:
                D.add(x)
                for y in neighbor[x]:
                    covcnt[y] = 0
            else:
                for y in neighbor[x]:
                    if covcnt[y] > 0:
                        covcnt[y] -= 1
                        score[y] += 1
            score[x] = float('inf')
        return D
    
    def run(self):
        low, high = 0, len(self.data.allDists) - 1
        res = None
        # print(self.data.allDists)
        while low < high:
            mid = (low + high) // 2
            C = self.check(self.data.allDists[mid])
            if len(C) <= self.k:
                high = mid
                res = C
            else:
                low = mid + 1
        return res, self.makespan(res)

class Scr_p(Scr):
    def __init__(self, data, p):
        super().__init__(data)
        self.p = p
    
    def topp(self, score, p):
        top_nodes = []
        for i in range(1, self.n + 1):
            if len(top_nodes) < p:
                top_nodes.append(i)
            else:
                top_nodes = sorted(top_nodes + [i], key=lambda x: score[x])[:p]
        return top_nodes

    def check(self, r):
        covcnt = [0] * (self.n + 1)
        D = set()
        score = [0] * (self.n + 1)
        neighbor = [[]] * (self.n + 1)
        for i in range(1, self.n + 1):
            neighbor[i] = self.getNeighbors(i, r)
            covcnt[i] = len(neighbor[i])
            score[i] = covcnt[i]
        for _ in range(self.n):
            x = random.choice(self.topp(score, self.p))
            y0 = 0
            for y in neighbor[x]:
                if covcnt[y] == 1:
                    y0 = y
                    break
            if y0:
                D.add(x)
                for y in neighbor[x]:
                    covcnt[y] = 0
            else:
                for y in neighbor[x]:
                    if covcnt[y] > 0:
                        covcnt[y] -= 1
                        score[y] += 1
            score[x] = float('inf')
        return D
    
    def run(self):
        low, high = 0, len(self.data.allDists) - 1
        res = None
        while low < high:
            mid = (low + high) // 2
            C = min([self.check(self.data.allDists[mid]) for _ in range(self.n)], key=lambda x: len(x))
            if len(C) <= self.k:
                high = mid
                res = C
            else:
                low = mid + 1
        return res, self.makespan(res)
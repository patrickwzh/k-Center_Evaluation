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



def compute_results(file_path):
    import math

    results = {}  # {(alg_name, data_num): {'make_span': [...], 'time': [...]}}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    current_key = None
    for line in lines:
        line = line.strip()
        if line.startswith('Algorithm:'):
            # Parse Algorithm and Data
            parts = line.split(',')
            alg_name = parts[0].split(':')[1].strip()
            data_part = parts[1].split(':')[1].strip()
            # Extract data_num from data_part (e.g., './data/pmed1.txt' -> '1')
            data_num = data_part.split('pmed')[-1].split('.txt')[0]
            if alg_name == 'Scr_p':
                p_num = parts[2].split(':')[1].strip()
                alg_name += f'_{p_num}'
            current_key = (alg_name, data_num)
            if current_key not in results:
                results[current_key] = {'make_span': [], 'time': []}
        elif line.startswith('- Make-span:'):
            # Parse Make-span and Time
            parts = line.split(',')
            make_span = int(parts[0].split(':')[1].strip())
            time = float(parts[1].split(':')[1].strip())
            # Append to results
            results[current_key]['make_span'].append(make_span)
            results[current_key]['time'].append(time)
    # Compute averages and standard deviations
    final_results = {}
    for key, value in results.items():
        make_span_list = value['make_span']
        time_list = value['time']
        make_span_avg = sum(make_span_list) / len(make_span_list)
        make_span_std = math.sqrt(sum((x - make_span_avg) ** 2 for x in make_span_list) / len(make_span_list))
        time_avg = sum(time_list) / len(time_list)
        time_std = math.sqrt(sum((x - time_avg) ** 2 for x in time_list) / len(time_list))
        final_results[key] = {
            'make_span_avg': make_span_avg,
            'make_span_stddev': make_span_std,
            'time_avg': time_avg,
            'time_stddev': time_std
        }
    return final_results

def plot_make_span(data):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # # Set up PGF backend
    # plt.rcParams.update({
    #     "pgf.texsystem": "pdflatex",  # Use pdflatex
    #     "text.usetex": True,          # Use LaTeX for text rendering
    #     "pgf.preamble": r"\usepackage{amsmath}"  # Load extra packages
        
    # })

    x = np.arange(1, 41)
    algorithms = ['Gon', 'HS', 'CDS', 'CDSh', 'CDSh_p', 'Scr', 'Gr']
    opt = [127, 98, 93, 74, 48, 84, 64, 55, 37, 20, 59, 51, 35, 26, 18, 47, 39, 28, 18, 13, 40, 38, 22, 15, 11, 38, 32, 18, 13, 9, 30, 29, 15, 11, 30, 27, 15, 29, 23, 13]
    y = {}
    yerr = {}
    datframe = {'x': x}
    datframe2 = {'Algorithms': algorithms, 'Avg': [], 'Std': []}
    for alg in algorithms:
        y[alg] = []
        for i in range(1, 41):
            if (alg, str(i)) not in data:
                break
            else:
                y[alg].append(data[(alg, str(i))]['make_span_avg'])
        y[alg] = np.array(y[alg], dtype=float)
        # y[alg] = np.array([data.get((alg, str(i)), {}).get('make_span_avg', None) for i in range(1, 41)], dtype=float)
        y[alg] /= np.array(opt[:len(y[alg])], dtype=float)
        yerr[alg] = []
        for i in range(1, 41):
            if (alg, str(i)) not in data:
                break
            else:
                yerr[alg].append(data[(alg, str(i))]['make_span_stddev'])
        yerr[alg] = np.array(yerr[alg], dtype=float)
        # yerr[alg] = np.array([data.get((alg, str(i)), {}).get('make_span_stddev', None) for i in range(1, 41)], dtype=float)
        yerr[alg] /= np.array(opt[:len(yerr[alg])], dtype=float)
        datframe[alg] = y[alg]
        datframe[alg + '_error'] = yerr[alg]
        datframe2['Avg'].append(np.mean(y[alg]))
        datframe2['Std'].append(np.mean(yerr[alg]))
        # plt.errorbar(x, y, yerr=yerr, label=alg)
    # df = pd.DataFrame(datframe)
    # df.to_csv('../data.csv', index=False)
    df2 = pd.DataFrame(datframe2)
    df2.to_csv('data2.csv', index=False)
    # plt.xlabel('Data')
    # plt.ylabel('Empirical Approximation Ratio')
    # plt.legend()
    # plt.title('Empirical Approximation Ratio of Different Algorithms')
    # # plt.show()
    # plt.savefig('make_span.pgf')

def plot_time(data):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # # Set up PGF backend
    # plt.rcParams.update({
    #     "pgf.texsystem": "pdflatex",  # Use pdflatex
    #     "text.usetex": True,          # Use LaTeX for text rendering
    #     "pgf.preamble": r"\usepackage{amsmath}"  # Load extra packages
        
    # })

    x = np.arange(1, 41)
    algorithms = ['Gon', 'HS', 'CDS', 'CDSh', 'CDSh_p', 'Scr', 'Gr']
    y = {}
    yerr = {}
    datframe = {'x': x}
    datframe2 = {'Algorithms': algorithms, 'Avg': [], 'Std': []}
    for alg in algorithms:
        y[alg] = []
        for i in range(1, 41):
            if (alg, str(i)) not in data:
                break
            else:
                y[alg].append(data[(alg, str(i))]['time_avg'])
        y[alg] = np.array(y[alg], dtype=float)
        # y[alg] = np.array([data.get((alg, str(i)), {}).get('time_avg', None) for i in range(1, 41)], dtype=float)
        yerr[alg] = []
        for i in range(1, 41):
            if (alg, str(i)) not in data:
                break
            else:
                yerr[alg].append(data[(alg, str(i))]['time_stddev'])
        yerr[alg] = np.array(yerr[alg], dtype=float)
        # yerr[alg] = np.array([data.get((alg, str(i)), {}).get('time_stddev', None) for i in range(1, 41)], dtype=float)
        datframe[alg] = y[alg]
        datframe[alg + '_error'] = yerr[alg]
        datframe2['Avg'].append(np.mean(y[alg]))
        datframe2['Std'].append(np.mean(yerr[alg]))
        # plt.errorbar(x, y, yerr=yerr, label=alg)
    # df = pd.DataFrame(datframe)
    # df.to_csv('../data_time.csv', index=False)
    df2 = pd.DataFrame(datframe2)
    df2.to_csv('data2_time.csv', index=False)
    # plt.xlabel('Data')
    # plt.ylabel('Empirical Approximation Ratio')
    # plt.legend()
    # plt.title('Empirical Approximation Ratio of Different Algorithms')
    # # plt.show()
    # plt.savefig('make_span.pgf')
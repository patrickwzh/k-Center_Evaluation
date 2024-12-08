from src.algorithms import eval
from src.utils import compute_results
from src.utils import plot_make_span, plot_time

def run():
    # algorithms = ['CDS', 'CDSh', 'CDSh_p']
    # data_files = {'CDS': [f'./data/pmed{i}.txt' for i in range(1, 6)],
    #               'CDSh': [f'./data/pmed{i}.txt' for i in range(1, 41)],
    #               'CDSh_p': [f'./data/pmed{i}.txt' for i in range(1, 11)]}
    algorithms = ['Scr_p']
    data_files = {'Scr': [f'./data/pmed{i}.txt' for i in range(1, 41)],
              'Scr_p': [f'./data/pmed{i}.txt' for i in range(1, 11)]}
    with open('./eval_result.txt', 'a') as result_file:
        for algorithm in algorithms:
            for data_path in data_files[algorithm]:
                if algorithm != 'Scr_p':
                    for run in range(1, 4):
                        (result_list, number), time_lapse = eval(algorithm, data_path)
                        result_file.write(f'Algorithm: {algorithm}, Data: {data_path}, Run: {run}\n')
                        result_file.write(f'- Make-span: {number}, Time: {time_lapse}, Result: {result_list}\n\n')
                else:
                    for p in range(2, 9):
                        for run in range(1, 4):
                            (result_list, number), time_lapse = eval(algorithm, data_path, p)
                            result_file.write(f'Algorithm: {algorithm}, Data: {data_path}, p: {p}, Run: {run}\n')
                            result_file.write(f'- Make-span: {number}, Time: {time_lapse}, Result: {result_list}\n\n')

def compute():
    res = compute_results('./eval_result.txt')
    plot_make_span(res)
    plot_time(res)

if __name__ == '__main__':
    compute()
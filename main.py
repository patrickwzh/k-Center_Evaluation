from src.algorithms import eval

algorithms = ['Gon', 'HS', 'Gr', 'CDS', 'CDSh', 'CDSh_p']
data_files = [f'./data/pmed{i}.txt' for i in range(1, 41)]

if __name__ == '__main__':
    with open('./eval_result.txt', 'w') as result_file:
        for algorithm in algorithms:
            for data_path in data_files:
                for run in range(1, 4):
                    (result_list, number), time_lapse = eval(algorithm, data_path)
                    result_file.write(f'Algorithm: {algorithm}, Data: {data_path}, Run: {run}\n')
                    result_file.write(f'- Make-span: {number}, Time: {time_lapse}, Result: {result_list}\n\n')
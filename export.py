import pickle, csv
from config import cfg

with open(cfg.data_bin, 'rb') as f:
    data = pickle.load(f)

for j in range(2):
    with open(cfg.answer_path[j], 'w') as f:
        writer = csv.writer(f, delimiter=',',)
        writer.writerow(['id', 'proba'])
        for i in range(m):
            writer.writerow((data[1][i], data[0][i, j]))

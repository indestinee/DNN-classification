import os, csv, pickle
import numpy as np
from IPython import embed
from config import cfg

def mkdir(path):# {{{
    if not os.path.isdir(path):
        os.mkdir(path)
        return True
    return False
# }}}
def load_csv(csv_file, shuffle=True):# {{{
    print('[OPR] load_csv %s..' % csv_file)
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        keys = next(reader)
        reader = csv.DictReader(f, fieldnames=keys)
        for i, row in enumerate(reader):
            each = {'feature': []}
            d = dict(row)
            for key, value in d.items():
                if not 'feature' == key[:7]:
                    each[key] = value
            for k in range(cfg.input_shape):
                each['feature'].append(d['feature%s' % k])
            each['feature'] = np.array(each['feature'], dtype=np.float32)
            data.append(each)
            if (i + 1) % 20000 == 0 or i == 0:
                print('[PRO] done %d..' % (i + 1))
    np.random.seed(233)
    if shuffle:
        np.random.shuffle(data)
    if 'weight' not in keys:
        weight = None
    else:
        weight = np.array([each['weight'] for each in data], np.float32)
    print('[SUC] load_csv done..')
    return data, weight
# }}}
def save_cache(obj, path):# {{{
    print('[OPR] save_cache %s..' % path)
    if os.path.isfile(path):
        print('[WRN] path exists (%s)..' % path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print('[SUC] save_cache done..')
# }}}
def load_cache(path):# {{{
    print('[OPR] load_cache %s..' % path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print('[SUC] load_cache done..')
    return data
# }}}
def list_to_array_train(d):# {{{
    feature = np.array([each['feature'] for each in d])
    labels = np.array([each['label'] for each in d], dtype=int)
    n = len(labels)
    res = np.zeros((n, 2))
    res[range(n), labels] = 1
    return feature, res
# }}}
def list_to_array_test(d):# {{{
    feature = np.array([each['feature'] for each in d])
    index = np.array([each['id'] for each in d], dtype=int)
    return feature, index
# }}}


import os, csv, pickle, bisect
import numpy as np

def mkdir(path):# {{{
    if not os.path.isdir(path):
        os.mkdir(path)
        return True
    return False
# }}}
def load_csv(csv_file):# {{{
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
                if 'feature' == key[:7]:
                    each['feature'].append(value)
                else:
                    each[key] = value
            each['feature'] = np.array(each['feature'], dtype=np.float32)
            data.append(each)
            if (i + 1) % 20000 == 0 or i == 0:
                print('[PRO] done %d..' % (i + 1))
    np.random.seed(233)
    np.random.shuffle(data)
    if 'weight' not in keys:
        weight = None
    else:
        weight = [each['weight'] for each in data]
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
def list_to_array(d):# {{{
    feature = np.array([each['feature'] for each in d])
    labels = np.array([each['label'] for each in d], dtype=int)
    n = len(labels)
    res = np.zeros((n, 2))
    res[range(n), labels] = 1
    return feature, res
# }}}


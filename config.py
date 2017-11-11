import os
class config:
    cache_path = './data/cache/'
    train_log = './train_log'
    data_cache = os.path.join(cache_path, 'data.pkl')

    train_csv = './data/raw_data/20171103/ai_challenger_stock_train_20171103/stock_train_data_20171103.csv'
    test_csv = './data/raw_data/20171103/ai_challenger_stock_test_20171103/stock_test_data_20171103.csv'

    input_shape = 105
    train_batch_size = 249008
    validation_batch_size = 27668
    
    result_path = './result'
    data_bin = os.path.join(result_path, 'data.bin')

cfg = config()



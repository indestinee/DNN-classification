import os
class config:
    cache_path = './data/cache/'
    data_cache = os.path.join(cache_path, 'data.pkl')
    train_log = './train_log'

    train_csv = './data/raw_data/20171103/ai_challenger_stock_train_20171103/stock_train_data_20171103.csv'
    test_csv = './data/raw_data/20171103/ai_challenger_stock_test_20171103/stock_test_data_20171103.csv'

    input_shape = 105


cfg = config()



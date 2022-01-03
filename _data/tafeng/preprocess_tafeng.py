import os
import pickle
import numpy as np
import random
import pandas as pd
from datetime import datetime


def main():
    """ Main function. """
    max_len = 1000
    n_sample = 4000
    remove_mono_date_session = True
    remove_bi_date_session = True

    random.seed(0)

    path = os.path.dirname(os.getcwd()) + '/tafeng/'
    file_name = 'ta_feng_all_months_merged.csv'
    df_tafeng_original = pd.read_csv(path + file_name)

    '''Remove outliers'''
    asset_98 = np.percentile(df_tafeng_original['ASSET'].values, 98)
    price_98 = np.percentile(df_tafeng_original['SALES_PRICE'].values, 98)
    df_tafeng = df_tafeng_original[(df_tafeng_original['ASSET'] < asset_98) & (df_tafeng_original['SALES_PRICE'] < price_98)].copy()

    df_tafeng = df_tafeng[['TRANSACTION_DT', 'CUSTOMER_ID', 'PRODUCT_ID']]
    df_tafeng.columns = ['date', 'user', 'item']
    df_tafeng.date = [(datetime.strptime(str(dt), '%m/%d/%Y')-datetime(2000, 11, 1)).days for dt in df_tafeng.date]

    df_tafeng.drop_duplicates(subset=df_tafeng.columns, keep='first', inplace=True)

    '''Drop user with less than 5 interactions'''
    list_user_5 = (df_tafeng.user.value_counts() >= 5).where(df_tafeng.user.value_counts() >= 5).dropna().index
    df_tafeng_sample = df_tafeng[df_tafeng.user.isin(list_user_5)]

    '''Drop long sequences' interactions exceeded 1000 '''
    count_interactions_by_user = df_tafeng_sample.value_counts('user')
    list_exceeded_user = count_interactions_by_user[count_interactions_by_user > 1000].index.tolist()
    for user in list_exceeded_user:
        df = df_tafeng_sample[df_tafeng_sample.user == user]
        df_tafeng_sample = df_tafeng_sample.drop(index=df[:-max_len].index)

    '''Drop user with less than 2 interactions date'''
    if remove_bi_date_session:
        if remove_mono_date_session:
            threshold_date = 1
        else:
            threshold_date = 2
    list_user_date = []

    for id_user, df in df_tafeng_sample.groupby('user'):
        _len = len(df.date.unique())
        if _len > threshold_date:
            list_user_date.append(id_user)

    df_tafeng_sample = df_tafeng_sample[df_tafeng_sample.user.isin(list_user_date)]

    '''Random sample user'''
    if n_sample < 26333:

        list_user_sample = random.sample(list(df_tafeng_sample.user.unique()), n_sample)
        random.shuffle(list_user_sample)

        df_tafeng_sample = df_tafeng_sample[df_tafeng_sample.user.isin(list_user_sample)]

    else:

        list_user_sample = list(df_tafeng_sample.user.unique())
        random.shuffle(list_user_sample)

    ''''Map item and user index'''
    list_item_tafeng = df_tafeng_sample.item.unique()
    map_item = {}
    for i, item in enumerate(list_item_tafeng):
        map_item[item] = i+1
    df_tafeng_sample.item = [map_item[x] for x in df_tafeng_sample.item.values]

    '''Output dataset'''
    train_item = []
    train_item_gt = []
    train_date = []
    train_date_gt = []

    test_item = []
    test_item_gt = []
    test_date = []
    test_date_gt = []

    len_user_sequence = []

    for i, user in enumerate(list_user_sample):

        df = df_tafeng_sample[df_tafeng_sample.user == user]
        len_user_sequence.append(df.shape[0])

        if i < (0.8*n_sample):
            train_item.append(list(df.item)[:-1])
            train_item_gt.append(list(df.item)[-1])

            train_date.append(list(df.date)[:-1])
            train_date_gt.append(list(df.date)[-1])
        else:
            test_item.append(list(df.item)[:-1])
            test_item_gt.append(list(df.item)[-1])

            test_date.append(list(df.date)[:-1])
            test_date_gt.append(list(df.date)[-1])

    tra = (train_item, train_item_gt, train_date, train_date_gt)
    tes = (test_item, test_item_gt, test_date, test_date_gt)
    '''Save _data into txt'''
    pickle.dump(tra, open('train.txt', 'wb'))
    pickle.dump(tes, open('test.txt', 'wb'))

    # Logs
    print('Total item: ', len(df_tafeng.item.unique()))
    print('Total user: ', len(df_tafeng.user.unique()))
    print('')

    print('Interaction>5 user: ', len(list_user_5))
    print('Date session>2 user: ', len(list_user_date))
    print('')

    print('Sampled user #: ', n_sample)
    print('Sampled item #: ', len(map_item))
    print('')

    print('Min interactions: ', min(len_user_sequence))
    print('Max interactions: ', max(len_user_sequence))
    print('Average interactions: ', np.mean(len_user_sequence))
    print('')


if __name__ == '__main__':
    main()



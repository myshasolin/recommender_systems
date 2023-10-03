import pandas as pd
import numpy as np


def get_result_table(data: pd.DataFrame):
    result = data.groupby('user_id')['item_id'].unique().reset_index()
    result.columns = ['user_id', 'actual']
    return result


def prefilter_items(data, take_n_popular=5000, item_features=None):
    # Уберём не интересные для рекомендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберём слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    #data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data.loc[:, 'price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Уберем слишком дорогие товары
    data = data[data['price'] < 50]

    # Возьмём топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999


    return data

def postfilter_items(user_id, recommednations):
    pass

if __name__ == '__main__':
    pass

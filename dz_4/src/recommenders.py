import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares

RANDOM_STATE = 42


class MainRecommender:
    """Рекомендации, которые можно получить из ALS"""

    def __init__(self, data, als_recommend=False, num_user=False):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        self.user_item_matrix = self.prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id =\
            self.prepare_dicts(self.user_item_matrix)

        if als_recommend is not False:
            self.model = self.ALS_fit(csr_matrix(self.user_item_matrix))

        if num_user is not False:
            self.num_user = num_user
            self.recommender = self.ALS_recommend(self.num_user, self.model, csr_matrix(self.user_item_matrix))

    @staticmethod
    def prepare_matrix(data: pd.DataFrame):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0)
        user_item_matrix = user_item_matrix.astype(np.float32)  # необходимый тип матрицы для implicit
        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values
        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))
        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))
        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id


    @staticmethod
    def ALS_fit(user_item_matrix, n_factors=100, regularization=0.001, iterations=15,
            num_threads=4, random_state=RANDOM_STATE):
        """Обучает модель AlternatingLeastSquares"""
        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        calculate_training_loss=True,
                                        num_threads=num_threads,
                                        random_state=random_state)
        model.fit(user_item_matrix)
        return model

    def ALS_recommend(self, user, model, sparse_user_item, N=5):
        """Рекомендуем топ-N товаров, основанный на матричной факторизации"""
        res = [self.id_to_itemid[rec] for rec in
               model.recommend(userid=self.userid_to_id[user],
                               user_items=sparse_user_item[self.userid_to_id[user]],  # на вход user-item matrix
                               N=N,
                               filter_already_liked_items=False,
                               filter_items=[self.itemid_to_id[999999]],
                               recalculate_user=True)[0]]
        return res

    def update_dict(self, user_id):
        """Если появился новый user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1
            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})


    def generate_recommendations(self):
        users = self.top_purchases['user_id'].unique()
        recommendations = {}

        for user in users:
            recs = self.ALS_recommend(user, self.model, csr_matrix(self.user_item_matrix))
            recommendations[user] = recs

        self.recommendations = recommendations

if __name__ == '__main__':
    pass

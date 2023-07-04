import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from pyspark.ml.recommendation import ALS
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as sf

from lightfm import LightFM

class MainDistributedRecommender:
    """Рекомендации, которые можно получить из ALS и LightFM"""

    def __init__(self, data,
                 spark_session=False, rank=30, userCol='user_id', itemCol='item_id',
                 ratingCol='relevance', maxIter=10, alpha=1.0, regParam=0.1, 
                 implicitPrefs=True, seed=42, coldStartStrategy='drop'):

        self.rank, self.maxIter = rank, maxIter
        self.alpha, self.regParam = alpha, regParam
        self.data = data

        self.recommender_als = self.get_ALS_recommend(self.data,  self.rank, userCol, itemCol, 
                                                      ratingCol, self.maxIter, self.alpha, self.regParam, 
                                                      implicitPrefs, seed, coldStartStrategy)    

        self.recommender_lightfm = self.get_LightFM_recommend(self.data, self.recommender_als, 
                                                              userCol, itemCol, ratingCol,
                                                              no_components=30, loss='warp', k=6)

    def get_ALS_recommend(self, data, rank, userCol, itemCol, ratingCol, maxIter, alpha,
                          regParam, implicitPrefs, seed, coldStartStrategy):

        model = ALS(rank=rank, userCol=userCol, itemCol=itemCol,
            ratingCol=ratingCol, maxIter=maxIter, alpha=alpha, regParam=regParam,
            implicitPrefs=implicitPrefs, seed=seed, coldStartStrategy=coldStartStrategy
            ).fit(data)
        recs_als = model.recommendForAllUsers(100)

        recs_als = (
            recs_als.withColumn('recommendations', sf.explode('recommendations'))
            .withColumn(itemCol, sf.col('recommendations.item_id'))
            .withColumn('relevance', sf.col('recommendations.rating')
            .cast(DoubleType()))
            .select('user_id', 'item_id', 'relevance')
        )
        return recs_als.toPandas()

    def get_LightFM_recommend(self, data, als_recs, userCol, itemCol, ratingCol, no_components=30, loss='warp', k=6):
        model = LightFM(no_components=no_components, loss=loss)

        # Получаем список всех пользователей и товаров
        users = [i[0] for i in data.select(userCol).distinct().collect()]
        items = [i[0] for i in data.select(itemCol).distinct().collect()]

        # Создаем словари для индексирования пользователей и товаров
        user_map = {uid: i for i, uid in enumerate(users)}
        item_map = {iid: j for j, iid in enumerate(items)}

        # Создаем матрицу взаимодействий пользователей и товаров на основе обучающего набора данных
        interactions = np.zeros((len(users), len(items)))
        for row in data.collect():
            interactions[user_map[row[userCol]], item_map[row[itemCol]]] = row[ratingCol]

        # Добавляем рекомендации ALS-модели к матрице взаимодействий
        for row in als_recs.itertuples():
            interactions[user_map[row.user_id], item_map[row.item_id]] = getattr(row, ratingCol)
        interactions_sparse = csr_matrix(interactions)

        model.fit(interactions_sparse, epochs=10)

        n_users, n_items = interactions.shape

        # здесь генерируем рекомендации для каждого пользователя
        result = []
        for user in users:
            user_id = user_map.get(user)
            if user_id is None:
                continue
            scores = model.predict(user_id, np.arange(n_items))
            top_items = sorted(zip(scores, range(n_items)), reverse=True)

            # Берем все сгенерированные рекомендации
            user_recs = []
            for score, item in top_items:
                if len(user_recs) >= k:
                    break
                if item_map.get(item) is not None and items[item_map.get(item)] not in [r[itemCol] for r in result]:
                    user_recs.append((user, items[item_map.get(item)], score))

            # Добавляем дополнительные рекомендации ALS-модели, если недостаточно уникальных рекомендаций
            if len(user_recs) < k:
                als_user_recs = als_recs[als_recs[userCol] == user][[userCol, itemCol, ratingCol]].drop_duplicates()
                for row in als_user_recs.itertuples():
                    if len(user_recs) >= k:
                        break
                    if item_map.get(row.item_id) not in [r[1] for r in result]:
                        user_recs.append((getattr(row, userCol), getattr(row, itemCol), getattr(row, ratingCol)))
            result.extend(user_recs)

        # Формируем датафрейм с результатами, убирай из него фейковый товр
        df = pd.DataFrame(result, columns=[userCol, itemCol, ratingCol])
        df_sorted = df[df[itemCol] != 999999].sort_values([userCol, ratingCol], ascending=[True, False])
        lightfm_recs_topk = df_sorted.groupby(userCol).head(k)

        return lightfm_recs_topk


class MatrixTransformation:
    """Делает user-item-матрицу и формирует вспомогательные словари"""
    
    def __init__(self, data):
        self.user_item_matrix = self.prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id =\
            self.prepare_dicts(self.user_item_matrix)

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

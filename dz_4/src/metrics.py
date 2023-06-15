import numpy as np

def hit_rate(recommended_list, bought_list):
    flags = np.isin(np.array(recommended_list), np.array(bought_list))
    hit_rate = int(flags.sum() > 0)
    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    flags = np.isin(np.array(recommended_list[:k]), np.array(bought_list))
    hit_rate = int(flags.sum() > 0)
    return hit_rate


def precision(recommended_list, bought_list):
    flags = np.isin(np.array(bought_list), np.array(recommended_list))
    return flags.sum() / len(recommended_list)


def precision_at_k(recommended_list, bought_list, k):
    flags = np.isin(np.array(recommended_list)[:k], np.array(bought_list))
    return flags.sum() / len(recommended_list)


def average_precision_at_k(recommended_list, bought_list, k=5):
    flags = np.isin(np.array(recommended_list), np.array(bought_list))
    if sum(flags) == 0:
        return 0
    sum_ = 0
    for i in range(k):
        if flags[i]:
            p_k = precision_at_k(recommended_list, bought_list, k=i+1)
            sum_ += p_k
    return sum_ / k


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)  # Тут нет [:k] !!
    recommended_list = np.array(recommended_list[:k])
    prices_recommended = np.array(prices_recommended[:k])
    flags = np.isin(bought_list, recommended_list)
    return (flags * prices_recommended).sum() / prices_recommended.sum()


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)
    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    flags = np.isin(np.array(recommended_list)[:k], np.array(bought_list))
    return flags.sum() / len(bought_list)


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    flags = np.isin(np.array(recommended_list)[:k], np.array(bought_list))
    return (flags * np.array(prices_recommended)[:k]).sum() / np.array(prices_bought).sum()


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(recommended_list, bought_list)
    if sum(flags) == 0:
        return 0
    sum_ = 0
    for i in range(k):
        if flags[i]:
            p_k = precision_at_k(recommended_list, bought_list, k=i + 1)
            sum_ += p_k
    result = sum_ / k
    return result


def mrr_at_k(recommended_list, bought_list, k):
    flags = np.isin(np.array(recommended_list)[:k], np.array(bought_list))
    if flags.sum() == 0:
        return 0
    rank = np.where(flags == True)[0][0] + 1 # np.where - индекс первого релевантного товара в списке рекомендаций
    return 1 / rank


def ndcg_at_k(recommended_list, bought_list, k=5):
    dcg_at_k, idcg_at_k = 0, 0
    flags = np.isin(np.array(recommended_list)[:k], np.array(bought_list))
    if sum(flags) == 0:
        return 0
    # находим DCG и IDCG
    for i in range(len(flags)):
        if flags[i]:
            dcg_at_k += 1 / np.log2(i+2) # числитель: считаем только совпадения, 2 - константа
        idcg_at_k += 1 / np.log2(i+2)    # знаменатель: считаем все
    # находим NDCG
    if idcg_at_k == 0:                   # если IDCG равен 0
        ndcg_at_k = 0                    # то возвращаем 0
    else:
        ndcg_at_k = dcg_at_k/idcg_at_k
    return ndcg_at_k

# бейзлайны


def random_recommendation(items, n=5):
    """Случайные рекоммендации"""
    recs = np.random.choice(np.array(items), size=n, replace=False)
    return recs.tolist()


def popularity_recommendation(data, n):
    """Топ-n популярных товаров"""
    popular = data.groupby('item_id')['sales_value'].sum().reset_index()
    popular.sort_values('sales_value', ascending=False, inplace=True)
    recs = popular.head(n)['item_id']
    return recs.tolist()


def weighted_random_recommendation(iw, n):
    """Случайные рекоммендации с учетом весов всех item"""
    recs = np.random.choice(iw['item_id'], n, p=iw['item_weight'], replace=False)
    return recs.tolist()


if __name__ == '__main__':
    a = [5, 10, 5, 4, 9, 44, 8, 1, 7, 3]
    b = [3, 8, 5, 6, 4]
    print(recall_at_k(b, a, k=5))

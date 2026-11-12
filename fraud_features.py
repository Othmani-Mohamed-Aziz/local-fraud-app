import numpy as np
import pandas as pd

CATEGORIES = ['entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos',
              'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos', 'personal_care',
              'shopping_net', 'shopping_pos', 'travel']

FEATURES = (['amt', 'amt_ratio_30d', 'heures_depuis_prec', 'cnt_1h', 'cnt_24h', 'dist_km',
             'hour_sin', 'hour_cos', 'dow', 'is_weekend', 'is_night', 'age', 'city_pop', 'gender_M']
            + ['cat_' + c for c in CATEGORIES])

LIBELLES = {
    'amt': 'Montant de la transaction',
    'amt_ratio_30d': 'Montant rapporté à la moyenne des 30 derniers jours',
    'heures_depuis_prec': 'Temps écoulé depuis la transaction précédente (h)',
    'cnt_1h': 'Nb de transactions sur la dernière heure',
    'cnt_24h': 'Nb de transactions sur les dernières 24 h',
    'dist_km': 'Distance client – marchand (km)',
    'hour_sin': "Heure de la journée (composante sinus)",
    'hour_cos': "Heure de la journée (composante cosinus)",
    'dow': 'Jour de la semaine',
    'is_weekend': 'Transaction le week-end',
    'is_night': 'Transaction de nuit (22h-3h)',
    'age': 'Âge du porteur',
    'city_pop': 'Population de la ville du client',
    'gender_M': 'Genre (homme)',
}
LIBELLES.update({'cat_' + c: 'Catégorie marchand : ' + c for c in CATEGORIES})


def derive(df):
    """Construit la matrice de variables à partir des colonnes de base :
    ts, amt, category, gender, city_pop, dist_km, age, cnt_1h, cnt_24h,
    mean_amt_30d (peut être NaN), heures_depuis_prec (peut être NaN)."""
    ts = pd.to_datetime(df['ts'])
    out = pd.DataFrame(index=df.index)
    out['amt'] = df['amt'].astype(float)
    ratio = out['amt'] / df['mean_amt_30d'].astype(float)
    out['amt_ratio_30d'] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out['heures_depuis_prec'] = df['heures_depuis_prec'].astype(float).fillna(24 * 30)
    out['cnt_1h'] = df['cnt_1h'].astype(float)
    out['cnt_24h'] = df['cnt_24h'].astype(float)
    out['dist_km'] = df['dist_km'].astype(float)
    h = ts.dt.hour + ts.dt.minute / 60.0
    out['hour_sin'] = np.sin(2 * np.pi * h / 24)      # codage cyclique de l'heure
    out['hour_cos'] = np.cos(2 * np.pi * h / 24)
    out['dow'] = ts.dt.dayofweek
    out['is_weekend'] = (out['dow'] >= 5).astype(int)
    out['is_night'] = ((ts.dt.hour >= 22) | (ts.dt.hour <= 3)).astype(int)
    out['age'] = df['age'].astype(float)
    out['city_pop'] = df['city_pop'].astype(float)
    out['gender_M'] = (df['gender'] == 'M').astype(int)
    for c in CATEGORIES:
        out['cat_' + c] = (df['category'] == c).astype(int)
    return out[FEATURES]


def codes_motifs(contrib_row, values_row, n=5):
    """Codes motifs : les n variables qui poussent le plus le score vers « fraude ».
    contrib_row : contributions (xgboost pred_contribs, dernier élément = biais)."""
    nf = len(FEATURES)
    c = np.asarray(contrib_row)[:nf]
    ordre = np.argsort(-c)
    res = [i for i in ordre if c[i] > 0][:n]
    for i in ordre:                     # on complète jusqu'à 3 motifs minimum
        if len(res) >= 3:
            break
        if i not in res:
            res.append(i)
    return [{'code': f'M{rang + 1}', 'variable': FEATURES[i], 'motif': LIBELLES[FEATURES[i]],
             'valeur': round(float(values_row[i]), 3), 'contribution': round(float(c[i]), 3)}
            for rang, i in enumerate(res)]

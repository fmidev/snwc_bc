import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from skopt.space import Real, Integer
from skopt import dump
from skopt import BayesSearchCV
import xgboost as xgb
import time
import sys, getopt
from XGBmodify import modify_fcdate
from sklearn.model_selection import PredefinedSplit

def main():
    options, remainder = getopt.getopt(sys.argv[1:], [], ['param=', 'help'])

    for opt, arg in options:
        if opt == '--help':
            print('param=T2m, WS, WG or RH')
            exit()
        elif opt == '--param':
            param = arg

    print(f"Parameter selected: {param}")

    df = pd.read_feather('training_data.ftr')
    #df['year'] = df['fcdate'].dt.year
    #df['hour'] = df['fcdate'].dt.hour

    #print(df.groupby(['year', 'hour']).size())
    print("Data loaded:", len(df))
    print("fcdate range:", df['fcdate'].min(), "-", df['fcdate'].max())
    #print(df['fcdate'])
    # print sorted unique fcdate
    df['fcdate'] = pd.to_datetime(df['fcdate'], utc=True)

    selected_hours = [0, 4, 8, 12, 16, 20]
    df = df[df['fcdate'].dt.hour.isin(selected_hours)].copy()
    print("Data 1/3:", len(df))
    print("fcdate range:", df['fcdate'].min(), "-", df['fcdate'].max())
    data2 = df.copy()

    date_ranges = [
    ('2020-01-01', '2020-12-31'),
    ('2021-01-01', '2021-12-31'),
    ('2022-01-01', '2022-12-31'),
    ('2023-01-01', '2023-06-30'),
    ]

    data = modify_fcdate(df, param)
    data = data[(data.leadtime != 0) & (data.leadtime != 1)]

    if param == 'WS':
        data = data[data['WS_PT10M_AVG'] < 45]
    elif param == 'WG':
        data = data[data['WG_PT1H_MAX'] < 60]

    y = data.iloc[:, 12]

    if param == 'T2m':
        remove = ['SID', 'validdate', 'TA_PT1M_AVG', 'Tero', 'T0bias']
    elif param == 'WS':
        remove = ['SID', 'validdate', 'WS_PT10M_AVG', 'WSero', 'WS0bias']
    elif param == 'RH':
        remove = ['SID', 'validdate', 'RH_PT1M_AVG', 'RHero', 'RH0bias']
    elif param == 'WG':
        remove = ['SID', 'validdate', 'WG_PT1H_MAX', 'WGero', 'WG0bias']

    X = data.drop(remove, axis=1)
    print("fcdate range:", X['fcdate'].min(), "-", X['fcdate'].max())

    split_indices = -1 * np.ones(len(X), dtype=int)
    for i, (start, end) in enumerate(date_ranges):
        mask = (X['fcdate'] >= start) & (X['fcdate'] < end)
        split_indices[mask] = i

    # Only assign test blocks (train = -1)
    psplit = PredefinedSplit(test_fold=split_indices)

    print("\nTimeSeries CV block ranges:")
    for i, (train_idx, test_idx) in enumerate(psplit.split(X)):
        train_start = X.iloc[train_idx[0]]['fcdate']
        train_end = X.iloc[train_idx[-1]]['fcdate']
        test_start = X.iloc[test_idx[0]]['fcdate']
        test_end = X.iloc[test_idx[-1]]['fcdate']
        print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        print(f"Fold {i + 1}: Train {train_start}–{train_end}, Test {test_start}–{test_end}")

    space = {
        'max_depth': Integer(15,20), #  (5, 20),
        'learning_rate': Real(0.05, 0.1, "uniform"), #Real(0.05, 0.55, "uniform"),
        'colsample_bytree': Real(0.5, 1, "uniform"), #Real(0.1, 1, 'uniform'),
        'subsample': Real(0.7, 1.0, "uniform"), #Real(0.4, 1.0, "uniform"),
        'min_child_weight': Integer(6, 10) #Integer(1, 10)
    }

    model = xgb.XGBRegressor(
        #tree_method= 'hist', #'gpu_hist',
        #predictor= 'cpu_predictor' #'gpu_predictor',
        tree_method= 'hist',
        predictor= 'cpu_predictor',
        random_state=10,
        #gpu_id=0
    )

    bsearch = BayesSearchCV(
        estimator=model,
        search_spaces=space,
        scoring='neg_mean_absolute_error',
        n_jobs= -1, #"-1,
        n_iter=35,
        cv=psplit,
        optimizer_kwargs={'acq_func_kwargs': {"xi": 0.01, "kappa": 2}, 'n_initial_points': 10}
    )

    start_time = time.time()
    # drop fcdate from X
    X = X.drop(['fcdate'], axis=1)
    bsearch.fit(X, y)
    # Print per-fold test scores (neg_mean_absolute_error, higher is better)
    print("\nCross-validation scores:")
    mean_test_scores = bsearch.cv_results_['mean_test_score']
    std_test_scores = bsearch.cv_results_['std_test_score']
    params = bsearch.cv_results_['params']

    for i, (mean, std, p) in enumerate(zip(mean_test_scores, std_test_scores, params)):
        print(f"Iteration {i + 1}: MAE = {-mean:.4f} ± {std:.4f}, Params: {p}")
    print(f"\nBest Score: {bsearch.best_score_}")
    print("Best Parameters:", bsearch.best_params_)
    print(f"Search duration: {time.time() - start_time:.2f} seconds")

    dump(bsearch, f'results_{param}_gpu.pkl')
    print(f"Saved to results_{param}_gpu.pkl")

if __name__ == "__main__":
    main()

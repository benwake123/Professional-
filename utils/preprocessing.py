from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_walk_forward(
    dates, train_years: int = 2, val_months: int = 3, test_months: int = 3
) -> List[Tuple[pd.Series, pd.Series, pd.Series]]:
    dates = pd.Series(sorted(pd.to_datetime(pd.Series(dates).unique())))
    windows = []
    start_idx = 0

    while start_idx < len(dates) - 1:
        train_start = dates.iloc[start_idx]
        train_end = train_start + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(months=val_months)
        test_end = val_end + pd.DateOffset(months=test_months)

        tr = dates[(dates >= train_start) & (dates < train_end)]
        va = dates[(dates >= train_end) & (dates < val_end)]
        te = dates[(dates >= val_end) & (dates < test_end)]

        if len(te) == 0:
            break

        windows.append((tr, va, te))
        start_idx = dates.searchsorted(val_end)

    return windows


def fit_transform_scaler(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols].values)
    x_test = scaler.transform(test_df[feature_cols].values)
    return x_train, x_test

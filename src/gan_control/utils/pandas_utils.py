# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd


def get_kmin(main_df, column_name='distance', k=5):
    main_df = main_df.copy(deep=True)
    kmin_df = pd.DataFrame(columns=list(main_df.columns))
    for i in range(k):
        min_index = main_df[column_name].idxmin()
        kmin_df = kmin_df.append(main_df.iloc[min_index].copy(deep=True), ignore_index=True)
        main_df.drop(min_index, inplace=True)
        main_df.reset_index(inplace=True, drop=True)
    return kmin_df
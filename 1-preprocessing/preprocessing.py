import os
import numpy as np
import pandas as pd
import json
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split

def merge_datasets(output_file, *input_files):
    """Merge columns from input datasets into a single output dataset.

    Use the Date column as the key for merging.

    Args:
        output_file: Output csv file.
        *input_files: Input csv files.

    Returns:
        None.
    """

    # Get min and max for the Date for each dataset
    date_ranges = []
    for file in input_files:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        date_ranges.append((min_date, max_date))

    # Find the smallest min and the largest max
    overall_min = min(start for start, end in date_ranges)
    overall_max = max(end for start, end in date_ranges)

    # Generate a Date column with a range of dates from min to max
    full_date_range = pd.date_range(start=overall_min, end=overall_max, freq='D')
    output_df = pd.DataFrame({'Date': full_date_range})
    output_df.set_index('Date', inplace=True)

    # Copy the columns from the input files in to the output file
    # matching by Date
    for file in input_files:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        output_df = output_df.join(df, how='left')

    output_df.reset_index(inplace=True)

    # Remove all rows with missing values
    output_df.dropna(inplace=True)

    # Save output dataset
    output_df.to_csv(output_file, index=False)

def remove_variables_with_missing_values(input_file, output_file):
    """Remove variables with missing values.

    Args:
        input_file: Input csv file.
        output_file: Output csv file.

    Returns:
        None.

    """

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Drop columns with missing values
    df_selected = df.dropna(axis=1, how='any')

    # Save the selected data to a new CSV file
    df_selected.to_csv(output_file, index=False)

def forward_fill(input_file, output_file):
    """Forward fill missing dates and values.

    Args:
        input_file: Input csv file.
        output_file: Output csv file.

    Returns:
        None.

    """

    # Load the CSV file
    data = pd.read_csv(input_file, index_col='Date', parse_dates=True)

    # Create a complete date range
    start_date = data.index.min()
    end_date = data.index.max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex the DataFrame
    data_reindexed = data.reindex(date_range)

    # Forward fill the values
    data_forward_filled = data_reindexed.ffill()

    # Save the updated DataFrame to a CSV file
    data_forward_filled.to_csv(output_file, index_label='Date')

def interpolate(input_file, output_file):
    """Interpolate missing dates and values.

    Args:
        input_file: Input csv file.
        output_file: Output csv file.

    Returns:
        None.

    """

    # Load the CSV file
    data = pd.read_csv(input_file, index_col='Date', parse_dates=True)

    # Create a complete date range
    start_date = data.index.min()
    end_date = data.index.max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex the DataFrame
    data_reindexed = data.reindex(date_range)

    # Interpolate the values
    data_interpolated = data_reindexed.interpolate(method='linear')

    # Save the updated DataFrame to a CSV file
    data_interpolated.to_csv(output_file, index_label='Date')

def log_transform(input_file, output_file):
    """Log transform the data values.

    Args:
        input_file: Input csv file.
        output_file: Output csv file.

    Returns:
        None.

    """

    # Load the CSV file
    data = pd.read_csv(input_file, index_col='Date')

    # Apply log transformation
    output_df = np.log(data)

    # Save the transformed DataFrame to a CSV file
    output_df.to_csv(output_file, index=True, header=True)

def difference(input_file, output_file, critical_value=0.05):
    """Perform first order differencing of non-stationary columns.

    Args:
        input_file (str): Input csv file.
        output_file (str): Output csv file.
        critical_value (float, optional): Critical value for stationarity test. Defaults to 0.05.

    Returns:
        None.

    """

    # Load the input CSV file
    df = pd.read_csv(input_file)

    date = df.iloc[:, 0].copy()
    data = df.iloc[:, 1:].copy()

    for column in data.columns:
        adf_result = adfuller(data[column])
        if adf_result[1] > critical_value:
            data[column] = data[column].diff()

    output_df = pd.concat([date, data], axis=1)

    output_df.dropna(inplace=True)

    # Save the normalized DataFrame to a CSV file
    output_df.to_csv(output_file, index=False, header=True)

def winsor(input_file, output_file):
    """Winsor the values from the input file.
    Args:
        input_file (str): Input csv file.
        output_file (str): Output csv file.

    Returns:
        None.

    """

    # Load the input CSV file
    df = pd.read_csv(input_file)

    # Apply winsorization
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    # Save the transformed data to an output CSV file
    df.to_csv(output_file, index=False)

def normalize(input_file, output_file, output_range=(-1,1)):
    """Normalize the data using MinMaxScaler.

    Args:
        input_file: Input csv file.
        output_file: Output csv file.
        output_range (tuple, optional): The desired output range. Defaults to (-1, 1).

    Returns:
        None.

    """

    # Load the CSV file
    data = pd.read_csv(input_file, index_col='Date')

    header = data.columns
    data_values = data.values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=output_range)
    data_normalized = scaler.fit_transform(data_values)

    # Convert the normalized data back to a DataFrame
    data_normalized_df = pd.DataFrame(data_normalized, index=data.index, columns=header)

    # Save the normalized DataFrame to a CSV file
    data_normalized_df.to_csv(output_file, index=True, header=True)

def standardize(input_file, output_file, method='std'):
    """Standardize the data using StdScaler.
    Args:
        input_file (str): Input csv file.
        output_file (str): Output csv file.

    Returns:
        None.

    """

    # Load the CSV file
    data = pd.read_csv(input_file, index_col='Date')

    # Standardize the differenced data, excluding the header
    header = data.columns
    data_values = data.values
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_values)

    # Convert the standardized data back to a DataFrame
    data_standardized_df = pd.DataFrame(data_standardized, index=data.index, columns=header)

    # Save the standardized DataFrame to a CSV file
    data_standardized_df.to_csv(output_file, index=True, header=True)

def create_groupings(input_file, output_dir, groupings):
    """Create CSV files for each currency grouping.

    Args:
        input_file (str): Input CSV file.
        output_dir (str): Output directory.
        groupings (dict): Dictionary of currency groupings.

    Returns:
        None.

    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(input_file, index_col='Date')

    # Process each grouping and save the CSV files
    for currency, columns in groupings.items():
        print(f"currency = {currency}")
        output_file = os.path.join(output_dir, f'{currency}.csv')
        output_columns = columns + [currency]
        df[output_columns].to_csv(output_file)

def create_features(input_dir, output_dir, nr_lags, offset=1):
    """Create CSV files with time lags for each currency.

    Args:
        input_dir (str): Input directory containing CSV files.
        output_dir (str): Output directory.
        nr_lags (int): Number of time lags.
        offset (int): Offset from start time (default: 1).

    Returns:
        None.

    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each CSV file
    for file in [f for f in os.listdir(input_dir) if f.endswith('.csv')]:
        print(f"file = {file}")
        df = pd.read_csv(os.path.join(input_dir, file))
        base_name = os.path.splitext(file)[0]

        # Get actual column data references
        date_col = df.columns[0] # Date column
        predictor_cols = df.columns[1:-1]
        target_col = df.columns[-1] # Target column

        # Initialize with date column
        shifted_data = df[[date_col]].copy()

        # Create shifted predictor_cols and target
        for shift in range(offset, nr_lags+1):
            print(f"    shift = {shift}")
            # Add shifted predictor_cols
            shifted = df[predictor_cols].shift(shift)
            shifted.columns = [f'{col}_shift{shift}' for col in predictor_cols]
            shifted_data = pd.concat([shifted_data, shifted], axis=1)
            # Add shifted target
            shifted = df[target_col].shift(shift)
            shifted.name = f'{target_col}_shift{shift}'
            shifted_data = pd.concat([shifted_data, shifted], axis=1)

        # Add target column
        shifted_data[target_col] = df[target_col]

        # Clean and save
        shifted_data.dropna().to_csv(
            os.path.join(output_dir, f'{base_name}.csv'),
            index=False
        )

def create_features_alt(input_dir, output_dir, nr_lags, offset=1):
    """Create seperate CSV files with time lags for each currency.

    Args:
        input_dir (str): Input directory containing CSV files.
        output_dir (str): Output directory.
        nr_lags (int): Number of time lags.
        offset (int): Offset from start time (default: 1).

    Returns:
        None.

    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each CSV file
    for file in [f for f in os.listdir(input_dir) if f.endswith('.csv')]:
        print(f"file = {file}")
        df = pd.read_csv(os.path.join(input_dir, file))
        base_name = os.path.splitext(file)[0]

        os.makedirs(os.path.join(output_dir, base_name), exist_ok=True)

        # Get actual column data references
        date_col = df.columns[0] # Date column
        predictor_cols = df.columns[1:-1]
        target_col = df.columns[-1] # Target column

        # Create shifted predictor_cols and target
        for shift in range(offset, nr_lags+1):
            print(f"    shift = {shift}")

            # Add shifted predictor_cols
            shifted = df[predictor_cols].shift(shift)
            shifted.columns = [f'{col}_shift{shift}' for col in predictor_cols]

            renamed_target_col = df.iloc[:,-1].rename(target_col.rstrip(".1"))

            # Renaming target columns back to initial due to duplication in grouping
            # and deduplication in the creation of time lags
            data = pd.concat([df.iloc[:,0], shifted,
                             renamed_target_col],
                             axis=1)

            # Clean and save
            data.dropna().to_csv(
                os.path.join(output_dir, base_name, f'{shift}.csv'),
                index=False
            )

def select_features(input_dir, output_dir, cutoff=0.1):
    """Remove weakly correlated features from dataset.

    Args:
        input_dir (str): Input directory containing CSV files.
        output_dir (str): Output directory.
        cutoff (float, optional): Correlation value below which features are rejected.

    Returns:
        None.

    """

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    for subdir in os.listdir(input_dir):
        current_dir = os.path.join(input_dir, subdir)

        result_dir = os.path.join(output_dir, subdir)
        os.makedirs(result_dir, exist_ok=True)

        for file in [f for f in os.listdir(current_dir) if f.endswith('.csv')]:
            print(f"file = {file}")

            # Load data
            data = pd.read_csv(os.path.join(current_dir, file))

            # Get base file name
            base_name = os.path.splitext(file)[0]

            # Get feature column names excluding date and target columns
            feature_columns = data.columns[1:-1]

            # Get target column name
            target_column = data.columns[-1]

            # Create metadata output directory
            meta_dir = os.path.join(result_dir, f"{base_name}-meta")
            os.makedirs(meta_dir, exist_ok=True)

            # Calculate the Pearson correlation matrix
            # [x, y] - x selects rows; y selects columns
            corr_matrix = data.iloc[:, 1:].corr(method='pearson')

            # Save the full correlation matrix to a new CSV file
            corr_matrix.to_csv(os.path.join(meta_dir, "all-full.csv"))

            # Feature target decorrelation - remove abs(corr_coef) >= 0.8
            # corr_coefs = corr_matrix[target_column]
            # highly_correlated_with_target = corr_matrix.columns[
            #     (np.abs(corr_coefs) >= 0.8) &
            #     (corr_coefs != 1.0) &
            #     (corr_matrix.columns != target_column)
            # ].tolist()

            weakly_correlated_with_target = []
            for column in feature_columns:
                corr_coef = corr_matrix.loc[column, target_column]
                if abs(corr_coef) < cutoff:
                    weakly_correlated_with_target.append(column)

            # Drop features weakly correlated with target from data
            data = data.drop(columns=weakly_correlated_with_target)

            # Save dropped feature correlation matrix
            dropped_corr_matrix = corr_matrix.copy().filter(items=[target_column], axis=1).filter(items=weakly_correlated_with_target, axis=0)

            # Save the dropped feature correlation matrix to a new CSV file
            dropped_corr_matrix.to_csv(os.path.join(meta_dir, "target-dropped-full.csv"))

            # Drop features weakly correlated with target from correlation matrix
            corr_matrix = corr_matrix.drop(columns=weakly_correlated_with_target)
            corr_matrix = corr_matrix.drop(index=weakly_correlated_with_target)

            # Save reduced correlation matrix
            corr_matrix.to_csv(os.path.join(meta_dir, "reduced-full.csv"))

            # Save reduced feature target set to output file
            data.to_csv(os.path.join(result_dir, f"{base_name}.csv"), index=False)

def select_features_rfecv(estimator, input_dir, output_dir):
    """Perform feature selection using RFECV.

    Args:
        estimator (Estimator): To be used with RFECV.
        input_dir (str): Input directory containing CSV files.
        output_dir (str): Directory to save output files.

    Returns:
        None.
    """

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each CSV file
    for file in [f for f in os.listdir(input_dir) if f.endswith('.csv')]:
        print(f"file = {file}")

        # Load data
        data = pd.read_csv(os.path.join(input_dir, file))

        # Get base file name
        base_name = os.path.splitext(file)[0]

        # Create output directory
        result_dir = os.path.join(output_dir, base_name)
        os.makedirs(result_dir, exist_ok=True)

        # Get the target column name
        target_column = data.columns[-1]

        # Separate feature data and target data
        X = data.iloc[:, 1:-1]  # Feature data
        y = data[target_column]   # Target data

        # Initialize RFECV
        selector = RFECV(estimator=estimator,
                         step=1,
                         cv=5,
                         verbose=3)

        # Fit the selector to the data
        selector.fit(X, y)

        # Get the optimal number of features
        optimal_n_features = selector.n_features_

        # Get the selected features
        selected_features = X.columns[selector.support_]

        # Create a dataframe with the results
        results_df = pd.DataFrame({
            'feature': X.columns,
            'importance': selector.ranking_
        })

        # Sort the dataframe by importance
        results_df = results_df.sort_values(by='importance')

        # Save the feature importance ranking to a CSV file
        results_df.to_csv(f'{result_dir}/feature_ranking.csv', index=False)

        # Save the cross-validation scores to a CSV file
        cv_scores_df = pd.DataFrame({
            'number_of_features': range(1, len(selector.support_) + 1),
            'cross_validation_score': selector.cv_results_['mean_test_score']
        })
        cv_scores_df.to_csv(f'{result_dir}/feature_cv_scores.csv', index=False)

        # Create an output dataframe with the optimal features and the target column
        output_df = pd.concat([data.iloc[:, 0], X[selected_features].copy(), y], axis=1)
        output_df.to_csv(f'{result_dir}/optimal_features.csv', index=False)

def decorrelate(input_dir, output_dir, cutoff=0.9):
    """Reduce to one feature each set of strongly intercorrelated features from dataset.

    TODO - retain features with highest correlation with target variable.

    Args:
        input_dir (str): Input directory containing CSV files.
        output_dir (str): Output directory.
        cutoff (float, optional): Correlation value above which features are rejected.

    Returns:
        None.

    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    for subdir in os.listdir(input_dir):
        current_dir = os.path.join(input_dir, subdir)

        result_dir = os.path.join(output_dir, subdir)
        os.makedirs(result_dir, exist_ok=True)

        for file in [f for f in os.listdir(current_dir) if f.endswith('.csv')]:
            print(f"file = {file}")

            # Load data
            data = pd.read_csv(os.path.join(current_dir, file))

            # Get base file name
            base_name = os.path.splitext(file)[0]

            # Get feature column names excluding date and target columns
            feature_columns = data.columns[1:-1]

            # Create metadata output directory
            meta_dir = os.path.join(result_dir, f"{base_name}-meta")
            os.makedirs(meta_dir, exist_ok=True)

            # Calculate the Pearson correlation matrix
            # [x, y] - x selects rows; y selects columns
            corr_matrix = data.iloc[:, 1:-1].corr(method='pearson')

            # Save the feature correlation matrix to a new CSV file
            corr_matrix.to_csv(os.path.join(meta_dir, "all-features.csv"))

            # Cross feature decorrelation - remove abs(corr_coef) >= 0.8
            # corr_feature_dict = {
            #     feature: [
            #         other_feature for other_feature in feature_columns
            #         if other_feature != feature and
            #         abs(corr_matrix.loc[feature, other_feature]) >= 0.8 and
            #         corr_matrix.loc[feature, other_feature] != 1.0
            #     ]
            #     for feature in feature_columns
            # }

            corr_feature_dict = {}
            for feature_1 in feature_columns:
                corr_feature_dict[feature_1] = []
            for feature_1 in feature_columns:
                for feature_2 in feature_columns:
                    corr_coef = corr_matrix.loc[feature_1, feature_2]
                    if corr_coef == 1:
                        continue
                    if abs(corr_coef) > cutoff:
                        corr_feature_dict[feature_1].append(feature_2)

            # Save corr_feature_dict to json
            with open(os.path.join(meta_dir, "corr_features.json"), 'w') as f:
                json.dump(corr_feature_dict, f)

            # Pseudocode
            # Going through the dictionary once for every keys feature list
            # remove the corresponding keys from the dictionary.
            # Finally merge all remaining key feature lists
            # into a complete feature drop list.

            # Process correlation dictionary
            corr_features = set()
            for feature_1 in corr_feature_dict.keys():
                if feature_1 not in corr_features:
                    for feature_2 in corr_feature_dict[feature_1]:
                        corr_features.add(feature_2)

            # Drop strongly intercorrelated features from data
            data = data.drop(columns=corr_features)

            # Get retained features
            retained_features = feature_columns.drop(corr_features)

            # Save dropped vs retained feature correlation matrix
            dropped_vs_retained_corr_matrix = corr_matrix.copy().filter(items=retained_features, axis=1).filter(items=corr_features, axis=0)
            dropped_vs_retained_corr_matrix.to_csv(os.path.join(meta_dir, "dropped-vs-retained.csv"))

            # Save dropped vs dropped feature correlation matrix
            dropped_vs_dropped_corr_matrix = corr_matrix.copy().filter(items=corr_features, axis=1).filter(items=corr_features, axis=0)
            dropped_vs_dropped_corr_matrix.to_csv(os.path.join(meta_dir, "dropped-vs-dropped.csv"))

            # Drop strongly intercorrelated features from correlation matrix
            corr_matrix = corr_matrix.drop(columns=corr_features)
            corr_matrix = corr_matrix.drop(index=corr_features)

            # Save decorrelated features set correlation matrix
            corr_matrix.to_csv(os.path.join(meta_dir, "decorrelated.csv"))

            # Save reduced feature target set to output file
            data.to_csv(os.path.join(result_dir, f"{base_name}.csv"), index=False)

def split_data(input_dir, output_dir, nr_lags, train_size=0.7, test_size=0.3):
    """Split each currency exchange rate time series dataset into training and testing sets.

    Args:
        input_dir (str): Input directory containing CSV files.
        output_dir (str): Output directory.
        nr_lags (int): Number of time lags.
        train_size (float, optional): Proportion of data for training. Defaults to 0.7.
        test_size (float, optional): Proportion of data for testing. Defaults to 0.3.

    Returns:
        None.

    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.csv') and 'optimal_features' in filename:
                input_file = os.path.join(root, filename)
                subfolder_name = os.path.basename(root)

                # Load data
                data = pd.read_csv(input_file)
                X = data.iloc[:, 0:-1]  # Date + Feature data
                y = data.iloc[:, -1]   # Target data

                # Calculate the number of rows for the train set
                train_rows = int((len(X) - nr_lags) * train_size)

                # Calculate the number of rows for the sep and test set
                sep_test_rows = int((len(X) - nr_lags) * test_size)

                # Split data sequentially into training and sep_test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_rows, shuffle=False)

                # Trim the data to remove the nr_lags rows
                X_test = X_test.iloc[nr_lags:]
                y_test = y_test.iloc[nr_lags:]

                # Combine the X and y data for each split
                train_data = pd.concat([X_train, y_train], axis=1)
                test_data = pd.concat([X_test, y_test], axis=1)

                # Create the output directory for the current file
                file_output_dir = os.path.join(output_dir, subfolder_name)
                if not os.path.exists(file_output_dir):
                    os.makedirs(file_output_dir)

                # Save the combined data to CSV files
                train_data.to_csv(os.path.join(file_output_dir, 'train_data.csv'), index=False)
                test_data.to_csv(os.path.join(file_output_dir, 'test_data.csv'), index=False)

                print(filename)

def split_data_alt(input_dir, output_dir, train_size=0.7, test_size=0.3):
    """Split each currency exchange rate time series dataset into training and testing sets.

    Args:
        input_dir (str): Input directory containing CSV files.
        output_dir (str): Output directory.
        train_size (float, optional): Proportion of data for training. Defaults to 0.7.
        test_size (float, optional): Proportion of data for testing. Defaults to 0.3.

    Returns:
        None.

    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir in os.listdir(input_dir):
        for subsubdir in os.listdir(os.path.join(input_dir, subdir)):
            rel_path = os.path.join(subdir, subsubdir)
            current_dir = os.path.join(input_dir, rel_path)

            for file in [f for f in os.listdir(current_dir) if f.endswith('.csv')]:
                basename = os.path.splitext(file)[0]
                input_file = os.path.join(current_dir, file)

                split_output_dir = os.path.join(output_dir, rel_path, basename)

                nr_lags = int(basename)

                # Load data
                data = pd.read_csv(input_file)
                X = data.iloc[:, 0:-1]  # Date + Feature data
                y = data.iloc[:, -1]   # Target data

                # Calculate the number of rows for the train set
                train_rows = int((len(X) - nr_lags) * train_size)

                # Calculate the number of rows for the sep and test set
                sep_test_rows = int((len(X) - nr_lags) * test_size)

                # Split data sequentially into training and sep_test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_rows, shuffle=False)

                # Trim the data to remove the nr_lags rows
                X_test = X_test.iloc[nr_lags:]
                y_test = y_test.iloc[nr_lags:]

                # Combine the X and y data for each split
                train_data = pd.concat([X_train, y_train], axis=1)
                test_data = pd.concat([X_test, y_test], axis=1)

                # Create the output directory for the current file
                if not os.path.exists(split_output_dir):
                    os.makedirs(split_output_dir)

                # Save the combined data to CSV files
                train_data.to_csv(os.path.join(split_output_dir, 'train_data.csv'), index=False)
                test_data.to_csv(os.path.join(split_output_dir, 'test_data.csv'), index=False)

                print(input_file)


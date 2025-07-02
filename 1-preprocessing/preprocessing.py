import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    
    # Create the output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
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

def log_transform(input_file, output_file):
    """Log transform the data values.

    Args:
        input_file: Input csv file.
        output_file: Output csv file.

    Returns:
        None.

    """

    # Load the CSV file
    df = pd.read_csv(input_file, index_col='Date')
    
    df_log = df
    
    # Apply log transformation
    df_log = df.apply(lambda x: pd.Series(np.log(x)))
    
    # Save the normalized DataFrame to a CSV file
    df_log.to_csv(output_file, index=True, header=True)

def difference(data, critical_value=0.05):
    """Perform in-place first order differencing of non-stationary columns.

    Args:
        data (pandas.DataFrame): Dataset containing multiple columns.
        critical_value (float, optional): Critical value for stationarity test. Defaults to 0.05.

    Returns:
        None.

    """
    for column in data.columns:
        adf_result = adfuller(data[column])
        if adf_result[1] > critical_value:
            data[column] = data[column].diff()
    data.dropna(inplace=True)

def normalize(input_file, output_file):
    """Normalize the data using MinMaxScaler.

    Args:
        input_file: Input csv file.
        output_file: Output csv file.

    Returns:
        None.

    """

    # Load the CSV file
    data = pd.read_csv(input_file, index_col='Date')
    
    # Normalize the differenced data, excluding the header
    header = data.columns
    data_values = data.values
    scaler = MinMaxScaler(feature_range=(-1,1))
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
        output_file = os.path.join(output_dir, f'{currency}.csv')
        output_columns = columns + [currency]
        df[output_columns].to_csv(output_file)

import os
import pandas as pd

def create_time_lags(input_dir, output_dir, nr_lags):
    """Create CSV files with time lags for each currency.

    Args:
        input_dir (str): Input directory containing CSV files.
        output_dir (str): Output directory.
        nr_lags (int): Number of time lags.

    Returns:
        None.

    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each CSV file
    for file in [f for f in os.listdir(input_dir) if f.endswith('.csv')]:
        df = pd.read_csv(os.path.join(input_dir, file))
        base_name = os.path.splitext(file)[0]

        # Get actual column data references
        date_col = df.columns[0]  # Name of date column
        predictors = df.columns[1:-1]
        target_col = df.columns[-1]  # Name of target column

        # Initialize with date column DATA (not just name)
        shifted_data = df[[date_col]].copy()

        # Create shifted predictors
        for shift in range(1, nr_lags+1):
            shifted = df[predictors].shift(shift)
            shifted.columns = [f'{col}_shift{shift}' for col in predictors]
            shifted_data = pd.concat([shifted_data, shifted], axis=1)

        # Add target column DATA (not just name)
        shifted_data[target_col] = df[target_col]

        # Clean and save
        shifted_data.dropna().to_csv(
            os.path.join(output_dir, f'{base_name}.csv'),
            index=False
        )

import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

    # Iterate through the CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_file = os.path.join(input_dir, filename)

            # Load data
            data = pd.read_csv(input_file)
            X = data.iloc[:, :-1]  # Feature data
            y = data.iloc[:, -1]   # Target data

            # Calculate the number of rows for the train set
            train_rows = int((len(X) - nr_lags) * train_size)
            sep_test_rows = int((len(X) - nr_lags) * (1 - train_size))

            # Split data sequentially into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sep_test_rows + nr_lags, shuffle=False)

            # Trim the data to remove the nr_lags rows
            X_train = X_train.iloc[nr_lags:]
            y_train = y_train.iloc[nr_lags:]

            # Combine the X and y data for each split
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            # Create the output directory for the current file
            file_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir)

            # Save the combined data to CSV files
            train_data.to_csv(os.path.join(file_output_dir, 'train_data.csv'), index=False)
            test_data.to_csv(os.path.join(file_output_dir, 'test_data.csv'), index=False)

            print(filename)


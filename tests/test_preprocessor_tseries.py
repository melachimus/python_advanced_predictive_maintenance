import os
import sys
import json
import pytest
import pandas as pd
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Modules/Time_Series')))

from preprocessor_tseries import read_config, read_time_series, transform_dataset

# Tests for read_config
def test_read_config_valid():
    valid_config = {"amplitude_file": "sample.csv"}
    config_file = 'test_config.json'
    with open(config_file, 'w') as f:
        json.dump(valid_config, f)
    
    config = read_config(config_file)
    assert config == valid_config
    
    os.remove(config_file)

def test_read_config_non_existent():
    config = read_config('non_existent_config.json')
    assert config is None

def test_read_config_invalid_json():
    config_file = 'invalid_config.json'
    invalid_json_content = "{ am}"
    with open(config_file, 'w') as f:
        f.write(invalid_json_content)
    
    config = read_config(config_file)
    assert config is None

    os.remove(config_file)

# Tests for read_time_series
def test_read_time_series_valid():
    csv_content = """time,file_name,Label,amplitude
    1,A,yes,0.1
    2,A,yes,0.2
    1,B,no,0.3
    2,B,no,0.4"""
    
    csv_file = 'test_sample.csv'
    with open(csv_file, 'w') as f:
        f.write(csv_content)
    
    df = read_time_series(csv_file)
    assert df.shape == (4, 4)

    os.remove(csv_file)

def test_read_time_series_non_existent():
    df = read_time_series('non_existent_sample.csv')
    assert df is None

# Tests for transform_dataset
def test_transform_dataset_different_length_columns():
    csv_content = """time,sample,target,value,feature
    1,A,yes,1,0
    2,A,yes,2,0
    1,A,yes,3,1
    2,A,yes,4,1
    1,B,no,5,0
    2,B,no,6,0
    1,B,no,7,1
    2,B,no,8,1
    1,C,yes,9,0
    2,C,yes,10,0
    3,C,yes,11,0
    1,C,yes,12,1
    2,C,yes,13,1
    3,C,yes,14,1"""
    
    df = pd.read_csv(StringIO(csv_content))
    
    transformed_df = transform_dataset(df)
    
    expected_columns = ['sample', 'feature', 0, 1, 2, 'target']
    assert list(transformed_df.columns) == expected_columns

def test_transform_dataset_different_length_row_count():
    csv_content = """time,sample,target,value,feature
    1,A,yes,1,0
    2,A,yes,2,0
    1,A,yes,3,1
    2,A,yes,4,1
    1,B,no,5,0
    2,B,no,6,0
    1,B,no,7,1
    2,B,no,8,1
    1,C,yes,9,0
    2,C,yes,10,0
    3,C,yes,11,0
    1,C,yes,12,1
    2,C,yes,13,1
    3,C,yes,14,1"""
    
    df = pd.read_csv(StringIO(csv_content))
    
    transformed_df = transform_dataset(df)
    
    assert transformed_df.shape[0] == 6  # We expect 6 unique samples

def test_transform_dataset_different_length_values():
    csv_content = """time,sample,target,value,feature
    1,A,yes,1,0
    2,A,yes,2,0
    1,A,yes,3,1
    2,A,yes,4,1
    1,B,no,5,0
    2,B,no,6,0
    1,B,no,7,1
    2,B,no,8,1
    1,C,yes,9,0
    2,C,yes,10,0
    3,C,yes,11,0
    1,C,yes,12,1
    2,C,yes,13,1
    3,C,yes,14,1"""
    
    df = pd.read_csv(StringIO(csv_content))
    
    transformed_df = transform_dataset(df)
    
    assert transformed_df.iloc[0, 0] == 0 # sample, first row
    assert transformed_df.iloc[0, 1] == 0 # feature, fist row
    assert transformed_df.iloc[0, 2] == 1 # time 0, first row
    assert transformed_df.iloc[0, 3] == 2 # time 1, first row
    assert transformed_df.iloc[0, 4] == 0 # time 2, first row
    assert transformed_df.iloc[0, 5] == 'yes' # target, first row
    assert transformed_df.iloc[4, 4] == 11 # time 2, fifth row
    assert transformed_df.iloc[3, 1] == 1 # feature, fourth row
    assert transformed_df.iloc[2, 5] == 'no' # target, third row

    def test_transform_dataset_no_feature_column():
        """time,sample,target,value,not_feature
    1,A,yes,1,0
    2,A,yes,2,0
    1,A,yes,1,1
    2,A,yes,2,1
    1,B,no,5,0
    2,B,no,6,0
    1,B,no,5,1
    2,B,no,6,1
    1,C,yes,9,0
    2,C,yes,10,0
    3,C,yes,11,0
    1,C,yes,9,1
    2,C,yes,10,1
    3,C,yes,11,1"""
        
    df = pd.read_csv(StringIO(csv_content))
        
    transformed_df = transform_dataset(df)
        
    expected_columns = ['sample', 'feature', 0, 1, 2, 'target']
    assert list(transformed_df.columns) == expected_columns
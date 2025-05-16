import sys
import os
import pandas as pd
from pathlib import Path
import numpy as np
import random
import glob
import gc
from copy import copy
import shutil

def get_data(path, fillna=True,sorted=False):
    if path.split(".")[-1] == "parquet" or bool(glob.glob(os.path.join(path, '*.parquet'))):
        df_joined = (
            pd.read_parquet(
                path,
                engine="pyarrow"
            )
        )
    else:
        print("[!] File type not supported")
        return pd.DataFrame()
    if pd.api.types.is_datetime64_any_dtype(df_joined['timestamp']):
        pass
    elif pd.api.types.is_string_dtype(df_joined['timestamp']):
        unique_len_timestamp_str = df_joined['timestamp'].str.len().unique()
        if len(unique_len_timestamp_str) > 1:
            print('[WARN]: Timestamp column contains different length of string, please check the format')
            df_joined['timestamp'] = df_joined['timestamp'].str[:19]
            print('[WARN]: Timestamp column has been truncated to 19 characters, in order to fit the format: YYYY-MM-DDTHH:MM:SS')
        elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 19):
            pass
        elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 32):
            df_joined['timestamp'] = df_joined['timestamp'].str[:26]
        elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 26):
            pass
        else:
            raise ValueError('Timestamp column contains invalid string length')
        df_joined['timestamp'] = pd.to_datetime(df_joined['timestamp'])
    elif pd.api.types.is_integer_dtype(df_joined['timestamp']):
        df_joined['timestamp'] = pd.to_datetime(df_joined['timestamp'])
    else:
        raise ValueError('Timestamp column must be datetime64 or string or int')
    if sorted:
        df_joined= df_joined.sort_values(by=["timestamp"])
    return df_joined


def read_failures(path):
    df = pd.read_csv(path, engine='pyarrow')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df['timestamp'] = df['timestamp'].dt.round('min')
    df['timestamp'] = df['timestamp'].astype('datetime64[ms]')
    df = df[~df['xid'].isin([-1])]
    df = df.groupby(['timestamp', 'hostname']).first().reset_index()
    df['yymmdd'] = df['timestamp'].dt.strftime('%Y%m%d')
    df['label'] = 1
    if df.duplicated(subset=['timestamp', 'hostname']).any():
        print("Failures: The combination of 'timestamp' and 'hostname' is not unique.")
        df = df.groupby(['timestamp', 'hostname']).first().reset_index()
    else:
        print("Failures: (> . <) The combination of 'timestamp' and 'hostname' is unique.")
    df.sort_values(by=['timestamp', 'hostname'], inplace=True)
    return df


def merge_data_with_failures(df_raw, df_fail, tol, output_file_path):
    old_cols = df_raw.columns.tolist()
    old_cols = [col for col in old_cols if col not in ['label', 'xid']]
    if df_raw['yymmdd'].nunique() == 1:
        file_date_str = df_raw['yymmdd'].unique()[0]
        df_fail_cut = df_fail[df_fail['yymmdd'] == file_date_str].copy()
    df = pd.merge_asof(df_raw[old_cols], 
                        df_fail_cut[['timestamp', 'hostname', 'label','xid']], 
                        on='timestamp', 
                        by='hostname',
                        tolerance=pd.Timedelta(tol),
                        direction='nearest')
    df['label'] = df['label'].fillna(0)     
    df['xid'] = df['xid'].fillna(-1)   
    df.sort_values(by=["timestamp","hostname"], inplace=True)
    df = df[old_cols+['label','xid']]
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved processed data to: {output_path}")

def relabel_anomaly(tol):
    Re_label_failure = True
    output_base_dir = Path('data/olcf_task')
    failures_path = 'data/failures.csv'
    df_fail = read_failures(path=failures_path)
    files = list(output_base_dir.rglob('*.parquet'))
    for i, file in enumerate(files):
        df = get_data(path=str(file))
        df = df.sort_values(by=["timestamp","hostname"])
        merge_data_with_failures(df, df_fail, tol=tol,output_file_path=file)

def relabel_test_anomaly(tol):
    failures_path = 'data/failures.csv'
    df_fail = read_failures(path=failures_path)
    file_path = 'data/olcf_task/test_anomaly.parquet'
    df = get_data(path=str(file_path))
    df = df.sort_values(by=["timestamp","hostname"])
    output_path = file_path[:-8] + f'_{tol}.parquet'
    merge_data_with_failures(df, df_fail, tol=tol,output_file_path=output_path)

if __name__=='__main__':
    relabel_test_anomaly(tol='60min')
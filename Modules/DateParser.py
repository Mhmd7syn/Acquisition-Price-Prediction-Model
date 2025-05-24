import pandas as pd
import numpy as np

def smart_date_parser(date_str):
    """
    Safely parses a date string into a pandas datetime object.
    Handles common invalid inputs and multiple date formats.
    """
    if pd.isna(date_str):
        return pd.NaT

    date_str = str(date_str).strip()
    date_str = date_str.replace('//', '/').replace('--', '-').replace('..', '.').strip()

    date_formats = [
        "%d/%m/%Y", "%m/%d/%Y",
    ]

    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt, exact=True)
        except (ValueError, TypeError):
            continue
    
    print("Not Valid Format:", date_str)


def clean_date_columns(df):
    date_columns = [
        'Deal_announced_on',
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(smart_date_parser)
            df[col] = pd.to_datetime(df[col], errors='coerce')

            df[f'{col}_Month'] = df[col].dt.month  # Extract Month
            df[f'{col}_Day'] = df[col].dt.day # Extract Day


    # Drop original date columns after extracting features
    df.drop(columns=date_columns, inplace=True)


def date_validate(df_merged):
    # Validate and correct year values (2000-curr_year):
    current_year = pd.Timestamp.now().year
    valid_years = range(1000, current_year + 1)
    year_columns = ['Acquired_Year_Founded',  'Acquiring_Year_Founded', 'Acquiring_IPO', 'Acquiring_Number_of_Employees_year_of_last_update']

    for col in year_columns:
        for i, year in enumerate(df_merged[col]):
            if pd.isna(year):
                continue
            try:
                year = int(year)
                if year not in valid_years:
                    if 0 <= year < 100:  # 23 → 2023
                        df_merged.at[i, col] = 2000 + year
                    elif year >= 2100:    # 2103 → 2013
                        df_merged.at[i, col] = year - 100
                    else:
                        df_merged.at[i, col] = np.nan
            except (ValueError, TypeError):
                df_merged.at[i, col] = np.nan
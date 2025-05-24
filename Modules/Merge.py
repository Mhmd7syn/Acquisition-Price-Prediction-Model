import pandas as pd

def merge(folder_path):
    acquisitions = pd.read_csv(folder_path + 'CSV Files\\' +'Acquisitions.csv')
    acquired = pd.read_csv(folder_path + 'CSV Files\\' + 'Acquired Tech Companies.csv')
    acquiring = pd.read_csv(folder_path + 'CSV Files\\' + 'Acquiring Tech Companies.csv')
    founders = pd.read_csv(folder_path + 'CSV Files\\' + 'Founders and Board Members.csv')

    # Merge for acquiring companies
    merged_data = pd.merge(
        acquisitions,
        acquiring,
        left_on='Acquiring Company',  # Column name in acquisition data
        right_on='Acquiring Company',      # Column name in company details CSV
        how='left',
        suffixes=('', '_acquiring')
    )

    # Merge for acquired companies
    merged_data = pd.merge(
        merged_data,
        acquired,
        left_on='Acquired Company',   # Column name in acquisition data
        right_on='Company',      # Column name in company details CSV
        how='left',
        suffixes=('', '_acquired')
    )

    # Optional: Drop original name columns if needed
    merged_data = merged_data.drop(['Acquired Company'], axis=1)

    return merged_data, founders
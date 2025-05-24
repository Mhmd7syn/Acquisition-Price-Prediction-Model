import pandas as pd
import numpy as np

def numerical_reformatting(df_merged):
    # Clean 'Price' column: remove unwanted chars ($,[,!) and convert to numeric
    df_merged['Price'] = pd.to_numeric(
        df_merged['Price'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip(),
        errors='coerce')

    # Remove commas and convert to nullable integer
    df_merged['Acquiring_Number_of_Employees'] = df_merged['Acquiring_Number_of_Employees'].astype(str).str.replace(',', '', regex=True).astype(float).astype(pd.Int64Dtype())
    df_merged['Acquiring_IPO'] = df_merged['Acquiring_IPO'].astype(str).replace('Not yet', np.nan).astype(float).astype(pd.Int64Dtype())


def ordinal_features(df_merged):
    status_map = {'Undisclosed': 0, np.nan:0, 'Pending': 1, 'Complete': 2}
    df_merged['status_encoded'] = df_merged['Status'].map(status_map)


def one_hot_encode_with_filters(df, columns_prefix, minThreshold, maxThreshold):
    for col in columns_prefix:
        for prefix in ['Acquiring_', 'Acquired_', '']:
            col_name = prefix + col
            if col_name not in df.columns:
                continue
                
            # Step 1: Split and handle missing values
            data = df[col_name].fillna('').astype(str)
            data = data.str.split(r',\s*')
            
            # Step 2: Create clean feature names with prefix
            one_hot = data.str.join('|').str.get_dummies()
            one_hot = one_hot.add_prefix(prefix + columns_prefix[col])
            
            # Step 3: Filter features by frequency
            freq = one_hot.mean()
            mask = (freq >= minThreshold) & (freq <= maxThreshold)
            one_hot = one_hot.loc[:, mask]
            print(col_name, "-> Extracted Features:", one_hot.columns)
            # Step 4: Concatenate with original dataframe
            df = pd.concat([df, one_hot], axis=1)

    return df


def multipleFeatures(df_merged, min_threshold = 0.0, max_threshold = 1.0):
    # Usage
    ohe_columns_prefix = {
        'Company': 'Company_',
        'City_HQ': 'City_',
        'State_Region_HQ': 'State_',
        'Market_Categories': 'category_',
        'Country_HQ': 'Country_',
        'Terms': 'Term_',
        'Founders': 'Founder_',
        'Board_Members': 'Member_',
        'Companies': 'Comp_'
    }

    df_merged['Terms'] = df_merged['Terms'].replace('Undisclosed', np.nan)
    df_merged = one_hot_encode_with_filters(
        df=df_merged,
        columns_prefix=ohe_columns_prefix,
        minThreshold= min_threshold, 
        maxThreshold = max_threshold
    )
    return df_merged
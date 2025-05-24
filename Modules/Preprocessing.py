from sklearn.impute import KNNImputer
from joblib import load

def renaming(df_merged, df_members):
    new_names = ['Acquisitions_ID', 'Acquiring_Company', 'Year_of_acquisition_announcement', 'Deal_announced_on', 'Price', 'Status', 'Terms', 'Acquisition_Profile_Link', 'Acquisition_News', 'Acquisition_News_Link', 'Acquiring_CrunchBase_Profile', 'Acquiring_Image', 'Acquiring_Tagline', 'Acquiring_Market_Categories', 'Acquiring_Year_Founded', 'Acquiring_IPO', 'Acquiring_Founders', 'Acquiring_Number_of_Employees', 'Acquiring_Number_of_Employees_year_of_last_update', 'Acquiring_Total_Funding', 'Acquiring_Number_of_Acquisitions', 'Acquiring_Board_Members', 'Acquiring_Address_HQ', 'Acquiring_City_HQ', 'Acquiring_State_Region_HQ', 'Acquiring_Country_HQ', 'Acquiring_Description', 'Acquiring_Homepage', 'Acquiring_Twitter', 'Acquired_Companies', 'Acquisitions_ID_acquiring', 'Acquiring_API', 'Acquired_Company', 'Acquired_CrunchBase_Profile', 'Acquired_Image', 'Acquired_Tagline', 'Acquired_Year_Founded', 'Acquired_Market_Categories', 'Acquired_Address_HQ', 'Acquired_City_HQ', 'Acquired_State_Region_HQ', 'Acquired_Country_HQ', 'Acquired_Description', 'Acquired_Homepage', 'Acquired_Twitter', 'Acquired_by', 'Acquisitions_ID_acquired', 'Acquired_API']
    df_merged.columns = new_names

    new_names = ['member_name', 'crunchbase_url', 'member_role', 'companies', 'member_image']
    df_members.columns = new_names


def null_numerical(df_merged, folder_path):
    imputer = load(folder_path + 'Models\\' + 'Knn_imputer.joblib')
    imputer_feature_names = load(folder_path + 'Models\\' + 'imputer_feature_names.joblib')

    df_merged[imputer_feature_names] = imputer.transform(df_merged[imputer_feature_names])
        
def feature_engineering(df_merged):
    # Preprocess selected columns by splitting comma-separated string values into lists
    for col in ['Acquiring_Market_Categories', 'Acquiring_Founders', 'Acquiring_Board_Members', 'Acquired_Companies', 'Acquired_Market_Categories']:
        df_merged[col] = df_merged[col].astype(str).str.split(r',\s*')
        df_merged[col + '_length'] = df_merged[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
    return df_merged
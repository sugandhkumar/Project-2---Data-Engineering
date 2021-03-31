# Import the necessary libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine 

def extract_and_merge(file1, file2, on = "id", how = "inner"):
    """
    Extracts DataFrames from the specified csv files and merges them
    
    Inputs:
    1. file1 (str) - specifies the path for the first file
    2. file2 (str) - specifies the path of the second file
    3. on (str or list)- specifies the key(s) on which the dataframes will be merged (default "id")
    4. how (str)- specifies the type of merge (default "inner")
    
    Output:
    1. df - The merged DataFrame   
    """
    
    on_list = []
    if type(on) == str:
        on_list.append(on)
        print("Input files have been {} joined on {}".format(how, on_list), "\n")
    elif type(on) ==list:
        on_list.extend(on)
        print("Input files have been {} joined on {}".format(how, on_list), "\n")
        
    df_1 = pd.read_csv(file1)
    df_2 = pd.read_csv(file2)
    
    df = df_1.merge(df_2, on=on_list, how=how)
    return df


def extract_columns(df, cols = "categories", delimiter = ";", retain_cols = False):
    """
    Extracts data from the specified column(s), based on the defined delimiter
    
    Inputs:
    1. df - DataFrame to extract and transform
    2. cols (string or list) - column(s) which need to be split and transformed (default "categories")
    3. delimiter (string) - the delimiter on which the specified columns would split (default ";")
    4. retain_cols (Boolean) - Specifies whether the original column has to be retained (default False)
    
    The function follows the following steps:
    1. Creates a DataFrame with expanded column from cols
    2. Renames the column appropriately and transforms the data into binary integers
    3. Merges the new DataFrame with the input df to create the final transformed df
    """
    
    cols_list = []
    if type(cols) == str:
        cols_list.append(cols)
        print("{} column(s) is being split by delimiter {}".format(cols_list, delimiter), "\n")
    elif type(cols) ==list:
        cols_list.extend(cols)
        print("{} column(s) is being split by delimiter {}".format(cols_list, delimiter), "\n")
        
    for col in cols_list:
        df_x = df[col].str.split(delimiter, expand=True)
        df_x_col_list = np.array(df_x.apply(lambda x: x.str[:-2]).iloc[:1, :]).tolist()[0]
        df_x.columns = df_x_col_list
        for x in df_x_col_list:
            df_x[x] = df_x[x].str[-1]
            df_x[x] = df_x[x].astype("int")
        if retain_cols:
            df = pd.concat([df, df_x], axis=1)
        else:
            df = pd.concat([df.drop(col, axis=1), df_x], axis=1)

    # Some 2's in the data. Replace that with 1 as 2 doesn't make sense for a flag            
    df.related.replace(2,1, inplace=True)
    
    return df
    
    
def dedup(df, keys = "id", keep= "first"):
    """
    Removes duplicates from the input dataframe using the provided key(s)
    
    Inputs:
    1. df - The input DataFrame
    2. keys (str or list) - Key(s) on which df would be deduped
    3. keep (str) - specifies drop_duplicates keep method
    
    Output:
    1. df - the deduped DataFrame
    """
    keys_list = []
    if type(keys) == str:
        keys_list.append(keys)
    elif type(keys) ==list:
        keys_list.extend(keys)
    
    print("dedup function removed {} duplicates on {} key(s)".format(df[df.duplicated(keys_list)].shape[0], keys_list), "\n")
    df.drop_duplicates(subset = keys_list, keep = keep, inplace=True)
    return df

def load_sql(df, db_name):
    """
    Functon to load df into a sql db using sqlalchemy
    
    Inputs:
    1. df - DataFrame to load
    2. db_name (str) - the name of the database and the table
    
    Outputs:
    None
    """
    
    engine = create_engine("sqlite:///{}".format(db_name))
    df.to_sql("Disaster_Response", engine, index=False, if_exists = "replace")
    print("DataFrame loaded to database {}. Table name Disaster_Response".format(db_name))
    
    
def main():
    """
    Execute the ETL pipeline
    """
    
    if len(sys.argv) == 4:
        
        file1, file2, db_name = sys.argv[1:]
        print('Loading files...\n    File1: {}, File2: {}'.format(file1, file2))
        
          
        #Create the merged dataframe to clean and transform
        print("\nMerging the files")
        df_merge = extract_and_merge(file1, file2)

        #Clean and Transform the merged dataframe
        print("\nTransforming the file..")
        df_transform = extract_columns(df_merge)

        #Dedup the cleaned dataframe
        df_load = dedup(df_transform)

        #load the dataframe to a SQl db
        print("\nCreating the Database: {}".format(db_name))
        load_sql(df_load, db_name)
        
        print("\nTable created succesfully..!")
        print("Shape of the new table: ", df_load.shape)

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
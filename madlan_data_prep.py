import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
from IPython.display import display
import openpyxl
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import StandardScaler

#file_name = "output_all_students_Train_v10.xlsx"
#df = pd.read_excel(file_name)
#display(df.head())

def prepare_data(df):
    df.columns = df.columns.str.strip() #clean up all unnecessary spaces
    df.dropna(subset=['price'], inplace=True) #delete all rows without price
    df.reset_index(drop=True, inplace=True) #reset index

    #Replace values in the 'price' column
    def convert_price(value):
        numeric_value = re.search(r'(\d[\d,]*)', str(value))
        if numeric_value:
            numeric_value = numeric_value.group(1).replace(',', '')
            return int(numeric_value)
        else:
            return None

    df['price'] = df['price'].apply(convert_price)

    #Clean Area value's 
    def clean_area_value(value):
        if isinstance(value, str):
            cleaned_value = re.sub(r'[^0-9.]', '', value)  # Remove non-numeric characters except dot
            try:
                return float(cleaned_value)
            except ValueError:
                return None
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return None

    df["Area"] = df["Area"].apply(clean_area_value)

    #display(df['Area'].unique()) #(df)
    #making columns price and area numeric
    #df['Area'] = df['Area'].str.extract(r'(\d+)').astype(float)
    #df['price'] = pd.to_numeric(df['price'], errors='coerce').astype(float)


    #Using RE to drop all 'סימני פיסוק'
    pattern = r'[^\u0590-\u05FF\s\d]'
    columns_to_clean = ['Street', 'city_area'] #'description'
    df[columns_to_clean] = df[columns_to_clean].replace(to_replace=pattern, value='', regex=True)
    df["description"] = df["description"].str.replace(r"[^A-Za-zא-ת0-9.\"']", " ", regex=True)


    #df["description"] = df["description"].str.replace(r"[^A-Za-zא-ת0-9.\"']", " ", regex=True)
    #df.description.unique()

    #Creating the new columns 'floor' and 'total_floors' and extracting the correct values to each column from the column 'floor_out_of'df['floor'] = np.nan
    df['floor'] = np.nan
    df['total_floors'] = np.nan
    df['floor_out_of'] = df['floor_out_of'].astype(str)
    matches = df['floor_out_of'].str.extract(r'(\d+).*?(\d+)')
    df['floor'] = matches[0].astype(float)
    df['total_floors'] = matches[1].astype(float)
    df.loc[df['type'].isin(["קוטג'", 'בית פרטי', 'מגרש', 'דו משפחתי', 'נחלה', "קוטג' טורי", 'דירת גן']), ['floor']] = 0
    #df = df.drop('floor_out_of', axis=1)

    #fill 0 where num_of_images is null
    df['num_of_images'].fillna(0, inplace=True)

    #replacing the values in entranceDate that are not dates with a built-in fixed date that we can identify later
    df['entranceDate'] = df['entranceDate'].replace('גמיש', '1996-09-13 00:00:00').replace('לא צויין', '1999-03-26 00:00:00').replace('מיידי', '2022-11-16 00:00:00')
    df['entranceDate'] = df['entranceDate'].apply(lambda x: '1996-09-13 00:00:00' if isinstance(x, str) and x.strip() == 'גמיש' else x)
    df['entranceDate'] = pd.to_datetime(df['entranceDate'])

    #making entranceDate categorial column
    above_year = 'above_year'
    months_6_12 = 'months_6_12'
    less_than_6_months = 'less_than_6_months'
    flexible = 'flexible'
    not_defined = 'not_defined'

    current_date = datetime.now()

    def replace_date(row):
        x = row['entranceDate']
        if x.year != 1996 and x.year != 1999 and x.year != 2022 and abs(current_date - x) > timedelta(days=365):
            return above_year
        elif x.year != 1996 and x.year != 1999 and x.year != 2022 and timedelta(days=182) <= abs(current_date - x) < timedelta(days=365):
            return months_6_12
        elif x.year != 1996 and x.year != 1999 and x.year != 2022 and current_date - x < timedelta(days=182):
            return less_than_6_months
        elif x == datetime(1996, 9, 13):
            return flexible
        elif x == datetime(1999, 3, 26): #or x == datetime(2022, 11, 16):
            return not_defined
        elif x.year == 2022:
            return less_than_6_months
        else:
            return x

    df['entranceDate'] = df.apply(replace_date, axis=1)

    #Cleaning room_number column
    replacement_mapping = {
        r'35': 3.5,
        r'5.5 חד׳': 5.5,
        r'4 חד׳': 4,
        r'2 חד׳': 2,
        r'3.5 חד׳': 3.5,
        r'5 חד׳': 5,
        r'3 חד׳': 3,
        r'6 חד׳': 6,
        r'6.5 חד׳': 6.5,
        r'4.5 חד׳': 4.5,
        r'2.5 חד׳': 2.5,
        r'8 חד׳': 8,
        r'7 חד׳': 7,
        r'-': np.nan,
        r'7.5 חד׳': 7.5,
        r'9.5 חד׳': 9.5,
        r"4 חד'": 4,
        r'10 חד׳': 10,
        r"3 חד'": 3,
        r"5 חד'": 5,
        r"6 חד'": 6,
        r"^\['6.5'\]$": 6.5,
        r"^\['3'\]$": 3,
        r"^\['4'\]$": 4,
        r"^\['4.5'\]$": 4.5,
        r"^\['5'\]$": 5,
        r"^\['7.5'\]$": 7.5,
        r"^\['7'\]$": 7,
        r"^\['6'\]$": 6
    }

    df['room_number'] = df['room_number'].replace(replacement_mapping, regex=True)
    df['room_number'] = df['room_number'].astype(float)


    ### Converting the columns 'hasElevator', 'hasParking', 'hasBars', 'hasStorage', 'hasAirCondition', 'hasBalcony', 'hasMamad' and 'handicapFriendly' to a boolean columns
    replace_values = [True, 'כן', 'יש', 'יש מעלית', 'yes', 'יש חניה', 'יש חנייה', 'יש סורגים', 'יש מחסן', 'יש מיזוג אויר', 'יש מיזוג אוויר', 'יש מרפסת', 'יש ממ"ד', 'יש ממ" ד', 'נגיש', 'נגיש לנכים', 'יש ממ״ד']

    df['hasElevator'] = df['hasElevator'].replace(replace_values, 1).astype(bool).astype(int)
    df['hasParking'] = df['hasParking'].replace(replace_values, 1).astype(bool).astype(int)
    df['hasBars'] = df['hasBars'].replace(replace_values, 1).astype(bool).astype(int)
    df['hasStorage'] = df['hasStorage'].replace(replace_values, 1).astype(bool).astype(int)
    df['hasAirCondition'] = df['hasAirCondition'].replace(replace_values, 1).astype(bool).astype(int)
    df['hasBalcony'] = df['hasBalcony'].replace(replace_values, 1).astype(bool).astype(int)
    df['hasMamad'] = df['hasMamad'].replace(replace_values, 1).astype(bool).astype(int)
    df['handicapFriendly'] = df['handicapFriendly'].replace(replace_values, 1).astype(bool).astype(int)

    replace_False_values = [False, 'לא', 'אין', 'אין מעלית', 'no', 'אין חניה', 'אין חנייה', 'אין סורגים', 'אין מחסן', 'אין מיזוג אויר', 'אין מיזוג אוויר', 'אין מרפסת', 'אין ממ"ד', 'אין ממ" ד', 'לא נגיש', 'לא נגיש לנכים', 'אין ממ״ד']

    df['hasElevator'] = df['hasElevator'].replace(replace_False_values, 0).astype(bool).astype(int)
    df['hasParking'] = df['hasParking'].replace(replace_False_values, 0).astype(bool).astype(int)
    df['hasBars'] = df['hasBars'].replace(replace_False_values, 0).astype(bool).astype(int)
    df['hasStorage'] = df['hasStorage'].replace(replace_False_values, 0).astype(bool).astype(int)
    df['hasAirCondition'] = df['hasAirCondition'].replace(replace_False_values, 0).astype(bool).astype(int)
    df['hasBalcony'] = df['hasBalcony'].replace(replace_False_values, 0).astype(bool).astype(int)
    df['hasMamad'] = df['hasMamad'].replace(replace_False_values, 0).astype(bool).astype(int)
    df['handicapFriendly'] = df['handicapFriendly'].replace(replace_False_values, 0).astype(bool).astype(int)

    #Preapering condition column to be categorial
    df["condition"] = df["condition"].replace(["לא צויין","nan", "None","False"], 'not_defind')
    df['condition'] = df['condition'].replace({'משופץ': 'renovated' , 'שמור': 'maintained' , 'חדש': 'new' , 'ישן': 'old' , 'דורש שיפוץ': 'requires_renovation'})

    #Cleaning City
    df["City"] = df["City"].replace('נהרייה', 'נהריה')

    #cleaning street column
    df['Street'] = df['Street'].replace('None', np.nan)
    df['Street'] = df['Street'].str.replace(r"\['(.*?)'\]", r"\1", regex=True)
    df['Street'] = df['Street'].str.replace(r"\[\'(.*?)\'\]", r"\1", regex=True)
    df['Street'] = df['Street'].str.replace(r"(\w) '", r"\1", regex=True)
    df['Street'] = df['Street'].str.replace(r"' (\w)", r"\1", regex=True)
    df['Street'] = df['Street'].str.replace('רמב ן', 'רמב"ן')
    df['Street'] = df['Street'].str.replace('רחבת חי ל', 'רחבת חיל')
    df['Street'] = df['Street'].str.replace('פינס 5', 'פינס')
    df['Street'] = df['Street'].str.strip()
    df['Street'] = df['Street'].str.replace('עין נטפים\n', 'עין נטפים') # Replace 'עין נטפים\n' with 'עין נטפים'


    df['city_area'] = df['city_area'].str.strip()

    #clean publishedDays
    df['publishedDays'] = df['publishedDays'].replace('60+', '60')
    df['publishedDays'] = df['publishedDays'].replace('None', np.nan)
    df['publishedDays'] = df['publishedDays'].replace('None ', np.nan)
    df['publishedDays'] = df['publishedDays'].replace('-', np.nan)
    df['publishedDays'] = df['publishedDays'].replace('חדש!', '0')
    df['publishedDays'] = df['publishedDays'].replace('חדש', '0')
    df['publishedDays'] = df['publishedDays'].replace('Nan', np.nan)

    #Making furniture column categorial
    replace_dict = {
        'חלקי': 'partial',
        'מלא': 'full',
        'אין': 'nothing',
        'לא צויין': 'not_defined'
    }

    df['furniture'].replace(replace_dict, inplace=True)

    #Sorting the columns before selecting the featuers
#passing_price_column = clean_df['price']  # Extract the 'price' column
#clean_df = clean_df.drop(columns=['price'])  # Drop the 'price' column from the DataFrame
#clean_df['price'] = passing_price_column  # Add the 'price' column back at the end
#display(clean_df.head())

#Creating vizulization
#Compute the correlation matrix

#+

#numerical_columns = ['Area', 'room_number', 'floor', 'hasElevator', 'hasBars', 'hasAirCondition', 'handicapFriendly', 'hasParking', 'hasBalcony', 'hasMamad', 'hasStorage', 'price']

#new_df = clean_df[numerical_columns].copy()

# Compute the correlation matrix on the encoded DataFrame
#correlation_matrix = new_df.corr()
#correlation_matrix = clean_df.corr()
#plt.figure(figsize=(16, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#plt.title('Correlation Matrix Heatmap')
#plt.show()

#Checking the predict for every numerical value
# Select the numerical columns
#columns = ['Area','room_number', 'floor', 'price', 'hasElevator', 'hasParking', 'hasBars', 'hasStorage', 'hasAirCondition', 'hasBalcony', 'hasMamad', 'handicapFriendly']

# Calculate the Predictive Power Score for each binary column
#feature_scores = {}
#for column in columns:
#    score = pps.score(clean_df, column, 'price')
#    feature_scores[column] = score['ppscore']
#sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True) # Sort the features by Predictive Power Score
#print("Predictive Power Scores:") # Print the features and their corresponding Predictive Power Scores
#for feature, score in sorted_features:
#    print(f"{feature}: {score}")

#Creating Chi-square test for categorical features
#from scipy.stats import chi2_contingency
#categorical_columns = ['City', 'type', 'city_area', 'condition', 'furniture', 'entranceDate']
#selected_features = []
#results = []
#print("Chi-square Test Results:")
#for column in categorical_columns:
#    contingency_table = pd.crosstab(clean_df[column], clean_df['price'])
#    chi2, p_value, _, _ = chi2_contingency(contingency_table)
#    results.append({'Feature': column, 'Chi2': chi2, 'P-value': p_value})
#    # Set a significance level (e.g., 0.05) to determine feature importance
#    if p_value < 0.05:
#        selected_features.append(column)
#print("Selected Categorical Features:", selected_features)
#print(pd.DataFrame(results))
    
    df.dropna(subset=['price'], inplace=True)
    df.dropna(subset=['Area'], inplace=True)
    df = df.drop_duplicates()
    columns_to_drop = ['Street', 'number_in_street', 'num_of_images', 'floor_out_of', 'hasElevator', 'hasBars', 'hasAirCondition', 'handicapFriendly', 'publishedDays', 'description', 'total_floors']
    df = df.drop(columns_to_drop, axis=1)
    df = df.replace('', np.nan).dropna()
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    #file_out_name = "clean_df.xlsx"
    #df.to_excel(file_out_name, index=False)

    return df

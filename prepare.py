import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # import needed for the train, test, split functions.
from scipy import stats

# Scalers
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#------------------------------------------------------------------------------------------------------------

# Function for Training, Validating, and Testing the data. 
def split_data(df, target= 'enter target column here'):
    ''' 
        This function is the train, validate, test, function.
        1. First we create the TRAIN and TEST dataframes at an 0.80 train_size( or test_size 0.2).

        2. Second, use the newly created TRAIN dataframe and split that at a 0.70 train_size
        ( or test_size 0.3), which means 70% of the train dataframe, so 56% of all the data.

        Now we have a train, validate, and test dataframes

    '''
    train, test = train_test_split(df, train_size=0.8, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, train_size = 0.7, random_state = 123, stratify=train[target])
    return train, validate, test

#------------------------------------------------------------------------------------------------------------

def prep_aac(intake_df, outcome_df):

    # Normalize columns
    outcome_df.columns = outcome_df.columns.str.replace(' ','_')
    intake_df.columns = intake_df.columns.str.replace(' ','_')
    outcome_df.columns = outcome_df.columns.str.lower()
    intake_df.columns = intake_df.columns.str.lower()

    # dropping subtype
    outcome_df = outcome_df.drop(['outcome_subtype'],axis=1)

    # Drop name columns
    intake_df = intake_df.drop('name',axis=1)
    outcome_df = outcome_df.drop('name',axis=1)

    return intake_df, outcome_df

#------------------------------------------------------------------------------------------------------------

def combined_df(intake_df, outcome_df):

    # Merged Data frames
    combined = intake_df.merge(outcome_df, on='animal_id')

    # Rename columns
    combined = combined.rename(columns={'datetime_x': 'date_intake', 'breed_y':'breed',
                            'datetime_y':'date_outcome','animal_type_x':'animal_type','color_y':'color'})

    # Drop columns: breed_x, animal_type_y, monthyear columns, found_location
    combined = combined.drop(['breed_x','animal_type_y','color_x', 'monthyear_x','monthyear_y', 'found_location','sex_upon_intake'], axis=1)

    return combined

#------------------------------------------------------------------------------------------------------------

def prep_combined(combined):

    # drop nulls, only 107 out of 190k
    combined = combined.dropna()

    # covert date into datetime dtype
    combined['date_intake'] = pd.to_datetime(combined.date_intake, infer_datetime_format=True)
    combined['date_outcome'] = pd.to_datetime(combined.date_outcome, infer_datetime_format=True)

    # Time in shelter
    combined['time_in_shelter'] = combined.date_outcome - combined.date_intake
    combined['time_in_shelter'] = combined.time_in_shelter.dt.days

    # Drop any negative time in shelter values. 48873 values dropped
    combined = combined[combined['time_in_shelter'] > 0]

    # Subset for only dogs
    combined_dog = combined[combined['animal_type'] == 'Dog' ]


    # Drop values that are Rto-Adopt, Died, Disposal, Missing, Stolen, Transfer
    combined_dog = combined_dog[combined_dog['outcome_type'] != 'Transfer']
    combined_dog = combined_dog[combined_dog['outcome_type'] != 'Rto-Adopt']
    combined_dog = combined_dog[combined_dog['outcome_type'] != 'Disposal']
    combined_dog = combined_dog[combined_dog['outcome_type'] != 'Missing']
    combined_dog = combined_dog[combined_dog['outcome_type'] != 'Stolen']

    # Simplify Top 10 Breeds to just main visible breed
    combined_dog['breed'] = np.where(combined_dog.breed == 'Pit Bull Mix', 'Pit Bull', combined_dog.breed)
    combined_dog['breed'] = np.where(combined_dog.breed == 'Labrador Retriever Mix', 'Labrador Retriever', combined_dog.breed)
    combined_dog['breed'] = np.where(combined_dog.breed == 'Chihuahua Shorthair Mix', 'Chihuahua', combined_dog.breed)
    combined_dog['breed'] = np.where(combined_dog.breed == 'German Shepherd Mix', 'German Shepherd', combined_dog.breed)
    combined_dog['breed'] = np.where(combined_dog.breed == 'Australian Cattle Dog Mix', 'Australian Cattle Dog', combined_dog.breed)
    combined_dog['breed'] = np.where(combined_dog.breed == 'Rat Terrier Mix', 'Rat Terrier', combined_dog.breed)
    combined_dog['breed'] = np.where(combined_dog.breed == 'Boxer Mix', 'Boxer', combined_dog.breed)
    combined_dog['breed'] = np.where(combined_dog.breed == 'Border Collie Mix', 'Border Collie', combined_dog.breed)
    combined_dog['breed'] = np.where(combined_dog.breed == 'Staffordshire Mix', 'Staffordshire', combined_dog.breed)
    combined_dog['breed'] = np.where(combined_dog.breed == 'Siberian Husky Mix', 'Siberian Husky', combined_dog.breed)
    combined_dog['breed'] = np.where(combined_dog.breed == 'Dachshund Mix', 'Dachshund', combined_dog.breed)
    combined_dog['breed'] = np.where(combined_dog.breed == 'Chihuahua Shorthair', 'Chihuahua', combined_dog.breed)

    # Top Ten Dogs
    top_ten = ['Pit Bull','Labrador Retriever','Chihuahua','German Shepherd','Australian Cattle Dog',
            'Boxer','Rat Terrier','Border Collie','Dachshund']

    # Top Ten Dogs isin
    combined_dog = combined_dog[combined_dog.breed.isin(top_ten)]

    # Top 10 Colors
    colors = ['Black/White','Black','Blue/White','Brown/White','Tan/White',
            'Tan','Brown','Black/Brown','Black/Tan','Brown Brindle/White']

    combined_dog = combined_dog[combined_dog.color.isin(colors)]

    return combined_dog

#------------------------------------------------------------------------------------------------------------

def feature_eng(combined_dog):


    
    # Duplicating time in shelter row to manipulate with np.where to make new column
    combined_dog['months_in_shelter'] = combined_dog.time_in_shelter
    combined_dog['months_in_shelter'] = np.where(combined_dog.months_in_shelter < 30, '< 30 days', '30+ days')

    # date of birth into datetime dtype
    combined_dog['date_of_birth'] = pd.to_datetime(combined_dog.date_of_birth)
    combined_dog['age_months_outcome'] = round(((combined_dog.date_outcome - combined_dog.date_of_birth).dt.days) / 30)
    combined_dog['age_years_outcome'] = round((((combined_dog.date_outcome - combined_dog.date_of_birth).dt.days) / 30) / 12)
    combined_dog['age_years_outcome'] = np.where(combined_dog.age_years_outcome <= 1, '1 year', '1+ years')

    return combined_dog

#------------------------------------------------------------------------------------------------------------



def trains(train, val, test):

    drop_these = ['animal_id', 'date_intake','intake_type', 'intake_condition',
              'animal_type','age_upon_intake','age_upon_outcome','months_in_shelter',
             'age_years_outcome','date_outcome','date_of_birth']

    train = train.drop(drop_these, axis=1)
    val = val.drop(drop_these, axis=1)
    test = test.drop(drop_these, axis=1)

    # Dummies
    train = pd.get_dummies(train, columns=['breed','color','sex_upon_outcome'])
    val = pd.get_dummies(val, columns=['breed','color','sex_upon_outcome'])
    test = pd.get_dummies(test, columns=['breed','color','sex_upon_outcome'])

    # X and y trains
    X_train = train.drop(['outcome_type'], axis=1)
    y_train = train.outcome_type

    X_val = val.drop(['outcome_type'], axis=1)
    y_val = val.outcome_type

    X_test = test.drop(['outcome_type'], axis=1)
    y_test = test.outcome_type

    return X_train, y_train, X_val, y_val, X_test, y_test

#------------------------------------------------------------------------------------------------------------

def get_scaled(X_train, X_val, X_test):


    # Scaler Object
    mm = MinMaxScaler()

    # Fit and Transform
    X_train[['time_in_shelter','age_months_outcome']] =\
    mm.fit_transform(X_train[['time_in_shelter','age_months_outcome']])

    X_val[['time_in_shelter','age_months_outcome']] =\
    mm.transform(X_val[['time_in_shelter','age_months_outcome']])

    X_test[['time_in_shelter','age_months_outcome']] =\
    mm.transform(X_test[['time_in_shelter','age_months_outcome']])

    return X_train, X_val, X_test

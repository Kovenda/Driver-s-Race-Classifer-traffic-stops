# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import pandas as pd
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Set Random Seed
np.random.seed(42)

# Read in data 
socioEcoZipCodesInfo = pd.read_csv ("socioEcoZipCodesInfo.csv")
racialProfUpdated = pd.read_csv ("racialProfUpdated.csv")

# Merge (Join) DataFrames
racialProf = racialProfUpdated.merge(socioEcoZipCodesInfo, on ='Zip_Code', how = 'outer')

# Show Missing Values 

Column_NaN_Values = {
                   "Stop_Key": racialProf["Stop_Key"].isna().sum(), 
                     "TCOLE_Sex": racialProf["TCOLE_Sex"].isna().sum(),
                    "TCOLE_RACE_ETHNICITY": racialProf["TCOLE_RACE_ETHNICITY"].isna().sum(),
    
                     "Search_Yes_or_No": racialProf["Search_Yes_or_No"].isna().sum(), 
                     "Reason_for_Stop": racialProf["Reason_for_Stop"].isna().sum(),
                    "Standardized_Race_Known": racialProf["Standardized_Race_Known"].isna().sum(),
                     
                    "TCOLE_Search_Based_On": racialProf["TCOLE_Search_Based_On"].isna().sum(), 
                     "TCOLE_Search_Found": racialProf["TCOLE_Search_Found"].isna().sum(),
                    "TCOLE_Result_of_Stop": racialProf["TCOLE_Result_of_Stop"].isna().sum(),
    
                    "TCOLE_Arrest_Based_On": racialProf["TCOLE_Arrest_Based_On"].isna().sum(), 
                    "Council_District": racialProf["Council_District"].isna().sum(),
                    "Standardized_Race": racialProf["Standardized_Race"].isna().sum(),
                    
                
                    "Stop_Time": racialProf["Stop_Time"].isna().sum(),
                    "Zip_Code": racialProf["Zip_Code"].isna().sum(),
    
    
                    "Type": racialProf["Type"].isna().sum(), 
                    "Street_Type": racialProf["Street_Type"].isna().sum(),
                   
    
                    "COUNTY": racialProf["COUNTY"].isna().sum(), 
                    "Custody": racialProf["Custody"].isna().sum(),
                    "Location": racialProf["Location"].isna().sum(),
                    
                    "Sector": racialProf["Sector"].isna().sum(), 
                    "X_COORDINATE": racialProf["X_COORDINATE"].isna().sum(),
                    "Y_COORDINATE": racialProf["Y_COORDINATE"].isna().sum(),
    
    
    
    
                         "Zip_Code": racialProf["Zip_Code"].isna().sum(), 
                     "latitude": racialProf["latitude"].isna().sum(),
                     "longitude": racialProf["longitude"].isna().sum(), 
                     "propertyTaxRate": racialProf["propertyTaxRate"].isna().sum(),
                    "numPriceChanges": racialProf["numPriceChanges"].isna().sum(), 
                     "avgSchoolRating": racialProf["avgSchoolRating"].isna().sum(),
                        "MedianStudentsPerTeacher": racialProf["MedianStudentsPerTeacher"].isna().sum(),
                            
                            
                    
     
       }
Column_NaN_Values

# Drop Missing values
racialProf = racialProf.dropna(subset =['TCOLE_RACE_ETHNICITY'])


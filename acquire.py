import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests

from pydataset import data
from scipy import stats

from env import host, username, password    # import needed for get_connection() to operate

#------------------------------------------------------------------------------------------------------------

def get_aac_intakes():

    '''
    Get data.
    '''

    filename = "aac_intakes.csv"

    return pd.read_csv(filename)

#------------------------------------------------------------------------------------------------------------




def get_aac_outcomes():

    '''
    Get data.
    '''

    filename = 'aac_outcomes.csv'

    return pd.read_csv(filename)



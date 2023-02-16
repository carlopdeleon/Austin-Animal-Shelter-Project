import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# modeling method
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

#--------------------------------------------------------------------------------------------------

def overall_outcome(df):

    '''
    Plots the overall outcome of all dogs regardless of breed, color, age, etc.
    '''

    # Outcomes
    ax = sns.countplot(x='outcome_type', data=df, ec='black')
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=35)
    plt.ylabel('Count')
    plt.xlabel('Outcome Type')
    plt.title('Outcome Type ')
    plt.show()

#--------------------------------------------------------------------------------------------------

def breed_plot(df):

    '''
    Plots the outcomes of the Top 5 Breeds.
    '''

    # Top five breeds
    top_five = ['Pit Bull','Labrador Retriever','Chihuahua','German Shepherd','Australian Cattle Dog']
    top_five_train = df[df.breed.isin(top_five)]

    # countplot
    ax = sns.countplot(x='outcome_type', data=top_five_train, hue ='breed', ec='black')
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=35)
    plt.ylabel('Count')
    plt.xlabel('Outcome')
    plt.title('Outcomes of Top 5 Breeds')
    plt.legend()
    plt.show()

#--------------------------------------------------------------------------------------------------

def color_plot(df):

    '''
    Plots the outcomes of the Top 5 dog colors.
    '''

    # Top five colors
    five_colors = ['Black/White','Brown/White','Blue/White','Black','Tan/White']
    top_five_train = df[df.color.isin(five_colors)]

    # countplot
    ax = sns.countplot(x='outcome_type', data=top_five_train, hue ='color', ec='black')
    ax.bar_label(ax.containers[4])
    plt.xticks(rotation=35)
    plt.ylabel('Count')
    plt.xlabel('Outcome')
    plt.title('Outcomes of the Top 5 Color')
    plt.show()
    


#--------------------------------------------------------------------------------------------------

def age_plot(df):

    '''
    Plots the outcomes of dogs depending on their age. Up to a year old and over a year old.
    '''

    # countplot
    ax = sns.countplot(x='outcome_type', hue='age_years_outcome', data=df, ec='black')
    for container in ax.containers:
        ax.bar_label(container)
    plt.xticks(rotation=35)
    plt.ylabel('Count')
    plt.xlabel('Outcome')
    plt.title('Outcomes based on Age')
    plt.show()

#--------------------------------------------------------------------------------------------------

def time_plot(df):

    '''
    Plots the outcomes of dogs depending on their time in shelter. Less than 30 days or more than 30 days.
    '''

    # countplot
    ax = sns.countplot(x='outcome_type', hue='months_in_shelter', data=df, ec='black')
    for container in ax.containers:
        ax.bar_label(container)
    plt.xticks(rotation=35)
    plt.ylabel('Count')
    plt.xlabel('Outcome')
    plt.title('Outcomes based on Time in Shelter')
    plt.show()

#--------------------------------------------------------------------------------------------------

def ttest1samp(df):

    '''
    1-Sample T-test
    '''
    # Compare variable
    x = df[df.outcome_type == 'Adoption']['time_in_shelter']

    t,p = stats.ttest_1samp(x, df.time_in_shelter.mean())

    print ('T-test Results')
    print('---------------')
    print(f'Test statistic: {round(t,2)}')
    print(f'P-value: {p}')

#--------------------------------------------------------------------------------------------------

def spearmanr(df, x, y):

    '''
    Spearmans R stats test. 
    '''
    corr , p = stats.spearmanr(df[x], df[y])

    print ('Spearmans R Results')
    print('--------------------')
    print(f'Correlation: {round(corr,4)}')
    print(f'P-value: {p}')

#--------------------------------------------------------------------------------------------------

def dec_tree_df(X_train, y_train, X_val, y_val):

    '''
    Decision Tree and returns dataframe of accuracy scores.
    '''

    metrics = []

    for i in range(2,31):
        
        # Model Object
        clf = DecisionTreeClassifier(max_depth= 10, max_leaf_nodes= i, random_state=123)
        
        # Fit Object
        clf.fit(X_train, y_train)
        
        train_clf = clf.score(X_train,y_train)
        val_clf = clf.score(X_val, y_val)
        
        output = {'max_leaf_nodes': i,
                'train_accuracy': train_clf,
                'validate_accuracy': val_clf}
        
        metrics.append(output)

    dt_df = pd.DataFrame(metrics)
    dt_df['difference'] = dt_df.train_accuracy - dt_df.validate_accuracy
    
    return dt_df
        

#--------------------------------------------------------------------------------------------------

def dec_tree(X_train, y_train, X_val, y_val):

    '''
    Decision Tree and returns dataframe of accuracy scores.
    '''

    metrics = []

    # Model Object
    clf = DecisionTreeClassifier(max_depth= 10, max_leaf_nodes= 16, random_state=123)
        
    # Fit Object
    clf.fit(X_train, y_train)
    
    # Accuracy Scores
    train_clf = clf.score(X_train,y_train)
    val_clf = clf.score(X_val, y_val)

    # Dictionary for dataframe   
    output = {'model_type': 'Decision Tree',
                'train_accuracy': train_clf,
                'validate_accuracy': val_clf}
        
    metrics.append(output)

    dt_df = pd.DataFrame(metrics)
    dt_df['difference'] = dt_df.train_accuracy - dt_df.validate_accuracy

    print('Results')
    print('---------')
    print(f'Train Accuracy: {round(train_clf,4)}')
    print(f'Validate Accuracy: {round(val_clf,4)}')
    print(f'Difference: {round(train_clf - val_clf,4)}')
    
    return dt_df


#--------------------------------------------------------------------------------------------------

def rand_forest(X_train, y_train, X_val, y_val):

    '''
    Random Forest and returns dataframe of accuracy scores.
    '''

    metrics_rf = []

    # Model Object
    rf = RandomForestClassifier(n_estimators= 200, max_depth= 5, random_state=123)
    
    # Fit Object
    rf.fit(X_train, y_train)
    
    # Accuracy Scores
    train_rf = rf.score(X_train,y_train)
    val_rf = rf.score(X_val, y_val)
    
    # Dictionary for dataframe
    output = {'model_type': 'Random Forest',
            'train_accuracy': train_rf,
            'validate_accuracy': val_rf}
    
    metrics_rf.append(output)

    # Create dataframe
    rf_df = pd.DataFrame(metrics_rf)
    rf_df['difference'] = rf_df.train_accuracy - rf_df.validate_accuracy
    
    print('Results')
    print('---------')
    print(f'Train Accuracy: {round(train_rf,4)}')
    print(f'Validate Accuracy: {round(val_rf,4)}')
    print(f'Difference: {round(train_rf - val_rf,4)}')

    return rf_df


#--------------------------------------------------------------------------------------------------

def log_reg(X_train, y_train, X_val, y_val):

    '''
    Logistic Regression and returns dataframe of accuracy scores.
    '''

    metrics = []

    lr = LogisticRegression(random_state=123)
    lr.fit(X_train, y_train)

    lr_acc_train = lr.score(X_train, y_train)
    lr_acc_val = lr.score(X_val, y_val)

    # Dictionary for dataframe
    output = {'model_type': 'Logistic Regression',
            'train_accuracy': lr_acc_train,
            'validate_accuracy': lr_acc_val}

    metrics.append(output)

    log_reg = pd.DataFrame(metrics)
    log_reg['difference'] = log_reg.train_accuracy - log_reg.validate_accuracy

    print('Results')
    print('---------')
    print(f'Train Accuracy: {round(lr_acc_train,4)}')
    print(f'Validate Accuracy: {round(lr_acc_val,4)}')
    print(f'Difference: {round(lr_acc_train - lr_acc_val,4)}')

    return log_reg
#--------------------------------------------------------------------------------------------------

def knn(X_train, y_train, X_val, y_val):

    '''
    KNN model and returns data frame of accuracy scores.
    '''

    metrics = []

    knn = KNeighborsClassifier(n_neighbors= 16)
    knn.fit(X_train, y_train)
    
     # Accuracy Scores
    train_knn = knn.score(X_train,y_train)
    val_knn = knn.score(X_val, y_val)

    # Dictionary for dataframe
    output = {'model_type': 'KNN',
            'train_accuracy': train_knn,
            'validate_accuracy': val_knn}

    metrics.append(output)

    knn_df = pd.DataFrame(metrics)
    knn_df['difference'] = knn_df.train_accuracy - knn_df.validate_accuracy

    print('Results')
    print('---------')
    print(f'Train Accuracy: {round(train_knn,4)}')
    print(f'Validate Accuracy: {round(val_knn,4)}')
    print(f'Difference: {round(train_knn - val_knn,4)}')

    return knn_df

#--------------------------------------------------------------------------------------------------

def dec_tree1(X_train, y_train, X_val, y_val):

    '''
    Decision Tree model and returns data frame of accuracy scores. Does not print results.
    '''

    metrics = []

    # Model Object
    clf = DecisionTreeClassifier(max_depth= 10, max_leaf_nodes= 16, random_state=123)
        
    # Fit Object
    clf.fit(X_train, y_train)
    
    # Accuracy Scores
    train_clf = clf.score(X_train,y_train)
    val_clf = clf.score(X_val, y_val)

    # Dictionary for dataframe   
    output = {'model_type': 'Decision Tree',
                'train_accuracy': train_clf,
                'validate_accuracy': val_clf}
        
    metrics.append(output)

    dt_df = pd.DataFrame(metrics)
    dt_df['difference'] = dt_df.train_accuracy - dt_df.validate_accuracy
    
    return dt_df


#--------------------------------------------------------------------------------------------------

def rand_forest1(X_train, y_train, X_val, y_val):

    '''
    Random Forest model and returns data frame of accuracy scores. Does not print results.
    '''

    metrics_rf = []

    # Model Object
    rf = RandomForestClassifier(n_estimators= 200, max_depth= 5, random_state=123)
    
    # Fit Object
    rf.fit(X_train, y_train)
    
    # Accuracy Scores
    train_rf = rf.score(X_train,y_train)
    val_rf = rf.score(X_val, y_val)
    
    # Dictionary for dataframe
    output = {'model_type': 'Random Forest',
            'train_accuracy': train_rf,
            'validate_accuracy': val_rf}
    
    metrics_rf.append(output)

    # Create dataframe
    rf_df = pd.DataFrame(metrics_rf)
    rf_df['difference'] = rf_df.train_accuracy - rf_df.validate_accuracy

    return rf_df


#--------------------------------------------------------------------------------------------------

def log_reg1(X_train, y_train, X_val, y_val):

    '''
    Logistic Regression model and returns data frame of accuracy scores. Does not print results.
    '''

    metrics = []

    lr = LogisticRegression(random_state=123)
    lr.fit(X_train, y_train)

    lr_acc_train = lr.score(X_train, y_train)
    lr_acc_val = lr.score(X_val, y_val)

    # Dictionary for dataframe
    output = {'model_type': 'Logistic Regression',
            'train_accuracy': lr_acc_train,
            'validate_accuracy': lr_acc_val}

    metrics.append(output)

    log_reg = pd.DataFrame(metrics)
    log_reg['difference'] = log_reg.train_accuracy - log_reg.validate_accuracy

    return log_reg

#--------------------------------------------------------------------------------------------------

def knn1(X_train, y_train, X_val, y_val):

    '''
    KNN model and returns data frame of accuracy scores. Does not print results.
    '''

    metrics = []

    knn = KNeighborsClassifier(n_neighbors= 16)
    knn.fit(X_train, y_train)
    
     # Accuracy Scores
    train_knn = knn.score(X_train,y_train)
    val_knn = knn.score(X_val, y_val)

    # Dictionary for dataframe
    output = {'model_type': 'KNN',
            'train_accuracy': train_knn,
            'validate_accuracy': val_knn}

    metrics.append(output)

    knn_df = pd.DataFrame(metrics)
    knn_df['difference'] = knn_df.train_accuracy - knn_df.validate_accuracy

    return knn_df

#--------------------------------------------------------------------------------------------------

def all_in_one(X_train, y_train, X_val, y_val):

    '''
    Using other functions to creates models and create a datafram of all their accuarcy scores.
    '''

    dt = dec_tree1(X_train, y_train, X_val, y_val)
    rf = rand_forest1(X_train, y_train, X_val, y_val)
    lr = log_reg1(X_train, y_train, X_val, y_val)
    kn = knn1(X_train, y_train, X_val, y_val)

    baseline = 0.67
    

    df = pd.concat([dt,rf,lr,kn], axis=0, ignore_index=True)
    df['baseline'] = baseline
    

    return df

#--------------------------------------------------------------------------------------------------

def test_model(X_train, y_train, X_val, y_val, X_test, y_test):

    '''
    Random Forest using test data set. Returns dataframe of validate and test scores.
    '''

    metrics = []

    # Model Object
    rf = RandomForestClassifier(n_estimators= 200, max_depth= 6, random_state=123)
        
    # Fit Object
    rf.fit(X_train, y_train)
        
    # Accuracy Scores
    val_rf = rf.score(X_val, y_val)
    test_rf = rf.score(X_test, y_test)

    output = {'Validate': round(val_rf,4),
                'Test': round(test_rf,4) }

    metrics.append(output)

    df = pd.DataFrame(metrics)
    df['difference'] = df.Validate - df.Test
    df['baseline'] = 0.67

    return df

#--------------------------------------------------------------------------------------------------


def chi_test(train, x, y ):
    
    # for crosstab
    br = train[x]
    ad = train[y]

    #crosstab
    ct = pd.crosstab(br,ad)

    # chi square test
    chi2, p, degf, expected = stats.chi2_contingency(ct)

    print ('Chi-Square Results')
    print('--------------------')
    print(f'P-value: {p}')

#--------------------------------------------------------------------------------------------------

def ttest1samp1(df):

    '''
    1-Sample T-test
    '''
    # Compare variable
    x = df[df['outcome_type'] == 'Adoption']['age_months_outcome']
    y = df.age_months_outcome.mean()

    t,p = stats.ttest_1samp(x, y )

    print ('T-test Results')
    print('---------------')
    print(f'Test statistic: {round(t,2)}')
    print(f'P-value: {p}')
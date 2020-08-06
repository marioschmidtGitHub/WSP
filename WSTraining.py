# TRAINING
#2. Random Forest, K Nearest Neighbors and Multiple Lineal Model Regression predictors are trained with the following Historic Dataframe data.
# Spike trend, Column D: it is 1 if the spike trend is UP, codes 34, 35, 40 and 43, and it is 0 if the spike trend is DOWN, as it happens in codes 30, 33, 44 and 45. This value is always knew.
# Spike code, Column E:  30, 33, 34, 35, 40, 43, 44 and 45. Note that column AB for spike codes 30, 33, 34 and 35 is 1, and 0 for spikes codes 40, 43, 44 and 45.
# Cycle, Column AB: it is 1, if the spike belong to one UP cycle, and 0 if it belongs to one DOWN cycle of the SPY.
# Columns F to AA are the original 22 Original value for NYSE indicators, 2 of the original 24OV are discard.
# The predictor is trained using the 22OV and AB column. Thus after the predictor be trained, the spike to be analysed will has a greater probability to belong to one UP cycle as its value be closest to 1, and it will has a greater probability to belong to one DWON cycle as its value be closest to 0.
import sklearn
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import os
from IPython.display import clear_output
from threading import Timer
from sklearn import linear_model
formato_hora = "%X"
clear_output(wait=True)
ahora = time.strftime(formato_hora)
def TRAINING (level): 
    print("Model is training. Please wait...")
    #23. Random Forest UP/DOWN cycle
    # The predictor is trained using the 22OV and AB column. 
    # Thus after the predictor be trained, the spike to be analysed will has a greater probability to belong to one UP cycle as the prediction value be closest to 1, and it will has a greater probability to belong to one DWON cycle as the prediction value be closest to 0.
    #26. Define Dataset for UP/DOWN First level training
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        #33 Define Dataset for UP/DOWN Random Forest Second level Training, adding the prediction of first level.
        X=dataset[['D'] + ['FRSTUP'] + ['FRSTDW'] + ['FRST30'] + ['FRST45'] + ['FRST33'] + ['FRST44'] + ['FRST34'] + ['FRST43'] + ['FRST35'] + ['FRST40']]
    else:
        print ("TRAINING FRST10 level error")
    #X = dataset.loc[:,'D':'AA']
    X = np.array(X)
    # X = np.reshape(X, X.shape[0])
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    from sklearn.ensemble import RandomForestClassifier
    if level=="first":
        firclfFRST10 = RandomForestClassifier(n_estimators = 15, criterion = 'entropy')
        #48. Train and save UP/DOWN Random Forest Predictor, first level
        firclfFRST10.fit(X_train, y_train)
        joblib.dump(firclfFRST10, 'FIRWSFRST10_entrenado.pkl')
        #firclfFRST10 = joblib.load('FIRWSFRST10_entrenado.pkl')
        print ("firclfFRST10 OK ")
    elif level == "second":
        secclfFRST10 = RandomForestClassifier(n_estimators = 15, criterion = 'entropy')
        #55. Train and save UP/DOWN Random Forest Predictor, second level
        secclfFRST10.fit(X_train, y_train)
        joblib.dump(secclfFRST10, 'SECWSFRST10_entrenado.pkl')
        #secclfFRST10 = joblib.load('SECWSFRST10_entrenado.pkl')
        print ("secclfFRST10 OK ")
    else:
        print ("TRAINING FRST10 level error")  
        
* #63. Random Forest for codes 30 & 45
* Only rowns/Spikes with code 30 and 45 are used to train this Predictor
* Spike codes 30 & 45 (column E) have both DOWN trend (column D:0), but spike code 30 belong to one UP cycle (Column AB:1) and code 45 spike belong to one DOWN cycle (Column AB:0).
* Thus after the predictor be trained, the spike to be analysed will has a greater probability to be one UP spike code 30 (which belong to one UP cycle) as the prediction value be closest to 1, and it will has a greater probability to be one DOWN spike code 45 (which belong to one DOWN cycle) as the prediction value be closest to 0.

    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==33)].index)
    dataset = dataset.drop(dataset[(dataset['E']==34)].index)
    dataset = dataset.drop(dataset[(dataset['E']==35)].index)
    dataset = dataset.drop(dataset[(dataset['E']==40)].index)
    dataset = dataset.drop(dataset[(dataset['E']==43)].index)
    dataset = dataset.drop(dataset[(dataset['E']==44)].index)
    #75. Define Dataset for spike codes 30 & 45 First level training
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        #80 Define Dataset for Random Forest for spikes codes 30 $ 45 Second level Training, adding the prediction of first level.
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+ ['FRSTUP'] + ['FRSTDW'] + ['FRST30'] + ['FRST45'] + ['FRST33'] + ['FRST44'] + ['FRST34'] + ['FRST43'] + ['FRST35'] + ['FRST40']]
    else:
        print ("TRAINING FRST3045 level error")
    X = np.array(X)
    # X = np.reshape(X, X.shape[0])
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    from sklearn.ensemble import RandomForestClassifier
    if level=="first":
        #94. Train and save 3045 Random Forest Predictor, first level
        firclfFRST3045 = RandomForestClassifier(n_estimators = 15, criterion = 'entropy')
        firclfFRST3045.fit(X_train, y_train)
        joblib.dump(firclfFRST3045, 'FIRWSFRST3045_entrenado.pkl')
        #firclfFRST3045 = joblib.load('FIRWSFRST3045_entrenado.pkl')
        print ("firclfFRST3045 OK ")
    elif level == "second":
        #101. Train and save 3045 Random Forest Predictor, second level
        secclfFRST3045 = RandomForestClassifier(n_estimators = 15, criterion = 'entropy')
        secclfFRST3045.fit(X_train, y_train)
        joblib.dump(secclfFRST3045, 'SECWSFRST3045_entrenado.pkl')
        #secclfFRST3045 = joblib.load('SECWSFRST3045_entrenado.pkl')
        print ("secclfFRST3045 OK ")
    else:
        print ("TRAINING FRST3045 level error")
    * #109. Random Forest for codes 33 & 44
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==30)].index)
    dataset = dataset.drop(dataset[(dataset['E']==34)].index)
    dataset = dataset.drop(dataset[(dataset['E']==35)].index)
    dataset = dataset.drop(dataset[(dataset['E']==40)].index)
    dataset = dataset.drop(dataset[(dataset['E']==43)].index)
    dataset = dataset.drop(dataset[(dataset['E']==45)].index)
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+ ['FRSTUP'] + ['FRSTDW'] + ['FRST30'] + ['FRST45'] + ['FRST33'] + ['FRST44'] + ['FRST34'] + ['FRST43'] + ['FRST35'] + ['FRST40']]
    else:
        print ("TRAINING FRST3344 level error")
    X = np.array(X)
    # X = np.reshape(X, X.shape[0])
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    from sklearn.ensemble import RandomForestClassifier
    if level=="first":
        firclfFRST3344 = RandomForestClassifier(n_estimators = 15, criterion = 'entropy')
        firclfFRST3344.fit(X_train, y_train)
        joblib.dump(firclfFRST3344, 'FIRWSFRST3344_entrenado.pkl')
        #firclfFRST3344 = joblib.load('FIRWSFRST3344_entrenado.pkl')
        print ("firclfFRST3344 OK ")
    elif level == "second":
        secclfFRST3344 = RandomForestClassifier(n_estimators = 15, criterion = 'entropy')
        secclfFRST3344.fit(X_train, y_train)
        joblib.dump(secclfFRST3344, 'SECWSFRST3344_entrenado.pkl')
        #secclfFRST3344 = joblib.load('SECWSFRST3344_entrenado.pkl')
        print ("secclfFRST3344 OK ")
    else:
        print ("TRAINING FRST3344 level error")               
    #147. Random Forest for codes 34 & 43
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==30)].index)
    dataset = dataset.drop(dataset[(dataset['E']==33)].index)
    dataset = dataset.drop(dataset[(dataset['E']==35)].index)
    dataset = dataset.drop(dataset[(dataset['E']==40)].index)
    dataset = dataset.drop(dataset[(dataset['E']==44)].index)
    dataset = dataset.drop(dataset[(dataset['E']==45)].index)
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)        
        X=dataset[['D']+ ['FRSTUP'] + ['FRSTDW'] + ['FRST30'] + ['FRST45'] + ['FRST33'] + ['FRST44'] + ['FRST34'] + ['FRST43'] + ['FRST35'] + ['FRST40']]
    else:
        print ("TRAINING FRST3443 level error")
    X = np.array(X)
    # X = np.reshape(X, X.shape[0])
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    from sklearn.ensemble import RandomForestClassifier
    if level=="first":
        firclfFRST3443 = RandomForestClassifier(n_estimators = 15, criterion = 'entropy')
        firclfFRST3443.fit(X_train, y_train) 
        joblib.dump(firclfFRST3443, 'FIRWSFRST3443_entrenado.pkl')
        #firclfFRST3443 = joblib.load('FIRWSFRST3443_entrenado.pkl')
        print ("firclfFRST3443 OK ")
    elif level == "second":
        secclfFRST3443 = RandomForestClassifier(n_estimators = 15, criterion = 'entropy')
        secclfFRST3443.fit(X_train, y_train)
        joblib.dump(secclfFRST3443, 'SECWSFRST3443_entrenado.pkl')
        #secclfFRST3443 = joblib.load('SECWSFRST3443_entrenado.pkl')
        print ("secclfFRST3443 OK ")
    else:
        print ("TRAINING FRST3443 level error")
    #185. Random Forest for codes 35 & 40
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==30)].index)
    dataset = dataset.drop(dataset[(dataset['E']==33)].index)
    dataset = dataset.drop(dataset[(dataset['E']==34)].index)
    dataset = dataset.drop(dataset[(dataset['E']==43)].index)
    dataset = dataset.drop(dataset[(dataset['E']==44)].index)
    dataset = dataset.drop(dataset[(dataset['E']==45)].index)
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+ ['FRSTUP'] + ['FRSTDW'] + ['FRST30'] + ['FRST45'] + ['FRST33'] + ['FRST44'] + ['FRST34'] + ['FRST43'] + ['FRST35'] + ['FRST40']]
    else:
        print ("TRAINING FRST3540 level error")
    X = np.array(X)
    # X = np.reshape(X, X.shape[0])
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    from sklearn.ensemble import RandomForestClassifier
    if level=="first":
        firclfFRST3540 = RandomForestClassifier(n_estimators = 15, criterion = 'entropy')
        firclfFRST3540.fit(X_train, y_train)
        joblib.dump(firclfFRST3540, 'FIRWSFRST3540_entrenado.pkl')
        #firclfFRST3540 = joblib.load('FIRWSFRST3540_entrenado.pkl')
        print ("firclfFRST3540 OK ")
    elif level == "second":
        secclfFRST3540 = RandomForestClassifier(n_estimators = 15, criterion = 'entropy')
        secclfFRST3540.fit(X_train, y_train)
        joblib.dump(secclfFRST3540, 'SECWSFRST3540_entrenado.pkl')
        #secclfFRST3540 = joblib.load('SECWSFRST3540_entrenado.pkl')
        print ("secclfFRST3540 OK ")
    else:
        print ("TRAINING FRST3540 level error")

    # TRAINING KNEAREST
    #225. K Nearest Neighbors UP/DOWN cycle
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    dataset.shape
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+ ['KNSTUP'] + ['KNSTDW'] + ['KNST30'] + ['KNST45'] + ['KNST33'] + ['KNST44'] + ['KNST34'] + ['KNST43'] + ['KNST35'] + ['KNST40']]
    else:
        print ("TRAINING KNST10 level error")
    X = np.array(X)
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    from sklearn.neighbors import KNeighborsClassifier
    if level=="first":
        firclfKNST10 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
        firclfKNST10.fit(X_train, y_train)
        joblib.dump(firclfKNST10, 'FIRWSKNST10_entrenado.pkl')
        #firclfKNST10 = joblib.load('FIRWSKNST10_entrenado.pkl')
        print ("firclfKNST10 OK ")
    elif level == "second":
        secclfKNST10 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
        secclfKNST10.fit(X_train, y_train)
        joblib.dump(secclfKNST10, 'SECWSKNST10_entrenado.pkl')
        #secclfKNST10 = joblib.load('SECWSKNST10_entrenado.pkl')
        print ("secclfKNST10 OK ")
    else:
        print ("TRAINING KNST10 level error")

    #258. K Nearest Neighbors for codes 30 & 45
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==33)].index)
    dataset = dataset.drop(dataset[(dataset['E']==34)].index)
    dataset = dataset.drop(dataset[(dataset['E']==35)].index)
    dataset = dataset.drop(dataset[(dataset['E']==40)].index)
    dataset = dataset.drop(dataset[(dataset['E']==43)].index)
    dataset = dataset.drop(dataset[(dataset['E']==44)].index)
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    dataset.shape
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+['KNSTUP'] + ['KNSTDW'] + ['KNST30'] + ['KNST45'] + ['KNST33'] + ['KNST44'] + ['KNST34'] + ['KNST43'] + ['KNST35'] + ['KNST40']]
    else:
        print ("TRAINING KNST3045 level error")
    X = np.array(X)
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    from sklearn.neighbors import KNeighborsClassifier
    if level=="first":
        firclfKNST3045 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
        firclfKNST3045.fit(X_train, y_train)
        joblib.dump(firclfKNST3045, 'FIRWSKNST3045_entrenado.pkl')
        #firclfKNST3045 = joblib.load('FIRWSKNST3045_entrenado.pkl')
        print ("firclfKNST3045 OK ")
    elif level == "second":
        secclfKNST3045 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
        secclfKNST3045.fit(X_train, y_train)
        joblib.dump(secclfKNST3045, 'SECWSKNST3045_entrenado.pkl')
        #secclfKNST3045 = joblib.load('SECWSKNST3045_entrenado.pkl')
        print ("secclfKNST3045 OK ")
    else:
        print ("TRAINING KNST3045 level error")
    #296. K Nearest Neighbors for codes 33 & 44
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==30)].index)
    dataset = dataset.drop(dataset[(dataset['E']==34)].index)
    dataset = dataset.drop(dataset[(dataset['E']==35)].index)
    dataset = dataset.drop(dataset[(dataset['E']==40)].index)
    dataset = dataset.drop(dataset[(dataset['E']==43)].index)
    dataset = dataset.drop(dataset[(dataset['E']==45)].index)
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    dataset.shape
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+['KNSTUP'] + ['KNSTDW'] + ['KNST30'] + ['KNST45'] + ['KNST33'] + ['KNST44'] + ['KNST34'] + ['KNST43'] + ['KNST35'] + ['KNST40']]
    else:
        print ("TRAINING KNST3344 level error")
    X = np.array(X)
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    from sklearn.neighbors import KNeighborsClassifier
    if level=="first":
        firclfKNST3344 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
        firclfKNST3344.fit(X_train, y_train)
        joblib.dump(firclfKNST3344, 'FIRWSKNST3344_entrenado.pkl')
        #firclfKNST3344 = joblib.load('FIRWSKNST3344_entrenado.pkl')
        print ("firclfKNST3344 OK ")        
    elif level == "second":
        secclfKNST3344 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
        secclfKNST3344.fit(X_train, y_train)
        joblib.dump(secclfKNST3344, 'SECWSKNST3344_entrenado.pkl')
        #secclfKNST3344 = joblib.load('SECWSKNST3344_entrenado.pkl')
        print ("secclfKNST3344 OK ")
    else:
        print ("TRAINING KNST3344 level error")

    #335. K Nearest Neighbors for codes 34 & 43
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==30)].index)
    dataset = dataset.drop(dataset[(dataset['E']==33)].index)
    dataset = dataset.drop(dataset[(dataset['E']==35)].index)
    dataset = dataset.drop(dataset[(dataset['E']==40)].index)
    dataset = dataset.drop(dataset[(dataset['E']==44)].index)
    dataset = dataset.drop(dataset[(dataset['E']==45)].index)
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    dataset.shape
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+ ['KNSTUP'] + ['KNSTDW'] + ['KNST30'] + ['KNST45'] + ['KNST33'] + ['KNST44'] + ['KNST34'] + ['KNST43'] + ['KNST35'] + ['KNST40']]
    else:
        print ("TRAINING KNST3443 level error")
    X = np.array(X)
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    from sklearn.neighbors import KNeighborsClassifier
    if level=="first":
        firclfKNST3443 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
        firclfKNST3443.fit(X_train, y_train)
        joblib.dump(firclfKNST3443, 'FIRWSKNST3443_entrenado.pkl')
        #firclfKNST3443 = joblib.load('FIRWSKNST3443_entrenado.pkl')
        print ("firclfKNST3443 OK ")
    elif level == "second":
        secclfKNST3443 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
        secclfKNST3443.fit(X_train, y_train)
        joblib.dump(secclfKNST3443, 'SECWSKNST3443_entrenado.pkl')
        #secclfKNST3443 = joblib.load('SECWSKNST3443_entrenado.pkl')
        print ("secclfKNST3443 OK ")
    else:
        print ("TRAINING KNST3443 level error")

    #374. K Nearest Neighbors for codes 35 & 40
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==30)].index)
    dataset = dataset.drop(dataset[(dataset['E']==33)].index)
    dataset = dataset.drop(dataset[(dataset['E']==34)].index)
    dataset = dataset.drop(dataset[(dataset['E']==43)].index)
    dataset = dataset.drop(dataset[(dataset['E']==44)].index)
    dataset = dataset.drop(dataset[(dataset['E']==45)].index)
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    dataset.shape
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+ ['KNSTUP'] + ['KNSTDW'] + ['KNST30'] + ['KNST45'] + ['KNST33'] + ['KNST44'] + ['KNST34'] + ['KNST43'] + ['KNST35'] + ['KNST40']]
    else:
        print ("TRAINING KNST3540 level error")
    X = np.array(X)
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    from sklearn.neighbors import KNeighborsClassifier
    if level=="first":
        firclfKNST3540 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
        firclfKNST3540.fit(X_train, y_train)
        joblib.dump(firclfKNST3540, 'FIRWSKNST3540_entrenado.pkl')
        #firclfKNST3540 = joblib.load('FIRWSKNST3540_entrenado.pkl')
        print ("firclfKNST3540 OK ")
    elif level == "second":
        secclfKNST3540 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
        secclfKNST3540.fit(X_train, y_train)
        joblib.dump(secclfKNST3540, 'SECWSKNST3540_entrenado.pkl')
        #ecclfKNST3540 = joblib.load('SECWSKNST3540_entrenado.pkl')
        print ("secclfKNST3540 OK ")
    else:
        print ("TRAINING FRST10 level error")

    #397 TRAINING MLRG SUPPORT VECTOR MACHINE CLASSIFICATION
    #414. Multiple Linear Regression UP/DOWN cycle
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    dataset.shape
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+ ['MLRGUP'] + ['MLRGDW'] + ['MLRG30'] + ['MLRG45'] + ['MLRG33'] + ['MLRG44'] + ['MLRG34'] + ['MLRG43'] + ['MLRG35'] + ['MLRG40']]
    else:
        print ("TRAINING MLRG10 level error")
    X = np.array(X)
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    if level=="first":
        firclfMLRG10 = linear_model.LinearRegression()
        #Entreno el modelo
        firclfMLRG10.fit(X, y)
        joblib.dump(firclfMLRG10, 'FIRWSMLRG10_entrenado.pkl')
        print ("firclfMLRG10 OK ")
    elif level == "second":
        secclfMLRG10 = linear_model.LinearRegression()
        #Entreno el modelo
        secclfMLRG10.fit(X, y)
        joblib.dump(secclfMLRG10, 'SECWSMLRG10_entrenado.pkl')
        #secclfMLRG10 = joblib.load('SECWSMLRG10_entrenado.pkl')
        print ("secclfMLRG10 OK ")
    else:
        print ("TRAINING MLRG10 level error")

    #447. Multiple Linear Regression for codes 30 & 45
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==33)].index)
    dataset = dataset.drop(dataset[(dataset['E']==34)].index)
    dataset = dataset.drop(dataset[(dataset['E']==35)].index)
    dataset = dataset.drop(dataset[(dataset['E']==40)].index)
    dataset = dataset.drop(dataset[(dataset['E']==43)].index)
    dataset = dataset.drop(dataset[(dataset['E']==44)].index)
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    dataset.shape
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+ ['MLRGUP'] + ['MLRGDW'] + ['MLRG30'] + ['MLRG45'] + ['MLRG33'] + ['MLRG44'] + ['MLRG34'] + ['MLRG43'] + ['MLRG35'] + ['MLRG40']]
    else:
        print ("TRAINING MLRG3045 level error")
    X = np.array(X)
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    if level=="first":
        firclfMLRG3045 = linear_model.LinearRegression()
        firclfMLRG3045.fit(X_train, y_train)
        joblib.dump(firclfMLRG3045, 'FIRWSMLRG3045_entrenado.pkl')
        #firclfMLRG3045 = joblib.load('FIRWSMLRG3045_entrenado.pkl')
        print ("firclfMLRG3045 OK ")    
    elif level == "second":
        secclfMLRG3045 = linear_model.LinearRegression()
        secclfMLRG3045.fit(X_train, y_train)
        joblib.dump(secclfMLRG3045, 'SECWSMLRG3045_entrenado.pkl')
        #secclfMLRG3045 = joblib.load('SECWSMLRG3045_entrenado.pkl')
        print ("secclfMLRG3045 OK ")
    else:
        print ("TRAINING MLRG3045 level error")

    #485. Multiple Linear Regression for codes 33 & 44
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==30)].index)
    dataset = dataset.drop(dataset[(dataset['E']==34)].index)
    dataset = dataset.drop(dataset[(dataset['E']==35)].index)
    dataset = dataset.drop(dataset[(dataset['E']==40)].index)
    dataset = dataset.drop(dataset[(dataset['E']==43)].index)
    dataset = dataset.drop(dataset[(dataset['E']==45)].index)
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    dataset.shape
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+ ['MLRGUP'] + ['MLRGDW'] + ['MLRG30'] + ['MLRG45'] + ['MLRG33'] + ['MLRG44'] + ['MLRG34'] + ['MLRG43'] + ['MLRG35'] + ['MLRG40']]
    else:
        print ("TRAINING MLRG3344 level error")
    X = np.array(X)
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    if level=="first":
        firclfMLRG3344 = linear_model.LinearRegression()
        firclfMLRG3344.fit(X_train, y_train)
        joblib.dump(firclfMLRG3344, 'FIRWSMLRG3344_entrenado.pkl')
        #firclfMLRG3344 = joblib.load('FIRWSMLRG3344_entrenado.pkl')
        print ("firclfMLRG3344 OK ")
    elif level == "second":
        secclfMLRG3344 = linear_model.LinearRegression()
        secclfMLRG3344.fit(X_train, y_train)
        joblib.dump(secclfMLRG3344, 'SECWSMLRG3344_entrenado.pkl')
        #secclfMLRG3344 = joblib.load('SECWSMLRG3344_entrenado.pkl')
        print ("secclfMLRG3344 OK ")
    else:
        print ("TRAINING MLRG3344 level error")

    #523. Multiple Linear Regression for codes 34 & 43
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==30)].index)
    dataset = dataset.drop(dataset[(dataset['E']==33)].index)
    dataset = dataset.drop(dataset[(dataset['E']==35)].index)
    dataset = dataset.drop(dataset[(dataset['E']==40)].index)
    dataset = dataset.drop(dataset[(dataset['E']==44)].index)
    dataset = dataset.drop(dataset[(dataset['E']==45)].index)
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    dataset.shape
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+ ['MLRGUP'] + ['MLRGDW'] + ['MLRG30'] + ['MLRG45'] + ['MLRG33'] + ['MLRG44'] + ['MLRG34'] + ['MLRG43'] + ['MLRG35'] + ['MLRG40']]
    else:
        print ("TRAINING MLRG3443 level error")
    X = np.array(X)
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    if level=="first":
        firclfMLRG3443 = linear_model.LinearRegression()
        firclfMLRG3443.fit(X_train, y_train)
        joblib.dump(firclfMLRG3443, 'FIRWSMLRG3443_entrenado.pkl')
        #firclfMLRG3443 = joblib.load('FIRWSMLRG3443_entrenado.pkl')
        print ("firclfMLRG3443 OK ")
    elif level == "second":
        secclfMLRG3443 = linear_model.LinearRegression()
        secclfMLRG3443.fit(X_train, y_train)
        joblib.dump(secclfMLRG3443, 'SECWSMLRG3443_entrenado.pkl')
        #secclfMLRG3443 = joblib.load('SECWSMLRG3443_entrenado.pkl')
        print ("secclfMLRG3443 OK ")
    else:
        print ("TRAINING FRST10 level error")

    #561. Multiple Linear Regression for codes 35 & 40
    dataset = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
    dataset = dataset.drop(dataset[(dataset['E']==30)].index)
    dataset = dataset.drop(dataset[(dataset['E']==33)].index)
    dataset = dataset.drop(dataset[(dataset['E']==34)].index)
    dataset = dataset.drop(dataset[(dataset['E']==43)].index)
    dataset = dataset.drop(dataset[(dataset['E']==44)].index)
    dataset = dataset.drop(dataset[(dataset['E']==45)].index)
    dataset.drop(['A', 'Date', 'C', 'E', 'F'], axis=1, inplace=True)
    dataset.shape
    if level=="first":
        X= dataset.loc[:,'D':'AA']
    elif level == "second":
        dataset.fillna(0, inplace=True)
        X=dataset[['D']+ ['MLRGUP'] + ['MLRGDW'] + ['MLRG30'] + ['MLRG45'] + ['MLRG33'] + ['MLRG44'] + ['MLRG34'] + ['MLRG43'] + ['MLRG35'] + ['MLRG40']]
    else:
        print ("TRAINING MLRG3540 level error")
    X = np.array(X)
    y = dataset.loc[:,'AB']
    y = np.array(y)
    y = np.reshape(y, y.shape[0])
    X_train = X
    y_train = y
    if level=="first":
        firclfMLRG3540 = linear_model.LinearRegression()
        firclfMLRG3540.fit(X_train, y_train)
        joblib.dump(firclfMLRG3540, 'FIRWSMLRG3540_entrenado.pkl')
        #firclfMLRG3540 = joblib.load('FIRWSMLRG3540_entrenado.pkl')
        print ("firclfMLRG3540 OK ")
    elif level == "second":
        secclfMLRG3540 = linear_model.LinearRegression()
        secclfMLRG3540.fit(X_train, y_train)
        joblib.dump(secclfMLRG3540, 'SECWSMLRG3540_entrenado.pkl')
        #secclfMLRG3540 = joblib.load('SECWSMLRG3540_entrenado.pkl')
        print ("secclfMLRG3540 OK ")
    else:
        print ("TRAINING MLRG3540 level error")
TRAINING("first")
TRAINING("second")
#first
clear_output(wait=True)
firclfFRST10 = joblib.load('FIRWSFRST10_entrenado.pkl')
firclfKNST10 = joblib.load('FIRWSKNST10_entrenado.pkl')
firclfMLRG10 = joblib.load('FIRWSMLRG10_entrenado.pkl')
firclfFRST3045 = joblib.load('FIRWSFRST3045_entrenado.pkl')
firclfKNST3045 = joblib.load('FIRWSKNST3045_entrenado.pkl')
firclfMLRG3045 = joblib.load('FIRWSMLRG3045_entrenado.pkl')
firclfFRST3344 = joblib.load('FIRWSFRST3344_entrenado.pkl')
firclfKNST3344 = joblib.load('FIRWSKNST3344_entrenado.pkl')
firclfMLRG3344 = joblib.load('FIRWSMLRG3344_entrenado.pkl')
firclfFRST3443 = joblib.load('FIRWSFRST3443_entrenado.pkl')
firclfKNST3443 = joblib.load('FIRWSKNST3443_entrenado.pkl')
firclfMLRG3443 = joblib.load('FIRWSMLRG3443_entrenado.pkl')
firclfFRST3540 = joblib.load('FIRWSFRST3540_entrenado.pkl')
firclfKNST3540 = joblib.load('FIRWSKNST3540_entrenado.pkl')
firclfMLRG3540 = joblib.load('FIRWSMLRG3540_entrenado.pkl')
#second
secclfFRST10 = joblib.load('SECWSFRST10_entrenado.pkl')
secclfKNST10 = joblib.load('SECWSKNST10_entrenado.pkl')
secclfMLRG10 = joblib.load('SECWSMLRG10_entrenado.pkl')
secclfFRST3045 = joblib.load('SECWSFRST3045_entrenado.pkl')
secclfKNST3045 = joblib.load('SECWSKNST3045_entrenado.pkl')
secclfMLRG3045 = joblib.load('SECWSMLRG3045_entrenado.pkl')
secclfFRST3344 = joblib.load('SECWSFRST3344_entrenado.pkl')
secclfKNST3344 = joblib.load('SECWSKNST3344_entrenado.pkl')
secclfMLRG3344 = joblib.load('SECWSMLRG3344_entrenado.pkl')
secclfFRST3443 = joblib.load('SECWSFRST3443_entrenado.pkl')
secclfKNST3443 = joblib.load('SECWSKNST3443_entrenado.pkl')
secclfMLRG3443 = joblib.load('SECWSMLRG3443_entrenado.pkl')
secclfFRST3540 = joblib.load('SECWSFRST3540_entrenado.pkl')
secclfKNST3540 = joblib.load('SECWSKNST3540_entrenado.pkl')
secclfMLRG3540 = joblib.load('SECWSMLRG3540_entrenado.pkl')
clear_output(wait=True)
ahora = time.strftime(formato_hora)
print(ahora)
print("TRAINING OK")
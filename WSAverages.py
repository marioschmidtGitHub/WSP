# A V E R A G E S
# This module update the patterns reference values which should be improved by the current spike so that one investment signals can be delivered.
# Patterns DataFrame is a parcial view of historic DF. It has the most representative, around 2%, Spikes codes.
# Patterns DF has date & time, code, trend, cycle, 30FBP and 30SBP spike data. 22OV are not includaded
# We will compare current spike with all the spikes in patterns in order to find if the current has better prediction that someone with the corresponding spike code in patterns.
# To make this comparison we need the averages for the 60 variables, 30FBP plus 30SBP. We wil consider the averages for the complete histaoric DF and another set of averages for the patterns DF, which are a subset of the historic DF.
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
print(ahora)
print("Close Patterns & Historic.xlsx")
print("Updating Averages. Please wait...")
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
historic = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
averages = pd.read_excel(r'C:\Users\Mario\Dropbox\WSGH\Averages.xlsx')
####### BACKTESTING ###########
# 54 for all the row/spikes and for the 3 predictor in Historic:
#        FIRST: using the 22OV as input, 6 predictions will be calculated (3 UP/DW and 3 for the spike codes, one for each predictor):
#                Probability that it belongs to one UP Cycle, if it did, and update the Historic DF corresponding field
#                Probability that it belongs to one DOWN cycle, if it did, and update the Historic DF corresponding field
#                Probability that it be one code 30 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 33 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 34 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 35 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 40 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 43 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 44 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 45 spike, if it was, and update the Historic DF corresponding field
#                These values will be used as reference to compare current spike.
#                Historic DF has 30 columns to store the values for the first level prediction, 24 of them are 0 for each row/spike.
#      SECOND: using the 6 values predicted in the previous step as input, another 6 will be updated as a second level prediction: 
#                Probability that it belongs to one UP cycle, if it did, and update the Historic DF corresponding field
#                Probability that it belongs to one DOWN cycle, if it did, and update the Historic DF corresponding field
#                Probability that it be one code 30 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 33 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 34 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 35 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 40 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 43 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 44 spike, if it was, and update the Historic DF corresponding field
#                Probability that it be one code 45 spike, if it was, and update the Historic DF corresponding field
#                These values will be also used as reference to compare current spike.
#                These is a second level prediction. It is the prediction of the prediction.
#                Historic DF has 30 columns to store the values for the first level prediction, 24 of them are 0 for each row/spike.
# Historic DF has 90 columns
for i in range (0,len(historic)):   
    spikefir=historic.loc[i,'Date':'AA']
    spikecode=spikefir.loc['E']
    spikeUP = [30,33,34,35]
    spikeDW=[40,43,44,45]
    if spikecode in spikeUP:
        UPDOWN=1
    elif spikecode in spikeDW:
        UPDOWN=0
    else:
        print("spikecode error in Averages 18")
    spikefir.drop(['A', 'Date', 'C', 'E', 'F'], inplace=True)
    spikefir = np.array(spikefir)
    spikefir=spikefir.reshape(1,spikefir.shape[0]) 
    forecastFIRFRST10 = firclfFRST10.predict_proba(spikefir)
    spikesecF=historic.loc[i,['D'] + ['FRSTUP'] + ['FRSTDW'] + ['FRST30'] + ['FRST45'] + ['FRST33'] + ['FRST44'] + ['FRST34'] + ['FRST43'] + ['FRST35'] + ['FRST40']]
    spikesecF.fillna(0, inplace=True)
    spikesecF = np.array(spikesecF)
    spikesecF=spikesecF.reshape(1,spikesecF.shape[0])  
    spikesecK=historic.loc[i,['D'] + ['KNSTUP'] + ['KNSTDW'] + ['KNST30'] + ['KNST45'] + ['KNST33'] + ['KNST44'] + ['KNST34'] + ['KNST43'] + ['KNST35'] + ['KNST40']] 
    spikesecK.fillna(0, inplace=True)
    spikesecK = np.array(spikesecK)
    spikesecK=spikesecK.reshape(1,spikesecK.shape[0]) 
    spikesecM=historic.loc[i,['D'] + ['MLRGUP'] + ['MLRGDW'] + ['MLRG30'] + ['MLRG45'] + ['MLRG33'] + ['MLRG44'] + ['MLRG34'] + ['MLRG43'] + ['MLRG35'] + ['MLRG40']]
    spikesecM.fillna(0, inplace=True)
    spikesecM = np.array(spikesecM)
    spikesecM=spikesecM.reshape(1,spikesecM.shape[0]) 
    forecastSECFRST10 = secclfFRST10.predict_proba(spikesecF) 
    if UPDOWN ==1:
        historic.loc[i,'FRSTUP'] = (forecastFIRFRST10[:,1])
        historic.loc[i,'SECFRSTUP'] = (forecastSECFRST10[:,1])
    else:
        historic.loc[i,'FRSTDW'] = (forecastFIRFRST10[:,1])
        historic.loc[i,'SECFRSTDW'] = (forecastSECFRST10[:,1])
    #print(forecastFIRFRST10[:,1]) 
    ##### KNeareST ########
    forecastFIRKNST10 = firclfKNST10.predict_proba(spikefir)
    forecastSECKNST10 = secclfKNST10.predict_proba(spikesecK)
    if UPDOWN ==1:
        historic.loc[i,'KNSTUP'] = (forecastFIRKNST10[:,1])
        historic.loc[i,'SECKNSTUP'] = (forecastSECKNST10[:,1])
    else:
        historic.loc[i,'KNSTDW'] = (forecastFIRKNST10[:,1])
        historic.loc[i,'SECKNSTDW'] = (forecastSECKNST10[:,1])
        ##### MLRG ########
    forecastFIRMLRG10 = firclfMLRG10.predict(spikefir)
    forecastSECMLRG10 = secclfMLRG10.predict(spikesecM)
    if UPDOWN ==1:
        historic.loc[i,'MLRGUP'] = forecastFIRMLRG10
        historic.loc[i,'SECMLRGUP'] = forecastSECMLRG10
    else:
        historic.loc[i,'MLRGDW'] = forecastFIRMLRG10
        historic.loc[i,'SECMLRGDW'] = forecastSECMLRG10
    if spikecode==30:
        forecastFIRFRST3045 = firclfFRST3045.predict_proba(spikefir)
        forecastSECFRST3045 = secclfFRST3045.predict_proba(spikesecF)
        historic.loc[i,'FRST30'] = (forecastFIRFRST3045[:,1])
        historic.loc[i,'SECFRST30'] = (forecastSECFRST3045[:,1])
        ##### Knearest ########
        forecastFIRKNST3045 = firclfKNST3045.predict_proba(spikefir)
        forecastSECKNST3045 = secclfKNST3045.predict_proba(spikesecK)
        historic.loc[i,'KNST30'] = (forecastFIRKNST3045[:,1])
        historic.loc[i,'SECKNST30'] = (forecastSECKNST3045[:,1])
        ##### MLRG ########
        forecastFIRMLRG3045 = firclfMLRG3045.predict(spikefir)
        forecastSECMLRG3045 = secclfMLRG3045.predict(spikesecM)
        historic.loc[i,'MLRG30'] = forecastFIRMLRG3045
        historic.loc[i,'SECMLRG30'] = forecastSECMLRG3045
    elif spikecode==45:      
        forecastFIRFRST3045 = firclfFRST3045.predict_proba(spikefir)
        forecastSECFRST3045 = secclfFRST3045.predict_proba(spikesecF)
        historic.loc[i,'FRST45'] = (forecastFIRFRST3045[:,1])
        historic.loc[i,'SECFRST45'] = (forecastSECFRST3045[:,1])
        ##### Knearest ########
        forecastFIRKNST3045 = firclfKNST3045.predict_proba(spikefir)
        forecastSECKNST3045 = secclfKNST3045.predict_proba(spikesecK)
        historic.loc[i,'KNST45'] = (forecastFIRKNST3045[:,1])
        historic.loc[i,'SECKNST45'] = (forecastSECKNST3045[:,1])
        ##### MLRG ########
        forecastFIRMLRG3045 = firclfMLRG3045.predict(spikefir)
        forecastSECMLRG3045 = secclfMLRG3045.predict(spikesecM)
        historic.loc[i,'MLRG45'] = forecastFIRMLRG3045
        historic.loc[i,'SECMLRG45'] = forecastSECMLRG3045
    elif spikecode==33:
        forecastFIRFRST3344 = firclfFRST3344.predict_proba(spikefir)
        forecastSECFRST3344 = secclfFRST3344.predict_proba(spikesecF)
        historic.loc[i,'FRST33'] = (forecastFIRFRST3344[:,1])
        historic.loc[i,'SECFRST33'] = (forecastSECFRST3344[:,1])
        ##### Knearest ########
        forecastFIRKNST3344 = firclfKNST3344.predict_proba(spikefir)
        forecastSECKNST3344 = secclfKNST3344.predict_proba(spikesecK)
        historic.loc[i,'KNST33'] = (forecastFIRKNST3344[:,1])
        historic.loc[i,'SECKNST33'] = (forecastSECKNST3344[:,1])
        ##### MLRG ########
        forecastFIRMLRG3344 = firclfMLRG3344.predict(spikefir)
        forecastSECMLRG3344 = secclfMLRG3344.predict(spikesecM)
        historic.loc[i,'MLRG33'] = forecastFIRMLRG3344
        historic.loc[i,'SECMLRG33'] = forecastSECMLRG3344
    elif spikecode==44:
        forecastFIRFRST3344 = firclfFRST3344.predict_proba(spikefir)
        forecastSECFRST3344 = secclfFRST3344.predict_proba(spikesecF)
        historic.loc[i,'FRST44'] = (forecastFIRFRST3344[:,1])
        historic.loc[i,'SECFRST44'] = (forecastSECFRST3344[:,1])
        ##### Knearest ########
        forecastFIRKNST3344 = firclfKNST3344.predict_proba(spikefir)
        forecastSECKNST3344 = secclfKNST3344.predict_proba(spikesecK)
        historic.loc[i,'KNST44'] = (forecastFIRKNST3344[:,1])
        historic.loc[i,'SECKNST44'] = (forecastSECKNST3344[:,1])
        ##### MLRG ########
        forecastFIRMLRG3344 = firclfMLRG3344.predict(spikefir)
        forecastSECMLRG3344 = secclfMLRG3344.predict(spikesecM)
        historic.loc[i,'MLRG44'] = forecastFIRMLRG3344
        historic.loc[i,'SECMLRG44'] = forecastSECMLRG3344
    elif spikecode==43:
        forecastFIRFRST3443 = firclfFRST3443.predict_proba(spikefir)
        forecastSECFRST3443 = secclfFRST3443.predict_proba(spikesecF)
        historic.loc[i,'FRST43'] = (forecastFIRFRST3443[:,1])
        historic.loc[i,'SECFRST43'] = (forecastSECFRST3443[:,1])
        ##### Knearest ########
        forecastFIRKNST3443 = firclfKNST3443.predict_proba(spikefir)
        forecastSECKNST3443 = secclfKNST3443.predict_proba(spikesecK)
        historic.loc[i,'KNST43'] = (forecastFIRKNST3443[:,1])   
        historic.loc[i,'SECKNST43'] = (forecastSECKNST3443[:,1])
        ##### MLRG ########
        forecastFIRMLRG3443 = firclfMLRG3443.predict(spikefir)
        forecastSECMLRG3443 = secclfMLRG3443.predict(spikesecM)
        historic.loc[i,'MLRG43'] = forecastFIRMLRG3443
        historic.loc[i,'SECMLRG43'] = forecastSECMLRG3443
    elif spikecode==34:
        forecastFIRFRST3443 = firclfFRST3443.predict_proba(spikefir)
        forecastSECFRST3443 = secclfFRST3443.predict_proba(spikesecF)
        historic.loc[i,'FRST34'] = (forecastFIRFRST3443[:,1])
        historic.loc[i,'SECFRST34'] = (forecastSECFRST3443[:,1])
        ##### Knearest ########
        forecastFIRKNST3443 = firclfKNST3443.predict_proba(spikefir)
        forecastSECKNST3443 = secclfKNST3443.predict_proba(spikesecK)
        historic.loc[i,'KNST34'] = (forecastFIRKNST3443[:,1]) 
        historic.loc[i,'SECKNST34'] = (forecastSECKNST3443[:,1])
        ##### MLRG ########
        forecastFIRMLRG3443 = firclfMLRG3443.predict(spikefir)
        forecastSECMLRG3443 = secclfMLRG3443.predict(spikesecM)
        historic.loc[i,'MLRG34'] = forecastFIRMLRG3443
        historic.loc[i,'SECMLRG34'] = forecastSECMLRG3443
    elif spikecode==35:
        forecastFIRFRST3540 = firclfFRST3540.predict_proba(spikefir)
        forecastSECFRST3540 = secclfFRST3540.predict_proba(spikesecF)
        historic.loc[i,'FRST35'] = (forecastFIRFRST3540[:,1])
        historic.loc[i,'SECFRST35'] = (forecastSECFRST3540[:,1])
        ##### Knearest ########
        forecastFIRKNST3540 = firclfKNST3540.predict_proba(spikefir)
        forecastSECKNST3540 = secclfKNST3540.predict_proba(spikesecK)
        historic.loc[i,'KNST35'] = (forecastFIRKNST3540[:,1])
        historic.loc[i,'SECKNST35'] = (forecastSECKNST3540[:,1])
        ##### MLRG ########
        forecastFIRMLRG3540 = firclfMLRG3540.predict(spikefir)
        forecastSECMLRG3540 = secclfMLRG3540.predict(spikesecM)
        historic.loc[i,'MLRG35'] = forecastFIRMLRG3540
        historic.loc[i,'SECMLRG35'] = forecastSECMLRG3540
    elif spikecode==40:
        forecastFIRFRST3540 = firclfFRST3540.predict_proba(spikefir)
        forecastSECFRST3540 = secclfFRST3540.predict_proba(spikesecF)
        historic.loc[i,'FRST40'] = (forecastFIRFRST3540[:,1])
        historic.loc[i,'SECFRST40'] = (forecastSECFRST3540[:,1])
        ##### Knearest ########
        forecastFIRKNST3540 = firclfKNST3540.predict_proba(spikefir)
        forecastSECKNST3540 = secclfKNST3540.predict_proba(spikesecK)
        historic.loc[i,'KNST40'] = (forecastFIRKNST3540[:,1])
        historic.loc[i,'SECKNST40'] = (forecastSECKNST3540[:,1])
        ##### MLRG ########
        forecastFIRMLRG3540 = firclfMLRG3540.predict(spikefir)
        forecastSECMLRG3540 = secclfMLRG3540.predict(spikesecM)
        historic.loc[i,'MLRG40'] = forecastFIRMLRG3540
        historic.loc[i,'SECMLRG40'] = forecastSECMLRG3540

historic.to_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx', engine='xlsxwriter')
#patterns = pd.read_excel(r'C:\Users\Mario\Dropbox\Historic.xlsx')
# *************** AVERAGES *****************
#263. In this module the averages, maximun and minimun, for historic DF and for patterns DF are calculated and saved to load them further as a fit data.
patterns = historic.loc[:,'Date':'SECMLRG40'].reset_index(drop=True)
delete_row = patterns[patterns["PATTUP"]==22].index
patterns = patterns.drop(delete_row)
#patterns = patterns.loc[:,'Date':'MLRG40']
patterns.to_excel(r'C:\Users\Mario\Dropbox\WSGH\Patterns.xlsx', engine='xlsxwriter')
averages = pd.read_excel(r'C:\Users\Mario\Dropbox\WSGH\Averages.xlsx')
#270. patterns = historic's view with the 30 variables/values defined in Backtesting (30BV).
# The averages of each one of them is calculated for the complete Historic Dataframe, and for the patterns view.
print("Updating Dataframes")
hstFIRFRSTUP=historic.loc[:,'FRSTUP'].mean()
hstFIRFRSTDW=historic.loc[:,'FRSTDW'].mean()
hstFIRKNSTUP=historic.loc[:,'KNSTUP'].mean()
hstFIRKNSTDW=historic.loc[:,'KNSTDW'].mean()
hstFIRMLRGUP=historic.loc[:,'MLRGUP'].mean()
hstFIRMLRGDW=historic.loc[:,'MLRGDW'].mean()
hstFIRFRST30=historic.loc[:,'FRST30'].mean()
hstFIRFRST45=historic.loc[:,'FRST45'].mean()
hstFIRKNST30=historic.loc[:,'KNST30'].mean()
hstFIRKNST45=historic.loc[:,'KNST45'].mean()
hstFIRMLRG30=historic.loc[:,'MLRG30'].mean()
hstFIRMLRG45=historic.loc[:,'MLRG45'].mean()
hstFIRFRST33=historic.loc[:,'FRST33'].mean()
hstFIRFRST44=historic.loc[:,'FRST44'].mean()
hstFIRKNST33=historic.loc[:,'KNST33'].mean()
hstFIRKNST44=historic.loc[:,'KNST44'].mean()
hstFIRMLRG33=historic.loc[:,'MLRG33'].mean()
hstFIRMLRG44=historic.loc[:,'MLRG44'].mean()
hstFIRFRST34=historic.loc[:,'FRST34'].mean()
hstFIRFRST43=historic.loc[:,'FRST43'].mean()
hstFIRKNST34=historic.loc[:,'KNST34'].mean()
hstFIRKNST43=historic.loc[:,'KNST43'].mean()
hstFIRMLRG34=historic.loc[:,'MLRG34'].mean()
hstFIRMLRG43=historic.loc[:,'MLRG43'].mean()
hstFIRFRST35=historic.loc[:,'FRST35'].mean()
hstFIRFRST40=historic.loc[:,'FRST40'].mean()
hstFIRKNST35=historic.loc[:,'KNST35'].mean()
hstFIRKNST40=historic.loc[:,'KNST40'].mean()
hstFIRMLRG35=historic.loc[:,'MLRG35'].mean()
hstFIRMLRG40=historic.loc[:,'MLRG40'].mean()

patFIRFRSTUP=patterns.loc[:,'FRSTUP'].mean()
patFIRFRSTDW=patterns.loc[:,'FRSTDW'].mean()
patFIRKNSTUP=patterns.loc[:,'KNSTUP'].mean()
patFIRKNSTDW=patterns.loc[:,'KNSTDW'].mean()
patFIRMLRGUP=patterns.loc[:,'MLRGUP'].mean()
patFIRMLRGDW=patterns.loc[:,'MLRGDW'].mean()
patFIRFRST30=patterns.loc[:,'FRST30'].mean()
patFIRFRST45=patterns.loc[:,'FRST45'].mean()
patFIRKNST30=patterns.loc[:,'KNST30'].mean()
patFIRKNST45=patterns.loc[:,'KNST45'].mean()
patFIRMLRG30=patterns.loc[:,'MLRG30'].mean()
patFIRMLRG45=patterns.loc[:,'MLRG45'].mean()
patFIRFRST33=patterns.loc[:,'FRST33'].mean()
patFIRFRST44=patterns.loc[:,'FRST44'].mean()
patFIRKNST33=patterns.loc[:,'KNST33'].mean()
patFIRKNST44=patterns.loc[:,'KNST44'].mean()
patFIRMLRG33=patterns.loc[:,'MLRG33'].mean()
patFIRMLRG44=patterns.loc[:,'MLRG44'].mean()
patFIRFRST34=patterns.loc[:,'FRST34'].mean()
patFIRFRST43=patterns.loc[:,'FRST43'].mean()
patFIRKNST34=patterns.loc[:,'KNST34'].mean()
patFIRKNST43=patterns.loc[:,'KNST43'].mean()
patFIRMLRG34=patterns.loc[:,'MLRG34'].mean()
patFIRMLRG43=patterns.loc[:,'MLRG43'].mean()
patFIRFRST35=patterns.loc[:,'FRST35'].mean()
patFIRFRST40=patterns.loc[:,'FRST40'].mean()
patFIRKNST35=patterns.loc[:,'KNST35'].mean()
patFIRKNST40=patterns.loc[:,'KNST40'].mean()
patFIRMLRG35=patterns.loc[:,'MLRG35'].mean()
patFIRMLRG40=patterns.loc[:,'MLRG40'].mean()

avgFIRFRSTUP=max(patFIRFRSTUP,hstFIRFRSTUP)
avgFIRFRSTDW=min(patFIRFRSTDW,hstFIRFRSTDW)
avgFIRKNSTUP=max(patFIRKNSTUP,hstFIRKNSTUP)
avgFIRKNSTDW=min(patFIRKNSTDW,hstFIRKNSTDW)
avgFIRMLRGUP=max(patFIRMLRGUP,hstFIRMLRGUP)
avgFIRMLRGDW=min(patFIRMLRGDW,hstFIRMLRGDW)
avgFIRFRST30=max(patFIRFRST30,hstFIRFRST30)
avgFIRFRST45=min(patFIRFRST45,hstFIRFRST45)
avgFIRKNST30=max(patFIRKNST30,hstFIRKNST30)
avgFIRKNST45=min(patFIRKNST45,hstFIRKNST45)
avgFIRMLRG30=max(patFIRMLRG30,hstFIRMLRG30)
avgFIRMLRG45=min(patFIRMLRG45,hstFIRMLRG45)
avgFIRFRST33=max(patFIRFRST33,hstFIRFRST33)
avgFIRFRST44=min(patFIRFRST44,hstFIRFRST44)
avgFIRKNST33=max(patFIRKNST33,hstFIRKNST33)
avgFIRKNST44=min(patFIRKNST44,hstFIRKNST44)
avgFIRMLRG33=max(patFIRMLRG33,hstFIRMLRG33)
avgFIRMLRG44=min(patFIRMLRG44,hstFIRMLRG44)
avgFIRFRST34=max(patFIRFRST34,hstFIRFRST34)
avgFIRFRST43=min(patFIRFRST43,hstFIRFRST43)
avgFIRKNST34=max(patFIRKNST34,hstFIRKNST34)
avgFIRKNST43=min(patFIRKNST43,hstFIRKNST43)
avgFIRMLRG34=max(patFIRMLRG34,hstFIRMLRG34)
avgFIRMLRG43=min(patFIRMLRG43,hstFIRMLRG43)
avgFIRFRST35=max(patFIRFRST35,hstFIRFRST35)
avgFIRFRST40=min(patFIRFRST40,hstFIRFRST40)
avgFIRKNST35=max(patFIRKNST35,hstFIRKNST35)
avgFIRKNST40=min(patFIRKNST40,hstFIRKNST40)
avgFIRMLRG35=max(patFIRMLRG35,hstFIRMLRG35)
avgFIRMLRG40=min(patFIRMLRG40,hstFIRMLRG40)

averages.loc[0,'FF10']=("%.3f" % avgFIRFRSTUP)
averages.loc[1,'FF10']=("%.3f" % avgFIRFRSTDW)
averages.loc[0,'FK10']=("%.3f" % avgFIRKNSTUP)
averages.loc[1,'FK10']=("%.3f" % avgFIRKNSTDW)
averages.loc[0,'FM10']=("%.3f" % avgFIRMLRGUP)
averages.loc[1,'FM10']=("%.3f" % avgFIRMLRGDW)
averages.loc[0,'FF3045']=("%.3f" % avgFIRFRST30)
averages.loc[1,'FF3045']=("%.3f" % avgFIRFRST45)
averages.loc[0,'FK3045']=("%.3f" % avgFIRKNST30)
averages.loc[1,'FK3045']=("%.3f" % avgFIRKNST45)
averages.loc[0,'FM3045']=("%.3f" % avgFIRMLRG30)
averages.loc[1,'FM3045']=("%.3f" % avgFIRMLRG45)
averages.loc[0,'FF3344']=("%.3f" % avgFIRFRST33)
averages.loc[1,'FF3344']=("%.3f" % avgFIRFRST44)
averages.loc[0,'FK3344']=("%.3f" % avgFIRKNST33)
averages.loc[1,'FK3344']=("%.3f" % avgFIRKNST44)
averages.loc[0,'FM3344']=("%.3f" % avgFIRMLRG33)
averages.loc[1,'FM3344']=("%.3f" % avgFIRMLRG44)
averages.loc[0,'FF3443']=("%.3f" % avgFIRFRST34)
averages.loc[1,'FF3443']=("%.3f" % avgFIRFRST43)
averages.loc[0,'FK3443']=("%.3f" % avgFIRKNST34)
averages.loc[1,'FK3443']=("%.3f" % avgFIRKNST43)
averages.loc[0,'FM3443']=("%.3f" % avgFIRMLRG34)
averages.loc[1,'FM3443']=("%.3f" % avgFIRMLRG43)
averages.loc[0,'FF3540']=("%.3f" % avgFIRFRST35)
averages.loc[1,'FF3540']=("%.3f" % avgFIRFRST40)
averages.loc[0,'FK3540']=("%.3f" % avgFIRKNST35)
averages.loc[1,'FK3540']=("%.3f" % avgFIRKNST40)
averages.loc[0,'FM3540']=("%.3f" % avgFIRMLRG35)
averages.loc[1,'FM3540']=("%.3f" % avgFIRMLRG40)

averages.loc[2,'FF10']=("%.3f" % hstFIRFRSTUP)
averages.loc[3,'FF10']=("%.3f" % hstFIRFRSTDW)
averages.loc[2,'FK10']=("%.3f" % hstFIRKNSTUP)
averages.loc[3,'FK10']=("%.3f" % hstFIRKNSTDW)
averages.loc[2,'FM10']=("%.3f" % hstFIRMLRGUP)
averages.loc[3,'FM10']=("%.3f" % hstFIRMLRGDW)
averages.loc[2,'FF3045']=("%.3f" % hstFIRFRST30)
averages.loc[3,'FF3045']=("%.3f" % hstFIRFRST45)
averages.loc[2,'FK3045']=("%.3f" % hstFIRKNST30)
averages.loc[3,'FK3045']=("%.3f" % hstFIRKNST45)
averages.loc[2,'FM3045']=("%.3f" % hstFIRMLRG30)
averages.loc[3,'FM3045']=("%.3f" % hstFIRMLRG45)
averages.loc[2,'FF3344']=("%.3f" % hstFIRFRST33)
averages.loc[3,'FF3344']=("%.3f" % hstFIRFRST44)
averages.loc[2,'FK3344']=("%.3f" % hstFIRKNST33)
averages.loc[3,'FK3344']=("%.3f" % hstFIRKNST44)
averages.loc[2,'FM3344']=("%.3f" % hstFIRMLRG33)
averages.loc[3,'FM3344']=("%.3f" % hstFIRMLRG44)
averages.loc[2,'FF3443']=("%.3f" % hstFIRFRST34)
averages.loc[3,'FF3443']=("%.3f" % hstFIRFRST43)
averages.loc[2,'FK3443']=("%.3f" % hstFIRKNST34)
averages.loc[3,'FK3443']=("%.3f" % hstFIRKNST43)
averages.loc[2,'FM3443']=("%.3f" % hstFIRMLRG34)
averages.loc[3,'FM3443']=("%.3f" % hstFIRMLRG43)
averages.loc[2,'FF3540']=("%.3f" % hstFIRFRST35)
averages.loc[3,'FF3540']=("%.3f" % hstFIRFRST40)
averages.loc[2,'FK3540']=("%.3f" % hstFIRKNST35)
averages.loc[3,'FK3540']=("%.3f" % hstFIRKNST40)
averages.loc[2,'FM3540']=("%.3f" % hstFIRMLRG35)
averages.loc[3,'FM3540']=("%.3f" % hstFIRMLRG40)

averages.loc[4,'FF10']=("%.3f" % patFIRFRSTUP)
averages.loc[5,'FF10']=("%.3f" % patFIRFRSTDW)
averages.loc[4,'FK10']=("%.3f" % patFIRKNSTUP)
averages.loc[5,'FK10']=("%.3f" % patFIRKNSTDW)
averages.loc[4,'FM10']=("%.3f" % patFIRMLRGUP)
averages.loc[5,'FM10']=("%.3f" % patFIRMLRGDW)
averages.loc[4,'FF3045']=("%.3f" % patFIRFRST30)
averages.loc[5,'FF3045']=("%.3f" % patFIRFRST45)
averages.loc[4,'FK3045']=("%.3f" % patFIRKNST30)
averages.loc[5,'FK3045']=("%.3f" % patFIRKNST45)
averages.loc[4,'FM3045']=("%.3f" % patFIRMLRG30)
averages.loc[5,'FM3045']=("%.3f" % patFIRMLRG45)
averages.loc[4,'FF3344']=("%.3f" % patFIRFRST33)
averages.loc[5,'FF3344']=("%.3f" % patFIRFRST44)
averages.loc[4,'FK3344']=("%.3f" % patFIRKNST33)
averages.loc[5,'FK3344']=("%.3f" % patFIRKNST44)
averages.loc[4,'FM3344']=("%.3f" % patFIRMLRG33)
averages.loc[5,'FM3344']=("%.3f" % patFIRMLRG44)
averages.loc[4,'FF3443']=("%.3f" % patFIRFRST34)
averages.loc[5,'FF3443']=("%.3f" % patFIRFRST43)
averages.loc[4,'FK3443']=("%.3f" % patFIRKNST34)
averages.loc[5,'FK3443']=("%.3f" % patFIRKNST43)
averages.loc[4,'FM3443']=("%.3f" % patFIRMLRG34)
averages.loc[5,'FM3443']=("%.3f" % patFIRMLRG43)
averages.loc[4,'FF3540']=("%.3f" % patFIRFRST35)
averages.loc[5,'FF3540']=("%.3f" % patFIRFRST40)
averages.loc[4,'FK3540']=("%.3f" % patFIRKNST35)
averages.loc[5,'FK3540']=("%.3f" % patFIRKNST40)
averages.loc[4,'FM3540']=("%.3f" % patFIRMLRG35)
averages.loc[5,'FM3540']=("%.3f" % patFIRMLRG40)
averages.to_excel(r'C:\Users\Mario\Dropbox\WSGH\Averages.xlsx', engine='xlsxwriter')

hstSECFRSTUP=historic.loc[:,'SECFRSTUP'].mean()
hstSECFRSTDW=historic.loc[:,'SECFRSTDW'].mean()
hstSECKNSTUP=historic.loc[:,'SECKNSTUP'].mean()
hstSECKNSTDW=historic.loc[:,'SECKNSTDW'].mean()
hstSECMLRGUP=historic.loc[:,'SECMLRGUP'].mean()
hstSECMLRGDW=historic.loc[:,'SECMLRGDW'].mean()
hstSECFRST30=historic.loc[:,'SECFRST30'].mean()
hstSECFRST45=historic.loc[:,'SECFRST45'].mean()
hstSECKNST30=historic.loc[:,'SECKNST30'].mean()
hstSECKNST45=historic.loc[:,'SECKNST45'].mean()
hstSECMLRG30=historic.loc[:,'SECMLRG30'].mean()
hstSECMLRG45=historic.loc[:,'SECMLRG45'].mean()
hstSECFRST33=historic.loc[:,'SECFRST33'].mean()
hstSECFRST44=historic.loc[:,'SECFRST44'].mean()
hstSECKNST33=historic.loc[:,'SECKNST33'].mean()
hstSECKNST44=historic.loc[:,'SECKNST44'].mean()
hstSECMLRG33=historic.loc[:,'SECMLRG33'].mean()
hstSECMLRG44=historic.loc[:,'SECMLRG44'].mean()
hstSECFRST34=historic.loc[:,'SECFRST34'].mean()
hstSECFRST43=historic.loc[:,'SECFRST43'].mean()
hstSECKNST34=historic.loc[:,'SECKNST34'].mean()
hstSECKNST43=historic.loc[:,'SECKNST43'].mean()
hstSECMLRG34=historic.loc[:,'SECMLRG34'].mean()
hstSECMLRG43=historic.loc[:,'SECMLRG43'].mean()
hstSECFRST35=historic.loc[:,'SECFRST35'].mean()
hstSECFRST40=historic.loc[:,'SECFRST40'].mean()
hstSECKNST35=historic.loc[:,'SECKNST35'].mean()
hstSECKNST40=historic.loc[:,'SECKNST40'].mean()
hstSECMLRG35=historic.loc[:,'SECMLRG35'].mean()
hstSECMLRG40=historic.loc[:,'SECMLRG40'].mean()

patSECFRSTUP=patterns.loc[:,'SECFRSTUP'].mean()
patSECFRSTDW=patterns.loc[:,'SECFRSTDW'].mean()
patSECKNSTUP=patterns.loc[:,'SECKNSTUP'].mean()
patSECKNSTDW=patterns.loc[:,'SECKNSTDW'].mean()
patSECMLRGUP=patterns.loc[:,'SECMLRGUP'].mean()
patSECMLRGDW=patterns.loc[:,'SECMLRGDW'].mean()
patSECFRST30=patterns.loc[:,'SECFRST30'].mean()
patSECFRST45=patterns.loc[:,'SECFRST45'].mean()
patSECKNST30=patterns.loc[:,'SECKNST30'].mean()
patSECKNST45=patterns.loc[:,'SECKNST45'].mean()
patSECMLRG30=patterns.loc[:,'SECMLRG30'].mean()
patSECMLRG45=patterns.loc[:,'SECMLRG45'].mean()
patSECFRST33=patterns.loc[:,'SECFRST33'].mean()
patSECFRST44=patterns.loc[:,'SECFRST44'].mean()
patSECKNST33=patterns.loc[:,'SECKNST33'].mean()
patSECKNST44=patterns.loc[:,'SECKNST44'].mean()
patSECMLRG33=patterns.loc[:,'SECMLRG33'].mean()
patSECMLRG44=patterns.loc[:,'SECMLRG44'].mean()
patSECFRST34=patterns.loc[:,'SECFRST34'].mean()
patSECFRST43=patterns.loc[:,'SECFRST43'].mean()
patSECKNST34=patterns.loc[:,'SECKNST34'].mean()
patSECKNST43=patterns.loc[:,'SECKNST43'].mean()
patSECMLRG34=patterns.loc[:,'SECMLRG34'].mean()
patSECMLRG43=patterns.loc[:,'SECMLRG43'].mean()
patSECFRST35=patterns.loc[:,'SECFRST35'].mean()
patSECFRST40=patterns.loc[:,'SECFRST40'].mean()
patSECKNST35=patterns.loc[:,'SECKNST35'].mean()
patSECKNST40=patterns.loc[:,'SECKNST40'].mean()
patSECMLRG35=patterns.loc[:,'SECMLRG35'].mean()
patSECMLRG40=patterns.loc[:,'SECMLRG40'].mean()
# 522. If the Spike belongs to one UP trend, the maximun value is stored, else the minimun.
avgSECFRSTUP=max(patSECFRSTUP,hstSECFRSTUP)
avgSECFRSTDW=min(patSECFRSTDW,hstSECFRSTDW)
avgSECKNSTUP=max(patSECKNSTUP,hstSECKNSTUP)
avgSECKNSTDW=min(patSECKNSTDW,hstSECKNSTDW)
avgSECMLRGUP=max(patSECMLRGUP,hstSECMLRGUP)
avgSECMLRGDW=min(patSECMLRGDW,hstSECMLRGDW)
avgSECFRST30=max(patSECFRST30,hstSECFRST30)
avgSECFRST45=min(patSECFRST45,hstSECFRST45)
avgSECKNST30=max(patSECKNST30,hstSECKNST30)
avgSECKNST45=min(patSECKNST45,hstSECKNST45)
avgSECMLRG30=max(patSECMLRG30,hstSECMLRG30)
avgSECMLRG45=min(patSECMLRG45,hstSECMLRG45)
avgSECFRST33=max(patSECFRST33,hstSECFRST33)
avgSECFRST44=min(patSECFRST44,hstSECFRST44)
avgSECKNST33=max(patSECKNST33,hstSECKNST33)
avgSECKNST44=min(patSECKNST44,hstSECKNST44)
avgSECMLRG33=max(patSECMLRG33,hstSECMLRG33)
avgSECMLRG44=min(patSECMLRG44,hstSECMLRG44)
avgSECFRST34=max(patSECFRST34,hstSECFRST34)
avgSECFRST43=min(patSECFRST43,hstSECFRST43)
avgSECKNST34=max(patSECKNST34,hstSECKNST34)
avgSECKNST43=min(patSECKNST43,hstSECKNST43)
avgSECMLRG34=max(patSECMLRG34,hstSECMLRG34)
avgSECMLRG43=min(patSECMLRG43,hstSECMLRG43)
avgSECFRST35=max(patSECFRST35,hstSECFRST35)
avgSECFRST40=min(patSECFRST40,hstSECFRST40)
avgSECKNST35=max(patSECKNST35,hstSECKNST35)
avgSECKNST40=min(patSECKNST40,hstSECKNST40)
avgSECMLRG35=max(patSECMLRG35,hstSECMLRG35)
avgSECMLRG40=min(patSECMLRG40,hstSECMLRG40)

averages.loc[0,'SF10']=("%.3f" % avgSECFRSTUP)
averages.loc[1,'SF10']=("%.3f" % avgSECFRSTDW)
averages.loc[0,'SK10']=("%.3f" % avgSECKNSTUP)
averages.loc[1,'SK10']=("%.3f" % avgSECKNSTDW)
averages.loc[0,'SM10']=("%.3f" % avgSECMLRGUP)
averages.loc[1,'SM10']=("%.3f" % avgSECMLRGDW)
averages.loc[0,'SF3045']=("%.3f" % avgSECFRST30)
averages.loc[1,'SF3045']=("%.3f" % avgSECFRST45)
averages.loc[0,'SK3045']=("%.3f" % avgSECKNST30)
averages.loc[1,'SK3045']=("%.3f" % avgSECKNST45)
averages.loc[0,'SM3045']=("%.3f" % avgSECMLRG30)
averages.loc[1,'SM3045']=("%.3f" % avgSECMLRG45)
averages.loc[0,'SF3344']=("%.3f" % avgSECFRST33)
averages.loc[1,'SF3344']=("%.3f" % avgSECFRST44)
averages.loc[0,'SK3344']=("%.3f" % avgSECKNST33)
averages.loc[1,'SK3344']=("%.3f" % avgSECKNST44)
averages.loc[0,'SM3344']=("%.3f" % avgSECMLRG33)
averages.loc[1,'SM3344']=("%.3f" % avgSECMLRG44)
averages.loc[0,'SF3443']=("%.3f" % avgSECFRST34)
averages.loc[1,'SF3443']=("%.3f" % avgSECFRST43)
averages.loc[0,'SK3443']=("%.3f" % avgSECKNST34)
averages.loc[1,'SK3443']=("%.3f" % avgSECKNST43)
averages.loc[0,'SM3443']=("%.3f" % avgSECMLRG34)
averages.loc[1,'SM3443']=("%.3f" % avgSECMLRG43)
averages.loc[0,'SF3540']=("%.3f" % avgSECFRST35)
averages.loc[1,'SF3540']=("%.3f" % avgSECFRST40)
averages.loc[0,'SK3540']=("%.3f" % avgSECKNST35)
averages.loc[1,'SK3540']=("%.3f" % avgSECKNST40)
averages.loc[0,'SM3540']=("%.3f" % avgSECMLRG35)
averages.loc[1,'SM3540']=("%.3f" % avgSECMLRG40)

averages.loc[2,'SF10']=("%.3f" % hstSECFRSTUP)
averages.loc[3,'SF10']=("%.3f" % hstSECFRSTDW)
averages.loc[2,'SK10']=("%.3f" % hstSECKNSTUP)
averages.loc[3,'SK10']=("%.3f" % hstSECKNSTDW)
averages.loc[2,'SM10']=("%.3f" % hstSECMLRGUP)
averages.loc[3,'SM10']=("%.3f" % hstSECMLRGDW)
averages.loc[2,'SF3045']=("%.3f" % hstSECFRST30)
averages.loc[3,'SF3045']=("%.3f" % hstSECFRST45)
averages.loc[2,'SK3045']=("%.3f" % hstSECKNST30)
averages.loc[3,'SK3045']=("%.3f" % hstSECKNST45)
averages.loc[2,'SM3045']=("%.3f" % hstSECMLRG30)
averages.loc[3,'SM3045']=("%.3f" % hstSECMLRG45)
averages.loc[2,'SF3344']=("%.3f" % hstSECFRST33)
averages.loc[3,'SF3344']=("%.3f" % hstSECFRST44)
averages.loc[2,'SK3344']=("%.3f" % hstSECKNST33)
averages.loc[3,'SK3344']=("%.3f" % hstSECKNST44)
averages.loc[2,'SM3344']=("%.3f" % hstSECMLRG33)
averages.loc[3,'SM3344']=("%.3f" % hstSECMLRG44)
averages.loc[2,'SF3443']=("%.3f" % hstSECFRST34)
averages.loc[3,'SF3443']=("%.3f" % hstSECFRST43)
averages.loc[2,'SK3443']=("%.3f" % hstSECKNST34)
averages.loc[3,'SK3443']=("%.3f" % hstSECKNST43)
averages.loc[2,'SM3443']=("%.3f" % hstSECMLRG34)
averages.loc[3,'SM3443']=("%.3f" % hstSECMLRG43)
averages.loc[2,'SF3540']=("%.3f" % hstSECFRST35)
averages.loc[3,'SF3540']=("%.3f" % hstSECFRST40)
averages.loc[2,'SK3540']=("%.3f" % hstSECKNST35)
averages.loc[3,'SK3540']=("%.3f" % hstSECKNST40)
averages.loc[2,'SM3540']=("%.3f" % hstSECMLRG35)
averages.loc[3,'SM3540']=("%.3f" % hstSECMLRG40)

averages.loc[4,'SF10']=("%.3f" % patSECFRSTUP)
averages.loc[5,'SF10']=("%.3f" % patSECFRSTDW)
averages.loc[4,'SK10']=("%.3f" % patSECKNSTUP)
averages.loc[5,'SK10']=("%.3f" % patSECKNSTDW)
averages.loc[4,'SM10']=("%.3f" % patSECMLRGUP)
averages.loc[5,'SM10']=("%.3f" % patSECMLRGDW)
averages.loc[4,'SF3045']=("%.3f" % patSECFRST30)
averages.loc[5,'SF3045']=("%.3f" % patSECFRST45)
averages.loc[4,'SK3045']=("%.3f" % patSECKNST30)
averages.loc[5,'SK3045']=("%.3f" % patSECKNST45)
averages.loc[4,'SM3045']=("%.3f" % patSECMLRG30)
averages.loc[5,'SM3045']=("%.3f" % patSECMLRG45)
averages.loc[4,'SF3344']=("%.3f" % patSECFRST33)
averages.loc[5,'SF3344']=("%.3f" % patSECFRST44)
averages.loc[4,'SK3344']=("%.3f" % patSECKNST33)
averages.loc[5,'SK3344']=("%.3f" % patSECKNST44)
averages.loc[4,'SM3344']=("%.3f" % patSECMLRG33)
averages.loc[5,'SM3344']=("%.3f" % patSECMLRG44)
averages.loc[4,'SF3443']=("%.3f" % patSECFRST34)
averages.loc[5,'SF3443']=("%.3f" % patSECFRST43)
averages.loc[4,'SK3443']=("%.3f" % patSECKNST34)
averages.loc[5,'SK3443']=("%.3f" % patSECKNST43)
averages.loc[4,'SM3443']=("%.3f" % patSECMLRG34)
averages.loc[5,'SM3443']=("%.3f" % patSECMLRG43)
averages.loc[4,'SF3540']=("%.3f" % patSECFRST35)
averages.loc[5,'SF3540']=("%.3f" % patSECFRST40)
averages.loc[4,'SK3540']=("%.3f" % patSECKNST35)
averages.loc[5,'SK3540']=("%.3f" % patSECKNST40)
averages.loc[4,'SM3540']=("%.3f" % patSECMLRG35)
averages.loc[5,'SM3540']=("%.3f" % patSECMLRG40)
averages.to_excel(r'C:\Users\Mario\Dropbox\WSGH\Averages.xlsx', engine='xlsxwriter')
clear_output(wait=True)
ahora = time.strftime(formato_hora)
print(ahora)
print("averages OK")
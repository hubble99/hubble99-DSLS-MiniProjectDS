# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 00:53:10 2020

@author: Noreen
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:48:13 2020

@author: Noreen
"""



import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve,classification_report
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


#    F1_scoretr=f1_score(y_train, y_pred)
#    ls.append(F1_scoretr)
#    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()


    #precision,recall,f1=metrics.classification_report(y_test, y_pred, digits=3)
   # Sensitivity=tp/(tp+fn)
#    ls.append(Sensitivity)
   
#    ls.append(score)
#    print(i,)
#    auc_ho = roc_auc_score(label, score)
#    ls.append(auc_ho)
   
   
   
   

def clean_data(data):
       '''in cleaning we have yet done a very little part i.e. converting string variables into integer or float variables
because machine learning models work with numerical values so it is mandatory to convert input and output variables into
numerical variables. We can also use pd.get_dummies() method to convert directly but in our case this method will be unable
to find corresponding numerical values'''
       # locate in data where the data['Day'] is equal to Sunday and replace the Day column at all those positions with 0
       data['Day'] = data['Day'].str.replace('Sunday', '0')
       data.loc[data['Day'] == 'Sunday', 'Day'] = 0
       data.loc[data['Day'] == 'Monday', 'Day'] = 1
       data.loc[data['Day'] == 'Tuesday', 'Day'] = 2
       data.loc[data['Day'] == 'Wednesday', 'Day'] = 3
       data.loc[data['Day'] == 'Thursday', 'Day'] = 4
       data.loc[data['Day'] == 'Friday', 'Day'] = 5
       data.loc[data['Day'] == 'Saturday', 'Day'] = 6

       data.loc[data['Weather'] == 'sunny', 'Weather'] = 0
       data.loc[data['Weather'] == 'cloudy', 'Weather'] = 1
       data.loc[data['Weather'] == 'Rain', 'Weather'] = 2
       data.loc[data['Weather'] == 'Showers', 'Weather'] = 2
       data.loc[data['Weather'] == 'Clear', 'Weather'] = 0
       data.loc[data['Weather'] == 'Mostly Cloudy', 'Weather'] = 1
       data.loc[data['Weather'] == 'Cloudy', 'Weather'] = 1
       data.loc[data['Weather'] == 'Partly Cloudy', 'Weather'] = 1
       data.loc[data['Weather'] == 'Mostly Clear', 'Weather'] = 0
       data.loc[data['Weather'] == 'Sunny', 'Weather'] = 0
       data.loc[data['Weather'] == 'Mostly Sunny', 'Weather'] = 0
       
       #data.loc[data['Time'] == 'peak_hour', 'Time'] = 1
       #data.loc[data['Time'] == 'non_peak_hour', 'Time'] = 0

       data.loc[data['Holiday'] == 'yes', 'Holiday'] = 1
       data.loc[data['Holiday'] == 'no', 'Holiday'] = 0

       data.loc[data['Special_Condition'] == 'yes', 'Special_Condition'] = 1
       data.loc[data['Special_Condition'] == 'no', 'Special_Condition'] = 0

       count = 0
       # unique() returns list of every unique item in the pandas series
       # pandas series means a column in pandas
       for location in data['Starting_Location'].unique():
              # replace the location names with count variable value both in starting and destination columns
              # e.g. saddar will have value of 1 in both columns
              data.loc[data['Starting_Location'] == location, 'Starting_Location'] = count
              data.loc[data['Destination_Location'] == location, 'Destination_Location'] = count
              count += 1
       # data['Starting_Location'].unique() will not return shamsabad because it is not present in data['Starting_Location']
       # so we'll have to seperately assign a count number to it in Destination_Location
       data.loc[data['Destination_Location'] == 'shamsabad', 'Destination_Location'] = count
      
       data.loc[data['Destination_Location'] == 'eventox_event_management', 'Destination_Location'] = 108
       data.loc[data['Destination_Location'] == 'pak_export', 'Destination_Location'] = 50
       data.loc[data['Destination_Location'] == 'gerry_centre', 'Destination_Location'] = 51
       data.loc[data['Destination_Location'] == 'geo_news_islamabad_office', 'Destination_Location'] = 52
       data.loc[data['Destination_Location'] == 'D_chowk', 'Destination_Location'] = 53
       data.loc[data['Destination_Location'] == 'directorat_general_of_intelligence_and_investigation_FBR', 'Destination_Location'] = 54
       data.loc[data['Destination_Location'] == 'Jamia Masjid Bilal', 'Destination_Location'] = 55
       data.loc[data['Destination_Location'] == 'lastpoint', 'Destination_Location'] = 56
       data.loc[data['Destination_Location'] == 'last_point', 'Destination_Location'] = 57
       data.loc[data['Destination_Location'] == 'last_stop', 'Destination_Location'] = 58
       data.loc[data['Destination_Location'] == 'DC_office', 'Destination_Location'] = 59
       data.loc[data['Destination_Location'] == 'faisa_avenue_last_stop', 'Destination_Location'] = 60
       
       data.loc[data['Fastest_Route_Name'] == 'Murree Rd', 'Fastest_Route_Name'] = 0
       data.loc[data['Fastest_Route_Name'] == '6th Rd', 'Fastest_Route_Name'] = 1
       data.loc[data['Fastest_Route_Name'] == '9th Ave', 'Fastest_Route_Name'] = 2
       data.loc[data['Fastest_Route_Name'] == 'Jinnah Avenue and 9th Ave', 'Fastest_Route_Name'] = 3
       data.loc[data['Fastest_Route_Name'] == 'Faisal Avenue Flyover and Jinnah Avenue', 'Fastest_Route_Name'] = 4
       data.loc[data['Fastest_Route_Name'] == 'Jinnah Ave and Faisal Avenue Flyover', 'Fastest_Route_Name'] = 5
       data.loc[data['Fastest_Route_Name'] == 'Jinnah Ave', 'Fastest_Route_Name'] = 6
       data.loc[data['Fastest_Route_Name'] == 'Agha Khan Rd and Jinnah Ave', 'Fastest_Route_Name'] = 7
       data.loc[data['Fastest_Route_Name'] == 'Constitution Ave and Jinnah Ave', 'Fastest_Route_Name'] = 8
       data.loc[data['Fastest_Route_Name'] == 'I.J.P. Road and Stadium Rd', 'Fastest_Route_Name'] = 9
       data.loc[data['Fastest_Route_Name'] == '9th Ave and Stadium Rd', 'Fastest_Route_Name'] = 10
       data.loc[data['Fastest_Route_Name'] == 'Constitution Ave, A.K.M. Fazl-ul-Haq Road and Jinnah Ave', 'Fastest_Route_Name'] = 11
       data.loc[data['Fastest_Route_Name'] == 'Service Rd South I 8', 'Fastest_Route_Name'] = 12
       data.loc[data['Fastest_Route_Name'] == 'Jinnah Avenue', 'Fastest_Route_Name'] = 13
       data.loc[data['Fastest_Route_Name'] == '7th Ave', 'Fastest_Route_Name'] = 15
       data.loc[data['Fastest_Route_Name'] == '9th Av', 'Fastest_Route_Name'] = 16
       data.loc[data['Fastest_Route_Name'] == '9th Ave and Kashmir Hwy', 'Fastest_Route_Name'] = 17
       data.loc[data['Fastest_Route_Name'] == '9th Ave,Kashmir Hwy and 7th Ave', 'Fastest_Route_Name'] = 18
       data.loc[data['Fastest_Route_Name'] == 'A.K. Fazl-ul-Haq Rd', 'Fastest_Route_Name'] = 19
       data.loc[data['Fastest_Route_Name'] == 'A.K. Fazl-ul-Haq Rd and Jinnah Ave', 'Fastest_Route_Name'] = 20
       data.loc[data['Fastest_Route_Name'] == 'Agha Khan Rd', 'Fastest_Route_Name'] = 21
       data.loc[data['Fastest_Route_Name'] == 'Agha Khan Rd and Jinnah Ave', 'Fastest_Route_Name'] = 22
       data.loc[data['Fastest_Route_Name'] == 'Ahmed Faraz Rd and Aun Muhammad Rizvi Rd', 'Fastest_Route_Name'] = 23
       data.loc[data['Fastest_Route_Name'] == 'Awan-e-Sanat-o-Tiijarat', 'Fastest_Route_Name'] = 24
       data.loc[data['Fastest_Route_Name'] == 'Awan-e-Sanat-o-Tiijarat and Kashmir Hwy', 'Fastest_Route_Name'] = 25
       data.loc[data['Fastest_Route_Name'] == 'Jinnah Avenue and 9th Ave', 'Fastest_Route_Name'] = 26
       data.loc[data['Fastest_Route_Name'] == 'Faisal Avenue Flyover and Jinnah Avenue', 'Fastest_Route_Name'] = 27
       data.loc[data['Fastest_Route_Name'] == 'Jinnah Ave and Faisal Avenue Flyover', 'Fastest_Route_Name'] = 28
       data.loc[data['Fastest_Route_Name'] == 'Jinnah Ave', 'Fastest_Route_Name'] = 29
       data.loc[data['Fastest_Route_Name'] == 'Constitution Ave and Jinnah Ave', 'Fastest_Route_Name'] = 30
       data.loc[data['Fastest_Route_Name'] == '9th Ave and Stadium Rd', 'Fastest_Route_Name'] = 31
       data.loc[data['Fastest_Route_Name'] == 'Agha Khan Rd and Jinnah Ave', 'Fastest_Route_Name'] =32 
       data.loc[data['Fastest_Route_Name'] == 'Constitution Ave', 'Fastest_Route_Name'] = 33
       data.loc[data['Fastest_Route_Name'] == 'Constitution Ave and Jinnah Ave', 'Fastest_Route_Name'] =34 
       data.loc[data['Fastest_Route_Name'] == 'I.J.P. Road', 'Fastest_Route_Name'] = 35
       data.loc[data['Fastest_Route_Name'] == 'Ibn-e-Sina Rd', 'Fastest_Route_Name'] =36 
       data.loc[data['Fastest_Route_Name'] == 'Ismail Zabeeh Rd and Faisal Ave', 'Fastest_Route_Name'] = 37
       data.loc[data['Fastest_Route_Name'] == 'Jinnah Avenue and 9th Ave', 'Fastest_Route_Name'] = 38
       data.loc[data['Fastest_Route_Name'] == 'Jinnah Avenue and Faisal Avenue Flyover', 'Fastest_Route_Name'] = 39
       data.loc[data['Fastest_Route_Name'] == 'Kashmir Hwy', 'Fastest_Route_Name'] = 40
       data.loc[data['Fastest_Route_Name'] == 'Main Margalla Rd', 'Fastest_Route_Name'] = 41
       data.loc[data['Fastest_Route_Name'] == 'Nazim-ud-din Rd', 'Fastest_Route_Name'] = 42
       data.loc[data['Fastest_Route_Name'] == 'Parbat Rd and 7th Ave', 'Fastest_Route_Name'] = 43
       data.loc[data['Fastest_Route_Name'] == 'Service Rd E', 'Fastest_Route_Name'] = 44
       data.loc[data['Fastest_Route_Name'] == 'Service Rd South I 8', 'Fastest_Route_Name'] =45 
       data.loc[data['Fastest_Route_Name'] == 'Service Rd W', 'Fastest_Route_Name'] =46 
       data.loc[data['Fastest_Route_Name'] == 'Sufi Tabasum Rd and Service Rd W', 'Fastest_Route_Name'] = 47
       data.loc[data['Fastest_Route_Name'] == 'Tipu Sultan Rd', 'Fastest_Route_Name'] = 48
       data.loc[data['Fastest_Route_Name'] == 'nishing School', 'Fastest_Route_Name'] = 49
       data.loc[data['Fastest_Route_Name'] == 'I.J.P. Rd and I.J.P. Road', 'Fastest_Route_Name'] = 61
       data.loc[data['Fastest_Route_Name'] == 'Faisal Ave', 'Fastest_Route_Name'] = 62
       data.loc[data['Fastest_Route_Name'] == 'Aun Muhammad Rizvi Rd', 'Fastest_Route_Name'] = 63
       data.loc[data['Fastest_Route_Name'] == 'Faisal Ave and Faisal Ave/Islamabad Expressway', 'Fastest_Route_Name'] = 64
       data.loc[data['Fastest_Route_Name'] == 'Jinnah Avenue Underpass and Faisal Ave', 'Fastest_Route_Name'] = 65
       data.loc[data['Fastest_Route_Name'] == 'Service Rd E, Street 40 and Service Road South', 'Fastest_Route_Name'] = 66
       data.loc[data['Fastest_Route_Name'] == 'Service Road East', 'Fastest_Route_Name'] = 67
       data.loc[data['Fastest_Route_Name'] == 'Murree Rd and I.J.P. Road', 'Fastest_Route_Name'] = 68
       data.loc[data['Fastest_Route_Name'] == 'Street 54 and Service Rd E', 'Fastest_Route_Name'] = 69
       data.loc[data['Fastest_Route_Name'] == 'Sufi Tabasum Rd', 'Fastest_Route_Name'] = 70
       data.loc[data['Fastest_Route_Name'] == 'Faisal Ave/Islamabad Expressway', 'Fastest_Route_Name'] = 71
       data.loc[data['Fastest_Route_Name'] == 'Ataturk Ave', 'Fastest_Route_Name'] = 72
       data.loc[data['Fastest_Route_Name'] == 'Ataturk Ave and Constitution Ave', 'Fastest_Route_Name'] = 73
       data.loc[data['Fastest_Route_Name'] == 'Stadium Rd and Murree Rd', 'Fastest_Route_Name'] = 74
       data.loc[data['Fastest_Route_Name'] == 'Aiwan-e-Sanat-o-Tijarat and Kashmir Hwy', 'Fastest_Route_Name'] = 75
       data.loc[data['Fastest_Route_Name'] == 'Murree Rd/N-75 and Khayaban-e-Suhrwardy', 'Fastest_Route_Name'] = 76
       data.loc[data['Fastest_Route_Name'] == 'Service Rd W and Faisal Ave', 'Fastest_Route_Name'] = 77
       data.loc[data['Fastest_Route_Name'] == 'Club Rd', 'Fastest_Route_Name'] = 78
       data.loc[data['Fastest_Route_Name'] == 'Service Rd E and Kashmir Hwy', 'Fastest_Route_Name'] = 79
       data.loc[data['Fastest_Route_Name'] == 'Service Rd I 11 (South) and I.J.P. Road', 'Fastest_Route_Name'] = 80
       data.loc[data['Fastest_Route_Name'] ==  'Club Rd and Constitution Ave', 'Fastest_Route_Name'] = 81 
       data.loc[data['Fastest_Route_Name'] == 'Main Margalla Rd, 9th Ave and Kashmir Hwy', 'Fastest_Route_Name'] = 82
       data.loc[data['Fastest_Route_Name'] == 'Main Margalla Rd and 7th Ave', 'Fastest_Route_Name'] = 83
       data.loc[data['Fastest_Route_Name'] == 'Service Rd E and 9th Ave', 'Fastest_Route_Name'] = 84
       data.loc[data['Fastest_Route_Name'] == 'Shan-ul-Haq Haqqee Rd and Kashmir Hwy', 'Fastest_Route_Name'] = 85
       data.loc[data['Fastest_Route_Name'] == '9th Ave, Kashmir Hwy and 7th Ave', 'Fastest_Route_Name'] = 86
       data.loc[data['Fastest_Route_Name'] == 'City-Sadar Road', 'Fastest_Route_Name'] = 87
       data.loc[data['Fastest_Route_Name'] == 'Service Rd W and 9th Ave', 'Fastest_Route_Name'] = 88
       data.loc[data['Fastest_Route_Name'] == 'Stadium Rd', 'Fastest_Route_Name'] = 89
       data.loc[data['Fastest_Route_Name'] == 'Nazim-ud-din Rd and Jinnah Ave', 'Fastest_Route_Name'] = 90
       data.loc[data['Fastest_Route_Name'] == 'Ibn-e-Sina Rd and 9th Ave', 'Fastest_Route_Name'] = 91
       data.loc[data['Fastest_Route_Name'] == 'Aiwan-e-Sanat-o-Tijarat', 'Fastest_Route_Name'] = 92
       data.loc[data['Fastest_Route_Name'] == 'Sufi Tabasum Rd and 9th Ave', 'Fastest_Route_Name'] = 93
       data.loc[data['Fastest_Route_Name'] == 'Luqman Hakeem Rd and Jinnah Ave', 'Fastest_Route_Name'] = 94
       data.loc[data['Fastest_Route_Name'] == 'Service Rd South I 8 and Stadium Rd', 'Fastest_Route_Name'] = 95
       data.loc[data['Fastest_Route_Name'] == 'Service Rd North (VR-30) and Main Margalla Rd', 'Fastest_Route_Name'] = 96
       data.loc[data['Fastest_Route_Name'] == 'Liaqat Rd and Murree Rd', 'Fastest_Route_Name'] = 97
       data.loc[data['Fastest_Route_Name'] == 'Liaqat Rd', 'Fastest_Route_Name'] = 98
       data.loc[data['Fastest_Route_Name'] == 'AK Brohi Rd and Faqir Aipee Road', 'Fastest_Route_Name'] = 99
       data.loc[data['Fastest_Route_Name'] == 'Gomal Rd and Main Margalla Rd', 'Fastest_Route_Name'] = 100
       data.loc[data['Fastest_Route_Name'] == 'Service Rd E and Aun Muhammad Rizvi Rd', 'Fastest_Route_Name'] = 101
       data.loc[data['Fastest_Route_Name'] == 'Service Rd N', 'Fastest_Route_Name'] = 102
       data.loc[data['Fastest_Route_Name'] == 'Pitras Bukhari Rd', 'Fastest_Route_Name'] = 103
       data.loc[data['Fastest_Route_Name'] == 'Park Rd', 'Fastest_Route_Name'] = 104
       data.loc[data['Fastest_Route_Name'] == 'Street 40', 'Fastest_Route_Name'] = 105
       data.loc[data['Fastest_Route_Name'] == 'Jinnah Ave and Constitution Ave', 'Fastest_Route_Name'] = 106
       data.loc[data['Fastest_Route_Name'] == 'Garden Ave', 'Fastest_Route_Name'] = 107
       data.loc[data['Fastest_Route_Name'] == 'Service Rd E, Muhammad Tufail Niazi Rd and Service Road South', 'Fastest_Route_Name'] = 109
       data.loc[data['Fastest_Route_Name'] == 'Bela Rd and Service Road South', 'Fastest_Route_Name'] = 110
       data.loc[data['Fastest_Route_Name'] == 'AK Brohi Rd', 'Fastest_Route_Name'] = 111
       data.loc[data['Fastest_Route_Name'] == 'Service Road South and Aun Muhammad Rizvi Rd', 'Fastest_Route_Name'] = 112
       # replace Sys_Time
       # replace Sys_Time
       data['Sys_Time'] = data['Sys_Time'].str.replace(r':(.*)', '')

       # replace Date
       data['Date'] = data['Date'].str.replace(r'/0', '')
       data['Date'] = data['Date'].str.replace(r'/2019', '')
       data['Date'] = data['Date'].str.replace(r'/', '')


# loading and cleaning training data
data_train = pd.read_csv('C:/Users/Noreen/Desktop/New folder/Traffic Congestion Prediction/plos_sample_ google_dataset.csv')
clean_data(data_train)

# convert Fastest_Route_Time from seconds to minutes to converge them near to each other
data_train['Fastest_Route_Time'] = data_train['Fastest_Route_Time'] / 60

# seperate the data into features and target
# as the brackets [] take only one argument so we have to give one list with many column names as argument in X_train
X = data_train[['Date', 'Day', 'Sys_Time', 'Weather', 'Holiday', 'Special_Condition', 'Starting_Location',
          'Destination_Location', 'Fastest_Route_Name']]
y = data_train['Data_prediction']

y = y.replace('slightly congested',1 )
y = y.replace('smooth', 0)
y = y.replace('blockage', 4)
y = y.replace('congested', 2)
y = y.replace('highly congested', 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle='true')
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

#for learning_rate in lr_list:
gb_clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1, max_features=9, max_depth=2, random_state=42)
gb_clf.fit(X_train, y_train)
y_pred = gb_clf.predict(X_test).round()




y_score = gb_clf.predict_proba(X_test)

print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
# model initialization and fitting
# fit model no training data
# fit model no training data
values = y_test
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#precision, recall, thresholds = precision_recall_curve(y_test,y_pred)
## precision recall curve
precision = dict()
recall = dict()
for i in range(5):
    precision[i], recall[i], _ = precision_recall_curve(onehot_encoded[:, i],
                                                        y_score[:, i])
    plt.plot(recall[i], precision[i], lw=5, label='class {}'.format(i))

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("PR CURVE FOR GRADIENT BOOST")
plt.show()


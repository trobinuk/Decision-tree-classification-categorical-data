train_data_file_path = 'C:\\Users\\trobi\\Desktop\\Dr.Wu_Risk_Assesment_Project\\Week_9\\training_data_1207recs.csv'
test_data_file_path = 'C:\\Users\\trobi\\Downloads\\test_data.csv'
result_file_path = 'C:\\Users\\trobi\\Downloads\\NER_test_result_final.csv'

def enc_str_to_int(file_path):

    import pandas as pd
    import numpy as np

    #x = 'C:\\Users\\trobi\\Desktop\\Dr.Wu_Risk_Assesment_Project\\Week_9\\training_data_ml_final.csv'

    df = pd.read_csv(file_path)

    df = df.rename(columns={0: "before_label_1", 1: "before_label_2",
                            2: "after_label_1", 3: "after_label_2", 4:"old_entities"})
    #print('The len of df is ',str(len(df.head(5))))
    #print(df.head(5))

    if len(df.columns) == 5:
        dict_val = {'before_label_1':'bef_lab_1','before_label_2':'bef_lab_2','after_label_1':'aft_lab_1',
                    'after_label_2':'aft_lab_2','old_entities':'old_ent'}
    else:
        dict_val = {'before_label_1':'bef_lab_1','before_label_2':'bef_lab_2','after_label_1':'aft_lab_1',
                    'after_label_2':'aft_lab_2','old_entities':'old_ent','new_entities':'new_ent'}

    for key in dict_val:
        df[dict_val[key]] = np.where(df[key] == 'PERSON', '1',
                    np.where(df[key] == 'NORP', '2',
                    np.where(df[key] == 'FAC', '3',
                    np.where(df[key] == 'ORG', '4',
                    np.where(df[key] == 'GPE', '5',
                    np.where(df[key] == 'LOC', '6',
                    np.where(df[key] == 'PRODUCT', '7',
                    np.where(df[key] == 'EVENT', '8',
                    np.where(df[key] == 'WORK OF ART', '9',
                    np.where(df[key] == 'LAW', '10',
                    np.where(df[key] == 'LANGUAGE', '11',
                    np.where(df[key] == 'DATE', '12', 
                    np.where(df[key] == 'TIME', '13',
                    np.where(df[key] == 'PERCENT', '14',
                    np.where(df[key] == 'MONEY', '15',
                    np.where(df[key] == 'QUANTITY', '16',
                    np.where(df[key] == 'ORDINAL', '17',
                    np.where(df[key] == 'CARDINAL', '18', '0')))))))))))))))))) 

    if len(df.columns) == 10:
        return_colns = ['bef_lab_1', 'bef_lab_2', 'aft_lab_1', 'aft_lab_2', 'old_ent']
    else:
        return_colns = ['bef_lab_1', 'bef_lab_2', 'aft_lab_1', 'aft_lab_2', 'old_ent','new_ent']
    
    return df[return_colns]

def dec_int_to_str(df):

    import pandas as pd
    import numpy as np

    dict_val = {'bef_lab_1':'before_label_1','bef_lab_2':'before_label_2','aft_lab_1':'after_label_1',
                'aft_lab_2':'after_label_2','old_ent':'old_entities','result':'new_entities'}

    for key in dict_val:
        df[dict_val[key]] = np.where(df[key] ==  '1','PERSON',
                   np.where(df[key] ==  '2','NORP',
                   np.where(df[key] ==  '3','FAC',
                   np.where(df[key] ==  '4','ORG',
                   np.where(df[key] ==  '5','GPE',
                   np.where(df[key] ==  '6','LOC',
                   np.where(df[key] ==  '7','PRODUCT', 
                   np.where(df[key] ==  '8','EVENT',
                   np.where(df[key] ==  '9','WORK OF ART',
                   np.where(df[key] ==  '10','LAW',
                   np.where(df[key] ==  '11','LANGUAGE', 
                   np.where(df[key] ==  '12', 'DATE',
                   np.where(df[key] ==  '13','TIME',
                   np.where(df[key] ==  '14','PERCENT', 
                   np.where(df[key] ==  '15','MONEY',
                   np.where(df[key] ==  '16','QUANTITY', 
                   np.where(df[key] ==  '17','ORDINAL',
                   np.where(df[key] ==  '18','CARDINAL', 'NA'))))))))))))))))))


    return_colns = ['before_label_1','before_label_2','after_label_1',
                    'after_label_2','old_entities','new_entities']

    return df[return_colns]
    
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics 
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt

feature_cols = ['bef_lab_1', 'bef_lab_2', 'aft_lab_1', 'aft_lab_2', 'old_ent']
training_data = enc_str_to_int(train_data_file_path)
training_data[feature_cols] = training_data[feature_cols].astype('category')    
training_data['bef_lab_2'] = training_data['bef_lab_2'].astype('category')             
print(training_data.tail(4))
print('The 1st one is ', type(training_data['bef_lab_1']))
print('The 2nd one is ', type(training_data['bef_lab_2']))

X = training_data[feature_cols] # Features
y = training_data.new_ent # Target variable

X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True) # 70% training and 30% test
print('Successfully splitted the data...')

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train_data,y_train_data)
print('Decision Tree Classifer is Successfully completed...')

#Predict the response for test dataset
y_pred = clf.predict(X_test_data)
print("Accuracy: ",metrics.accuracy_score(y_test_data, y_pred))

#tree.plot_tree(clf)
#plt.show()
#print(y_test_data)
print(y_pred)
'''
import pickle
 
# Save the trained model as a pickle string.
saved_model = pickle.dumps(clf)
 
# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)


ner_test_data = enc_str_to_int(test_data_file_path)
print(ner_test_data.head(5))

feature_cols = ['bef_lab_1', 'bef_lab_2', 'aft_lab_1', 'aft_lab_2', 'old_ent']
ner_test_data = ner_test_data[feature_cols]
#print(ner_test_data.head(5))

print('Result of the Test data...')
print(clf_from_pickle.predict(ner_test_data))

ner_test_data['result'] = clf_from_pickle.predict(ner_test_data)

print(len(ner_test_data))
print(ner_test_data.head(5))
final_result_df = dec_int_to_str(ner_test_data)

final_result_df.to_csv(result_file_path,index=False)
'''

from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from utils import make_unsup, cooks
import numpy as np
#### Constants
random_state = 41

cv = StratifiedKFold(5, random_state=random_state)

targets = ['miscatch_P3a_novel', 'miscatch_P3b_target']

input_path = r"S:\Data_Science\Core\FDA_submission_10_2019\08-Reports\STAR_reports\Labeling_project\kaggle_miscatches\both"

Features = ['P3a_Delta_Novel_similarity_spatial',
            'P3a_Delta_Novel_similarity_locationLR',
            'P3a_Delta_Novel_similarity_locationPA',
            'P3a_Delta_Novel_similarity_timing',
            'P3a_Delta_Novel_similarity_amplitude',
            'P3a_Delta_Novel_matchScore',
            'P3a_Delta_Novel_attr_timeMSfromTriger',
            'P3a_Delta_Novel_attr_leftRight',
            'P3a_Delta_Novel_attr_posteriorAnterior',
            'P3a_Delta_Novel_attr_amplitude',
            'P3a_Delta_Novel_topo_topographicCorrCoeffAligned',
            'P3a_Delta_Novel_topo_topographicSimilarity',
            'P3b_Delta_Target_similarity_spatial',
            'P3b_Delta_Target_similarity_locationLR',
            'P3b_Delta_Target_similarity_locationPA',
            'P3b_Delta_Target_similarity_timing',
            'P3b_Delta_Target_similarity_amplitude',
            'P3b_Delta_Target_matchScore',
            'P3b_Delta_Target_attr_timeMSfromTriger',
            'P3b_Delta_Target_attr_leftRight',
            'P3b_Delta_Target_attr_posteriorAnterior',
            'P3b_Delta_Target_attr_amplitude',
            'P3b_Delta_Target_topo_topographicCorrCoeffAligned',
            'P3b_Delta_Target_topo_topographicSimilarity']


### load data
df = pd.read_csv(r"C:\Users\nogag\aob-miscatch-detection\data\non_dq_data.csv")

if 0:
    ## remove DQ list from Eran
    dq = pd.read_csv(r"DQ\AOB_Novel_remove.csv")
    dq = pd.read_csv(r"DQ\AOB_Target_remove.csv")
    df = df[~df['taskData._id.$oid'].isin(dq['taskData.elm_id'])]


### dropna or iterpolate
if 0:
    df = df.dropna(subset=features+["miscatch_P3a_novel", "miscatch_P3b_target"])
    df = df.dropna(subset=Features)



## create target variable
targets = ['miscatch_P3a_novel', 'miscatch_P3b_target', 'miscatches']
df['miscatches'] = np.ceil((df["miscatch_P3a_novel"] + df["miscatch_P3b_target"]) / 2)


unsupervised_df = []
for age in df['agebin'].unique():
    agedf = df[df['agebin'] == age]
    if 0:
        ss = StandardScaler()
        ss.fit(df[df['reference_agebin'].str.contains(age)][Features])
        agedf[Features] = ss.transform(agedf[Features])
    agedf = make_unsup(agedf, df[df['reference_agebin'].str.contains(age)], Features, random_state=random_state) # replace Feature to subset if you what

    ## regression error anomaly detection
    agedf = cooks(agedf, df[df['reference_agebin'].str.contains(age)], 'P3a_Delta_Novel_matchScore', ['P3a_Delta_Novel_similarity_locationPA', 'P3a_Delta_Novel_similarity_timing',
            'P3a_Delta_Novel_similarity_amplitude',], 0)
    agedf = cooks(agedf, df[df['reference_agebin'].str.contains(age)], 'P3a_Delta_Novel_similarity_locationPA', ['P3a_Delta_Novel_attr_timeMSfromTriger',
            'P3a_Delta_Novel_similarity_amplitude'], 1)

    unsupervised_df.append(agedf)

features = Features +[ 'osvm', 'isof', 'gmm2_0', 'gmm2_1', 'gmm3_0', 'gmm3_1', 'gmm3_2', 'gmm4_0', 'gmm4_1', 'gmm4_2','gmm4_3',  'cooks_d0', 'cooks_d1']

df = pd.concat(unsupervised_df)#.dropna(subset=features)


## grouping over visits
df = df[df.visit == 1]
df_vis1 = df[df.visit == 1]

for train_index, test_index in cv.split(df_vis1[Features + targets], df_vis1[targets[2]]):
    df_vis1_train, df_vis1_test = df_vis1.iloc[train_index], df_vis1.iloc[test_index]

    # these are you sklearn-like sets
    Xt = pd.concat([df_vis1_train[Features]])
    Xv = pd.concat([df_vis1_test[Features]])
    yt = pd.concat([df_vis1_train[targets]])
    yv = pd.concat([df_vis1_test[targets]])

    # 3 clfs
    target_out0 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8, n_estimators=1000, random_state=random_state).fit(Xt, yt[targets[0]],  early_stopping_rounds=10).predict_proba(Xv)[:, 0]
    target_out1 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8, n_estimators=1000, random_state=random_state).fit(Xt, yt[targets[1]],  early_stopping_rounds=10).predict_proba(Xv)[:, 0]
    target_out2 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8, n_estimators=1000, random_state=random_state).fit(Xt, yt[targets[2]],  early_stopping_rounds=10).predict_proba(Xv)[:, 0]

    # single predict
    target_out = target_out0 * target_out1 * target_out2

    print('auc',  roc_auc_score(yv['target'], 1 - target_out))
    print('recall', recall_score(yv['target'], target_out < 0.65))
    print('precision', precision_score(yv['target'], target_out < 0.65))
    print('confusion martix\n', confusion_matrix(yv['target'], target_out < 0.65), '\n')
















### test set



X_test = pd.read_csv(os.path.join(input_path, "X_test.csv"))
y_test = pd.read_csv(os.path.join(input_path, "y_test.csv"))


Features = ['P3a_Delta_Novel_similarity_spatial',
            'P3a_Delta_Novel_similarity_locationLR',
            'P3a_Delta_Novel_similarity_locationPA',
            'P3a_Delta_Novel_similarity_timing',
            'P3a_Delta_Novel_similarity_amplitude',
            'P3a_Delta_Novel_matchScore',
            'P3a_Delta_Novel_attr_timeMSfromTriger',
            'P3a_Delta_Novel_attr_leftRight',
            'P3a_Delta_Novel_attr_posteriorAnterior',
            'P3a_Delta_Novel_attr_amplitude',
            'P3a_Delta_Novel_topo_topographicCorrCoeffAligned',
            'P3a_Delta_Novel_topo_topographicSimilarity',
            'P3b_Delta_Target_similarity_spatial',
            'P3b_Delta_Target_similarity_locationLR',
            'P3b_Delta_Target_similarity_locationPA',
            'P3b_Delta_Target_similarity_timing',
            'P3b_Delta_Target_similarity_amplitude',
            'P3b_Delta_Target_matchScore',
            'P3b_Delta_Target_attr_timeMSfromTriger',
             'P3b_Delta_Target_attr_leftRight',
             'P3b_Delta_Target_attr_posteriorAnterior',
             'P3b_Delta_Target_attr_amplitude',
             'P3b_Delta_Target_topo_topographicCorrCoeffAligned',
             'P3b_Delta_Target_topo_topographicSimilarity']


# splitting to agebins should be done here
df_test = pd.merge(y_test, X_test, on='taskData._id.$oid')

df_test = df_test[(df_test['agebin'] == "aob_25-39") | (df_test['agebin'] == "aob_35-50")]

df_test = df_test.dropna(subset=Features)
df_test=df_test[df_test.visit==1]
ss = StandardScaler()
all_data = pd.DataFrame(ss.fit_transform(dq_df[Features]), columns=Features)
df_test[Features] = ss.transform(df_test[Features])
all_data['taskData._id.$oid'] = dq_df['taskData.elm_id'].values
all_data['agebin'] = dq_df['agebin'].values
all_data['visit'] = dq_df['visit'].values
unsupervised_df = []
for age in df_test['agebin'].unique():
    agedf = df_test[df_test['agebin'] == age]
    agedf = make_unsup(agedf, all_data[all_data['agebin'] == age], Features)
    # IQR & Z score
    unsupervised_df.append(agedf)

df_test = pd.concat(unsupervised_df).dropna(subset=Features)
# #
print("test set")
Features = Features + ['moc', 'osvm', 'isof', 'gmm2_0', 'gmm2_1', 'gmm3_0', 'gmm3_1', 'gmm3_2', 'gmm4_0', 'gmm4_1', 'gmm4_2', 'gmm4_3', 'iqr', 'zscore']
df_test = df_test.dropna(subset=Features)
df_test['target'] = df_test[targets[0]] | df_test[targets[1]]

target_out0 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8, n_estimators=1000).fit(df[Features], df[targets[0]], early_stopping_rounds=10).predict_proba(df_test[Features])[:, 0]
target_out1 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8, n_estimators=1000).fit(df[Features], df[targets[1]], early_stopping_rounds=10).predict_proba(df_test[Features])[:, 0]
target_out2 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8, n_estimators=1000).fit(df[Features], df['target'], early_stopping_rounds=10).predict_proba(df_test[Features])[:, 0]

target_out = target_out0 * target_out1 * target_out2
print('auc',  roc_auc_score(df_test['target'], 1 - target_out))
print('recall', recall_score(df_test['target'], target_out < 0.65))
print('precision', precision_score(df_test['target'], target_out < 0.65))
print('confusion martix\n', confusion_matrix(df_test['target'], target_out < 0.65), '\n')

# clf = CatBoostClassifier(verbose=0, class_weights=[1, 5], depth=4).fit(df[features], df['target'])
# for i, j in sorted(zip(clf.feature_importances_, features), reverse=1):
#     print(i, j)


### real
print("real pred")
df = pd.concat([df, df_test])

Features = ['P3a_Delta_Novel_similarity_spatial',
            'P3a_Delta_Novel_similarity_locationLR',
            'P3a_Delta_Novel_similarity_locationPA',
            'P3a_Delta_Novel_similarity_timing',
            'P3a_Delta_Novel_similarity_amplitude',
            'P3a_Delta_Novel_matchScore',
            'P3a_Delta_Novel_attr_timeMSfromTriger',
            'P3a_Delta_Novel_attr_leftRight',
            'P3a_Delta_Novel_attr_posteriorAnterior',
            'P3a_Delta_Novel_attr_amplitude',
            'P3a_Delta_Novel_topo_topographicCorrCoeffAligned',
            'P3a_Delta_Novel_topo_topographicSimilarity',
            'P3b_Delta_Target_similarity_spatial',
            'P3b_Delta_Target_similarity_locationLR',
            'P3b_Delta_Target_similarity_locationPA',
            'P3b_Delta_Target_similarity_timing',
            'P3b_Delta_Target_similarity_amplitude',
            'P3b_Delta_Target_matchScore',
            'P3b_Delta_Target_attr_timeMSfromTriger',
             'P3b_Delta_Target_attr_leftRight',
             'P3b_Delta_Target_attr_posteriorAnterior',
             'P3b_Delta_Target_attr_amplitude',
             'P3b_Delta_Target_topo_topographicCorrCoeffAligned',
             'P3b_Delta_Target_topo_topographicSimilarity']

X_pred = all_data[~all_data['taskData._id.$oid'].isin(df['taskData._id.$oid'])]

unsupervised_df = []
for age in X_pred['agebin'].unique():
    agedf = X_pred[X_pred['agebin'] == age]
    agedf = make_unsup(X_pred, all_data[all_data['agebin'] == age], Features)
    # IQR & Z score
    unsupervised_df.append(agedf)
Features = Features + ['moc', 'osvm', 'isof', 'gmm2_0', 'gmm2_1', 'gmm3_0', 'gmm3_1', 'gmm3_2', 'gmm4_0', 'gmm4_1', 'gmm4_2', 'gmm4_3', 'iqr', 'zscore']

X_pred = pd.concat(unsupervised_df).dropna(subset=Features)
X_pred = X_pred[X_pred.visit==1]
target_out0 = CatBoostClassifier(verbose=0, class_weights=[1,10], depth=8, n_estimators=1000).fit(df[Features], df[targets[0]], early_stopping_rounds=10).predict_proba(X_pred[Features])[:, 0]
target_out1 = CatBoostClassifier(verbose=0, class_weights=[1, 10], depth=8, n_estimators=1000).fit(df[Features], df[targets[1]], early_stopping_rounds=10).predict_proba(X_pred[Features])[:, 0]
target_out2 = CatBoostClassifier(verbose=0, class_weights=[1,10], depth=8, n_estimators=1000).fit(df[Features], df['target'], early_stopping_rounds=10).predict_proba(X_pred[Features])[:, 0]

target_out = target_out0 * target_out1 * target_out2
print(X_pred['taskData._id.$oid'])
print(X_pred[target_out<0.65]['taskData._id.$oid'])
X_pred[target_out<0.65]['taskData._id.$oid'].to_csv(r'S:\Data_Science\Core\FDA_submission_10_2019\08-Reports\STAR_reports\Labeling_project\resource_files\miscatches_detected\aob_25-50\delegated_reports.csv')

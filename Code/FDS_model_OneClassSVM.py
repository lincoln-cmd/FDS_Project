'''
 모델 학습 - 모델 선택
'''

# 필요 라이브러리 및 데이터셋 로드

import pandas as pd
import time

start = time.time()

x_train = pd.read_csv('./x_train.csv')
y_train = pd.read_csv('./y_train.csv')
x_val = pd.read_csv('./x_val.csv')
y_val = pd.read_csv('./y_val.csv')
x_test = pd.read_csv('./x_test.csv')
y_test = pd.read_csv('./y_test.csv')


print('x_train: ', x_train.head())
print('y_train: ', y_train.head())
print('x_val: ', x_val.head())
print('y_val: ', y_val.head())
print('x_test: ', x_test.head())
print('y_test: ', y_test.head())

from sklearn.svm import OneClassSVM


# 문자열을 포함한 컬럼 제거 (해당 컬럼들은 현재 시나리오인 위도/경도 위치 기반 이상 거래 탐지에 불필요한 컬럼으로 판단)
x_train = x_train.drop(['merchant', 'category', 'first', 'last', 'city'], axis = 1)
x_val = x_val.drop(['merchant', 'category', 'first', 'last', 'city'], axis = 1)
x_test = x_test.drop(['merchant', 'category', 'first', 'last', 'city'], axis = 1)





'''
 모델 학습 - 모델 초기화
'''

model = OneClassSVM(nu = 0.1, kernel = 'rbf', gamma = 0.1)

model.fit(x_train)

end1 = time.tme()
print(f'모델 학습까지 걸린 시간: {end1 - start}')



y_pred = model.predict(x_val)

y_pred = [1 if pred == -1 else 0 for pred in y_pred]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_val['is_fraud'], y_pred)
precision = precision_score(y_val['is_fraud'], y_pred)
recall = recall_score(y_val['is_fraud'], y_pred)
f1 = f1_score(y_val['is_fraud'], y_pred)
roc_auc = roc_auc_score(y_val['is_fraud'], y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'ROC-AUC: {roc_auc}')

end2 = time.time()
print(f'모델 검증까지 소요 시간: {end2 - start}')
print(f'모델 검증 자체 소요 시간: {end2 - end1}')


from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_distributions = {
    'nu': np.linspace(0.01, 0.2, 20),
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': np.logspace(-3, 2, 6)
}

random_search = RandomizedSearchCV(
    estimator = OneClassSVM(),
    param_distributions = param_distributions,
    n_iter = 10,
    cv = 5,
    scoring = 'accuracy',
    random_state = 42
)

random_search.fit(x_train)

print('Best Hyperparameters: ', random_search.best_params_)

best_model = random_search.best_estimator_

end3 = time.time()
print(f'하이퍼파라미터 튜닝까지 소요 시간: {end3 - start}')
print(f'하이퍼파라미터 자체 소요 시간: {end3 - end2}')


y_pred = best_model.predict(x_test)

y_pred = [1 if pred == -1 else 0 for pred in y_pred]

accuracy = accuracy_score(y_test['is_fraud'], y_pred)
precision = precision_score(y_test['is_fraud'], y_pred)
recall = recall_score(y_test['is_fraud'], y_pred)
f1 = f1_score(y_test['is_fraud'], y_pred)
roc_auc = roc_auc_score(y_test['is_fruad'], y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'ROC-AUC: {roc_auc}')

end4 = time.time()
print(f'모델 테스트까지 소요 시간: {end4 - start}')
print(f'모델 테스트 자체 소요 시간: {end4 - end3}')
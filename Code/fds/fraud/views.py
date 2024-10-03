from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
#from .serializers import FraudDetectionSerializer
#from .models import FraudData
from sklearn.ensemble import IsolationForest # 모델 import

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import pickle
import pandas as pd
import os
from django.conf import settings
import numpy as np

from .models import Transaction

import numpy as np
import pandas as pd
import random
from django.shortcuts import render
import pickle
from .models import Transaction

import numpy as np
import pandas as pd
import random
from django.shortcuts import render
import pickle
from .models import Transaction
from sklearn.preprocessing import LabelEncoder

from django.contrib.auth.decorators import login_required




import numpy as np
import random
from fraud.models import Transaction
from sklearn.preprocessing import LabelEncoder
import pickle
from django.shortcuts import render
from django.http import HttpResponse

from django.db import connections, OperationalError, connection

def index(request):
    return render(request, 'fraud/index.html')  # 메인 페이지 렌더링

@login_required
def test_fraud(request):
    # 모델 파일 경로
    model_path = 'fraud/models/FDS_model_isolationForest.pkl'

    # 모델 로드
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return HttpResponse("Model file not found.")

    # DB에서 모든 트랜잭션 데이터 로드
    transactions = Transaction.objects.all()
    
    if not transactions:
        return HttpResponse("No transaction data found.")

    # 레이블 인코더 초기화
    label_encoders = {
        'customer_id': LabelEncoder(),
        'merchant_id': LabelEncoder()
    }

    # 예측할 데이터 준비
    predictions = []
    
    customer_ids = [transaction.customer_id for transaction in transactions]
    merchant_ids = [transaction.merchant_id for transaction in transactions]

    label_encoders['customer_id'].fit(customer_ids)
    label_encoders['merchant_id'].fit(merchant_ids)

    for transaction in transactions:
        # 필요한 특성만 추출
        transaction_data = np.array([
            0,  # transaction_id는 사용하지 않거나 0으로 설정
            transaction.amount, 
            label_encoders['customer_id'].transform([transaction.customer_id])[0], 
            label_encoders['merchant_id'].transform([transaction.merchant_id])[0], 
            transaction.lat, 
            transaction.long
        ])

        random_features = [random.uniform(0, 1) for _ in range(17)]
        complete_data = np.concatenate([transaction_data, random_features])

        prediction = model.predict(complete_data.reshape(1, -1))
        predictions.append(prediction[0])

        # 예측 결과를 DB에 저장
        transaction.fraud_prediction = prediction[0]
        transaction.save()

        complete_data = np.concatenate([transaction_data, random_features, prediction])

        

    context = {
        'predictions': predictions,
    }

    return render(request, 'test_fraud.html', context)


import pickle
from django.conf import settings

# 모델을 로드하는 함수
def load_model():
    model_path = 'fraud/models/FDS_model_isolationForest.pkl'  # 모델 경로
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model



import pandas as pd
import numpy as np
import random
from elasticsearch import Elasticsearch
from django.shortcuts import render, redirect
from .models import Transaction
from urllib.parse import quote
import requests
from datetime import datetime

# Elasticsearch 연결 설정
es = Elasticsearch([{'host': '10.0.1.7', 'port': 9200}])  # DB 서버의 Elasticsearch IP와 포트 사용


import requests
import json
from django.shortcuts import render, redirect
from django.conf import settings
from django.db import connection, connections
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Elasticsearch 연결 설정
es = Elasticsearch([{'host': '10.0.1.7', 'port': 9200}])  # DB 서버의 Elasticsearch IP와 포트 사용


# 인덱스 자동 생성 함수
def create_user_index(es, index_name):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body={
            "mappings": {
                "properties": {
                    "amount": {"type": "float"},
                    "customer_id": {"type": "float"},
                    "merchant_id": {"type": "float"},
                    "lat": {"type": "float"},
                    "long": {"type": "float"},
                    "fraud_prediction": {"type": "long"},
                    "timestamp": {"type": "date"},
                    "user_id": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    }
                }
            }
        })
        print(f"인덱스 {index_name}가 성공적으로 생성되었습니다.")




@login_required
def upload_data(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        df = pd.read_csv(uploaded_file)
        
        # 업로드된 데이터를 세션에 저장
        request.session['uploaded_data'] = df.to_json()

        # 데이터 처리 후 dashboard2.html로 리다이렉트
        return redirect('dashboard2')
    
    return render(request, 'fraud/upload_form.html')



import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse

import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse

import numpy as np
import pandas as pd

@login_required
def dashboard2(request):
    # 세션에서 데이터를 로드
    uploaded_data = request.session.get('uploaded_data')

    if uploaded_data:
        df = pd.read_json(uploaded_data)

        # IsolationForest 모델 로드
        model = load_model()

        # 주요 특성 (학습에 사용한 주요 특성)
        main_features = ['Amount', 'Customer ID', 'Merchant ID', 'Lat', 'Long']
        
        # 모델이 학습된 모든 특성을 가져옴
        model_features = model.feature_names_in_

        # 입력 데이터를 학습에 사용된 주요 특성 이름으로 변경
        df = df.rename(columns={
            'Amount': 'amt',
            'Customer ID': 'customer_id',
            'Merchant ID': 'merchant_id',
            'Lat': 'lat',
            'Long': 'long'
        })

        # 주요 특성에 대해서는 입력 데이터를 사용하고, 나머지 특성에 대해서는 랜덤 값을 채움
        for feature in model_features:
            if feature not in df.columns:
                df[feature] = np.random.rand(len(df))  # 랜덤 값으로 대체

        # 모델 입력을 위한 데이터 준비 (모델에 맞게 모든 학습된 특성을 포함)
        X = df[model_features]

        # 예측 수행
        fraud_prediction = model.predict(X)

        # 예측 결과를 데이터프레임에 추가
        df['Fraud Prediction'] = fraud_prediction

        # 이상 거래 탐지
        df['Transaction Type'] = np.where(df['Fraud Prediction'] == -1, 'Fraud', 'Normal')  # -1: 이상 거래, 1: 정상 거래

        lat = df['lat']
        long = df['long']
        fraud_prediction = df['Fraud Prediction']

        



        # 시각화: 정상 거래는 파란색, 이상 거래는 빨간색으로 표시
        plt.figure(figsize=(10, 6))
        colors = np.where(df['Fraud Prediction'] == 1, 'red', 'lavender')  # 이상 거래는 빨간색, 정상 거래는 파란색
        plt.scatter(long, lat, c=colors, s=100)
        plt.colorbar(label='Fraud Prediction')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Fraud Prediction Map')

        

        # 그래프를 이미지로 변환하여 HTML에 전달
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png).decode('utf-8')


        # 이상 거래만 추출하여 HTML 테이블에 전달
        fraud_transactions = df[df['Fraud Prediction'] == 1][['amt', 'customer_id', 'merchant_id', 'lat', 'long']]


        # 다운로드 요청 처리 (기능 추가)
        if 'download' in request.GET:
            response = HttpResponse(image_png, content_type='image/png')
            response['Content-Disposition'] = 'attachment; filename="fraud_prediction_map.png"'
            return response

        elif 'download_csv' in request.GET:
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="fraud_predictions.csv"'
            fraud_transactions.to_csv(path_or_buf=response, index=False)
            return response



            

        

        return render(request, 'fraud/dashboard2.html', {
            'graphic': graphic,
            'fraud_transactions': fraud_transactions.to_html(classes='table table-striped', index=False)
        })
    else:
        return render(request, 'fraud/dashboard2.html', {'error': '데이터가 없습니다.'})



@login_required
def upload_form2(request):
    if request.method == 'POST':
        # 업로드된 CSV 파일 처리
        uploaded_file = request.FILES['file']
        df = pd.read_csv(uploaded_file)

        # 사용자 데이터베이스 선택 및 설정
        db_name = f'user_{request.user.username}_db'
        create_db_sql = f"CREATE DATABASE IF NOT EXISTS `{db_name}`"

        if db_name not in settings.DATABASES:
            settings.DATABASES[db_name] = {
                'ENGINE': 'django.db.backends.mysql',
                'NAME': db_name,
                'USER': 'fds001',
                'PASSWORD': 'fds001',
                'HOST': '10.0.1.7',
                'PORT': '3306',
            }

        try:
            # MySQL 연결 및 데이터 삽입
            with connection.cursor() as cursor:
                cursor.execute(create_db_sql)

            # 사용자 데이터베이스 작업
            with connections[db_name].cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS fraud_transaction (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        amount DECIMAL(10, 2),
                        customer_id VARCHAR(255),
                        merchant_id VARCHAR(255),
                        lat DECIMAL(10, 8),
                        `long` DECIMAL(11, 8),
                        fraud_prediction DECIMAL(10, 8)
                    )
                ''')

                index_name = f'user_{request.user.username}_transactions'
                customer_id_encoder = LabelEncoder()
                merchant_id_encoder = LabelEncoder()

                df['Customer ID'] = customer_id_encoder.fit_transform(df['Customer ID'])
                df['Merchant ID'] = merchant_id_encoder.fit_transform(df['Merchant ID'])

                model = load_model()

                for _, row in df.iterrows():
                    try:
                        if not all(col in row for col in ['Amount', 'Customer ID', 'Merchant ID', 'Lat', 'Long']):
                            return HttpResponse("CSV 파일의 형식이 잘못되었습니다. 필수 열이 누락되었습니다.")
                    except Exception as e:
                        return HttpResponse(f"데이터 처리 중 오류 발생: {str(e)}")

                    prediction_input = np.concatenate([np.array([row['Amount'], row['Customer ID'], row['Merchant ID'], row['Lat'], row['Long']]), random_features()])
                    prediction = model.predict(prediction_input.reshape(1, -1))[0]

                    cursor.execute('''
                        INSERT INTO fraud_transaction (amount, customer_id, merchant_id, lat, `long`, fraud_prediction)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    ''', (row['Amount'], row['Customer ID'], row['Merchant ID'], row['Lat'], row['Long'], prediction))

                    try:
                        es.index(index=index_name, body={
                            'amount': row['Amount'],
                            'customer_id': row['Customer ID'],
                            'merchant_id': row['Merchant ID'],
                            'lat': row['Lat'],
                            'long': row['Long'],
                            'fraud_prediction': prediction,
                            'user_id': request.user.username,
                            'timestamp': datetime.now(),  # Kibana용 타임스탬프 추가
                        })
                        print(f"Successfully indexed transaction to {index_name}")
                    except Exception as e:
                        print(f"Failed to index data in Elasticsearch: {str(e)}")

            # 데이터를 세션에 저장
            request.session['uploaded_data'] = df.to_json()

            # dashboard2로 리다이렉트
            return redirect('dashboard2')

        except OperationalError as e:
            return HttpResponse(f"Database connection for {db_name} failed: {str(e)}")

    return render(request, 'fraud/upload_form2.html')




"""

@login_required
def redirect_to_dashboard(request):
    user_id = request.user.username  # 로그인한 사용자 ID 가져오기
    kibana_url = f'http://211.188.51.137:5601/app/dashboards#/view/78e0ff70-7b56-11ef-9b66-13538d49ff90?_a=(filters:!((meta:(alias:!n,disabled:!f,key:user_id,negate:!f,params:(query:\'{user_id}\'),type:phrase))))'
    return redirect(kibana_url)

"""

"""

@login_required
def upload_form(request):
    if request.method == 'POST':
        # 업로드된 CSV 파일 처리
        uploaded_file = request.FILES['file']
        df = pd.read_csv(uploaded_file)

        # 사용자 데이터베이스 선택 및 설정
        db_name = f'user_{request.user.username}_db'
        create_db_sql = f"CREATE DATABASE IF NOT EXISTS `{db_name}`"

        if db_name not in settings.DATABASES:
            settings.DATABASES[db_name] = {
                'ENGINE': 'django.db.backends.mysql',
                'NAME': db_name,
                'USER': 'fds001',
                'PASSWORD': 'fds001',
                'HOST': '10.0.1.7',
                'PORT': '3306',
            }

        try:
            # MySQL 연결 및 데이터 삽입
            with connection.cursor() as cursor:
                cursor.execute(create_db_sql)

            # 사용자 데이터베이스 작업
            with connections[db_name].cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS fraud_transaction (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        amount DECIMAL(10, 2),
                        customer_id VARCHAR(255),
                        merchant_id VARCHAR(255),
                        lat DECIMAL(10, 8),
                        `long` DECIMAL(11, 8),
                        fraud_prediction DECIMAL(10, 8)
                    )
                ''')

                index_name = f'user_{request.user.username}_transactions'
                customer_id_encoder = LabelEncoder()
                merchant_id_encoder = LabelEncoder()

                df['Customer ID'] = customer_id_encoder.fit_transform(df['Customer ID'])
                df['Merchant ID'] = merchant_id_encoder.fit_transform(df['Merchant ID'])

                model = load_model()

                for _, row in df.iterrows():
                    try:
                        if not all(col in row for col in ['Amount', 'Customer ID', 'Merchant ID', 'Lat', 'Long']):
                            return HttpResponse("CSV 파일의 형식이 잘못되었습니다. 필수 열이 누락되었습니다.")
                    except Exception as e:
                        return HttpResponse(f"데이터 처리 중 오류 발생: {str(e)}")

                    prediction_input = np.concatenate([np.array([row['Amount'], row['Customer ID'], row['Merchant ID'], row['Lat'], row['Long']]), random_features()])
                    prediction = model.predict(prediction_input.reshape(1, -1))[0]

                    cursor.execute('''
                        INSERT INTO fraud_transaction (amount, customer_id, merchant_id, lat, `long`, fraud_prediction)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    ''', (row['Amount'], row['Customer ID'], row['Merchant ID'], row['Lat'], row['Long'], prediction))

                    try:
                        es.index(index=index_name, body={
                            'amount': row['Amount'],
                            'customer_id': row['Customer ID'],
                            'merchant_id': row['Merchant ID'],
                            'lat': row['Lat'],
                            'long': row['Long'],
                            'fraud_prediction': prediction,
                            'user_id': request.user.username,
                            'timestamp': datetime.now(),  # Kibana용 타임스탬프 추가
                        })
                        print(f"Successfully indexed transaction to {index_name}")
                    except Exception as e:
                        print(f"Failed to index data in Elasticsearch: {str(e)}")

            # Elasticsearch에 인덱스 패턴 생성
            create_kibana_index_pattern(index_name)

            # Kibana 대시보드 생성
            dashboard_id = create_kibana_dashboard(request.user.username)

            # Kibana 대시보드 URL 생성 및 템플릿으로 전달
            kibana_url = f"http://211.188.51.137:5601/app/dashboards#/view/{dashboard_id}?_a=(query:(language:kuery,query:'user_id:{request.user.username}'))"
            return render(request, 'fraud/dashboard.html', {'kibana_url': kibana_url})

        except OperationalError as e:
            return HttpResponse(f"Database connection for {db_name} failed: {str(e)}")

    return render(request, 'fraud/upload_form.html')

"""


# Kibana에서 개별 대시보드를 자동으로 생성하고 시각화요소를 동적으로 추가하는 기능은 유료 기능
# 대신 하나의 대시보드를 공유하고 URL 필터링을 통해 사용자별로 데이터를 구분하여 띄워주는 방식 사용
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from django.db import connection, connections, OperationalError
from datetime import datetime
from elasticsearch import Elasticsearch

# Elasticsearch 연결 설정
es = Elasticsearch([{'host': '10.0.1.7', 'port': 9200}])

# 모델 로드 함수
def load_model():
    model_path = 'fraud/models/FDS_model_isolationForest.pkl'
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# 랜덤으로 나머지 17개의 특성 생성하는 함수
def random_features():
    return [random.uniform(0, 1) for _ in range(18)]

@login_required
def upload_form(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        df = pd.read_csv(uploaded_file)

        # 사용자 데이터베이스 설정
        db_name = f'user_{request.user.username}_db'
        create_db_sql = f"CREATE DATABASE IF NOT EXISTS `{db_name}`"

        if db_name not in settings.DATABASES:
            settings.DATABASES[db_name] = {
                'ENGINE': 'django.db.backends.mysql',
                'NAME': db_name,
                'USER': 'fds001',
                'PASSWORD': 'fds001',
                'HOST': '10.0.1.7',
                'PORT': '3306',
            }

        try:
            # MySQL 데이터베이스 연결 및 테이블 생성
            with connection.cursor() as cursor:
                cursor.execute(create_db_sql)

            with connections[db_name].cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS fraud_transaction (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        amount DECIMAL(10, 2),
                        customer_id VARCHAR(255),
                        merchant_id VARCHAR(255),
                        lat DECIMAL(10, 8),
                        `long` DECIMAL(11, 8),
                        fraud_prediction DECIMAL(10, 8)
                    )
                ''')

                index_name = f'user_{request.user.username}_transactions'
                customer_id_encoder = LabelEncoder()
                merchant_id_encoder = LabelEncoder()

                df['Customer ID'] = customer_id_encoder.fit_transform(df['Customer ID'])
                df['Merchant ID'] = merchant_id_encoder.fit_transform(df['Merchant ID'])

                model = load_model()

                for _, row in df.iterrows():
                    prediction_input = np.concatenate([np.array([row['Amount'], row['Customer ID'], row['Merchant ID'], row['Lat'], row['Long']]), random_features()])
                    prediction = model.predict(prediction_input.reshape(1, -1))[0]

                    cursor.execute('''
                        INSERT INTO fraud_transaction (amount, customer_id, merchant_id, lat, `long`, fraud_prediction)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    ''', (row['Amount'], row['Customer ID'], row['Merchant ID'], row['Lat'], row['Long'], prediction))

                    # Elasticsearch에 인덱싱
                    es.index(index=index_name, body={
                        'amount': row['Amount'],
                        'customer_id': row['Customer ID'],
                        'merchant_id': row['Merchant ID'],
                        'lat': row['Lat'],
                        'long': row['Long'],
                        'fraud_prediction': prediction,
                        'user_id': request.user.username,
                        'timestamp': datetime.now(),
                    })

            # Kibana 대시보드를 iframe으로 보여주기 위한 URL 생성
            kibana_url = f"http://211.188.51.137:5601/app/dashboards#/view/78e0ff70-7b56-11ef-9b66-13538d49ff90?_a=(filters:!((meta:(alias:!n,disabled:!f,key:user_id,negate:!f,params:(query:'{{ request.user.username }}'),type:phrase)))"
            
            


            # dashboard.html로 리다이렉트하면서 Kibana 대시보드를 iframe으로 보여줌
            return render(request, 'fraud/dashboard.html', {'kibana_url': kibana_url})

        except OperationalError as e:
            return HttpResponse(f"데이터베이스 연결 오류: {str(e)}")

    return render(request, 'fraud/upload_form.html')





# Kibana에 Index Pattern 생성 함수
def create_kibana_index_pattern(index_name):
    kibana_url = "http://localhost:5601/api/saved_objects/index-pattern"
    headers = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true"
    }
    data = {
        "attributes": {
            "title": index_name,
            "timeFieldName": "timestamp",
            "fields": '[{"name":"user_id","type":"keyword"}]'  # user_id 필드를 명시적으로 추가
        }
    }

    try:
        response = requests.post(kibana_url, headers=headers, json=data, auth=('kibana_user', 'kibana_password'))
        response.raise_for_status()
        print(f"Successfully created index pattern for {index_name}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to create index pattern in Kibana: {str(e)}")








"""
# Kibana 대시보드 생성 함수
def create_kibana_dashboard(username):
    kibana_url = "http://211.188.51.137:5601/api/saved_objects/dashboard"
    headers = {
        "kbn-xsrf": "true",
        "Content-Type": "application/json"
    }

    # 새로운 대시보드 설정
    dashboard_data = {
        "attributes": {
            "title": f"user_{username}_dashboard",
            "description": f"Dashboard for {username}",
            "panelsJSON": json.dumps([  # 기본 패널 설정
                {
                    "panelIndex": "1",
                    "gridData": {
                        "x": 0,
                        "y": 0,
                        "w": 24,
                        "h": 15,
                        "i": "1"
                    },
                    "embeddableConfig": {},
                    "panelRefName": "panel_0"
                }
            ]),
            "optionsJSON": '{"hidePanelTitles":false,"useMargins":true}',
            "version": "1"
        }
    }

    # Kibana API 호출
    response = requests.post(kibana_url, json=dashboard_data, headers=headers)

    # 응답 데이터 출력
    print(response.json())  # Kibana API 응답 데이터 확인

    # 응답 확인 및 id 필드가 있는지 확인
    if response.status_code == 200:
        response_data = response.json()
        if "id" in response_data:
            dashboard_id = response_data["id"]  # _id 대신 id 사용
            return dashboard_id
        else:
            # 로그에 기록 또는 디버깅 메시지 추가
            print("Kibana 응답에 id 필드가 없습니다:", response_data)
            return None
    else:
        # 오류 처리
        print(f"Kibana 대시보드 생성 실패: {response.status_code}")
        return None

"""







# 랜덤으로 나머지 17개의 특성 생성하는 함수
def random_features():
    return [random.uniform(0, 1) for _ in range(18)]





from django.shortcuts import render, redirect
from .models import Transaction

@login_required
def dashboard(request):
    db_name = f'user_{request.user.username}_db'

    # 데이터베이스 연결 설정을 동적으로 추가
    if db_name not in settings.DATABASES:
        settings.DATABASES[db_name] = {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': db_name,
            'USER': 'fds001',
            'PASSWORD': 'fds001',
            'HOST': '10.0.1.7',
            'PORT': '3306',
        }

    try:
        # db_name 데이터베이스에서 데이터를 가져옴
        with connections[db_name].cursor() as cursor:
            cursor.execute('SELECT amount, customer_id, merchant_id, lat, `long`, fraud_prediction FROM fraud_transaction')
            transactions = cursor.fetchall()

        # 데이터를 JSON으로 변환하여 템플릿으로 전달
        transaction_data = [{
            'amount': row[0],
            'customer_id': row[1],
            'merchant_id': row[2],
            'lat': row[3],
            'long': row[4],
            'fraud_prediction': row[5],
        } for row in transactions]


        # 대시보드 페이지로 렌더링
        return render(request, 'fraud/dashboard.html', {'transactions': transaction_data})
    
    except OperationalError as e:
        return HttpResponse(f"Database connection for {db_name} failed: {str(e)}")



from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages

# 로그인 뷰
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome, {username}!')
                return redirect('home')  # 로그인 성공 시 리다이렉션
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()
    return render(request, 'fraud/login.html', {'form': form})

# 로그아웃 뷰
def logout_view(request):
    logout(request)
    messages.success(request, 'You have successfully logged out.')
    return redirect('home')



from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.contrib.auth import login

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # 회원가입 후 자동 로그인
            return redirect('index')  # 성공 후 리다이렉트
    else:
        form = UserCreationForm()
    return render(request, 'fraud/signup.html', {'form': form})








# API 인증 (시큐어 코딩)
'''
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authentication import TokenAuthentication
from rest_framework.permission import IsAuthenticated
import pandas as pd
from sklearn.ensemble import IsolationForest

class PredictView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        data = request.data

        model = IsolationForest(contamination = 0.01, n_estimators = 100, max_samples = 'auto', random_state = 42)
        model.fit(x_train)
        prediction = model.perdict([data])

        return Response({'prediction': prediction.tolist()})
'''
from django.urls import path
from . import views

urlpatterns = [
    path('test-fraud/', views.test_fraud, name='test_fraud'), 
    path('upload/', views.upload_form, name='upload_form'),      # 업로드 페이지
    path('dashboard/', views.dashboard, name='dashboard'),        # 대시보드 페이지
    path('index/', views.index, name='index'),                   # 인덱스 페이지 추가
    
]

from django.urls import path
from . import views

urlpatterns = [
    path('test-fraud/', views.test_fraud, name='test_fraud'), 
    path('upload/', views.upload_form, name='upload_form'),      # 업로드 페이지
    path('dashboard/', views.dashboard, name='dashboard'),        # 대시보드 페이지
    path('index/', views.index, name='index'),                   # 인덱스 페이지 추가
    path('dashboard2/', views.dashboard2, name='dashboard2'),  # dashboard2 URL 패턴 추가
    path('upload_form2/', views.upload_form2, name='upload_form2'),  # upload_form2 URL 패턴 추가

]

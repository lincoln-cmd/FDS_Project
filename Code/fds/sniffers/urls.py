"""sniffers URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
#from fraud.views import FraudDetectionView # FraudDetectionView 클래스 import
from django.views.generic.base import RedirectView
from fraud import views

from django.contrib.auth import views as auth_views
from django.urls import path, include
from fraud import views as fraud_views
from django.views.generic.base import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('fraud/', include('fraud.urls')),
    path('', RedirectView.as_view(url='/fraud/index/', permanent=False), name='home'),
    path('login/', auth_views.LoginView.as_view(template_name='fraud/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', fraud_views.signup, name='signup'),

    # /accounts/login/ 경로에도 로그인 템플릿 지정
    path('accounts/login/', auth_views.LoginView.as_view(template_name='fraud/login.html')),
]

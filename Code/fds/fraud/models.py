from django.db import models

# Create your models here.

class FraudData(models.Model):
    transaction_id = models.IntegerField(primary_key=True)
    amount = models.FloatField()
    customer_id = models.IntegerField()
    merchant_id = models.IntegerField()
#    transaction_date = models.DateTimeField()
    lat = models.IntegerField(default = 0)
    long = models.IntegerField(default = 0)

    def __str__(self):
        return str(self.transaction_id)
    

from django.db import models

class Transaction(models.Model):
    transaction_id = models.CharField(max_length=100)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    customer_id = models.CharField(max_length=100)
    merchant_id = models.CharField(max_length=100)
    lat = models.FloatField()
    long = models.FloatField()
    fraud_prediction = models.FloatField(null=True)  # 예측 값 필드 추가

    def __str__(self):
        return self.transaction_id

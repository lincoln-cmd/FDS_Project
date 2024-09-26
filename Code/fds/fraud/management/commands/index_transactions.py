# fraud/management/commands/index_transactions.py

from django.core.management.base import BaseCommand
from fraud.models import Transaction
from elasticsearch import Elasticsearch
import pickle

class Command(BaseCommand):
    help = 'Index transactions to Elasticsearch'

    def handle(self, *args, **options):
        es = Elasticsearch([{'host': '10.0.1.7', 'port': 9200}])  # Elasticsearch URL
        
        # Load the model
        with open('fraud/models/FDS_model_isolationForest.pkl', 'rb') as f:
            model = pickle.load(f)

        for transaction in Transaction.objects.all():
            doc = {
                'transaction_id': transaction.transaction_id,
                'amount': transaction.amount,
                'customer_id': transaction.customer_id,
                'merchant_id': transaction.merchant_id,
                'lat': transaction.lat,
                'long': transaction.long,
                'fraud_prediction': transaction.fraud_prediction,
            }
            es.index(index='transactions', id=transaction.id, body=doc)

        self.stdout.write(self.style.SUCCESS('Successfully indexed transactions.'))

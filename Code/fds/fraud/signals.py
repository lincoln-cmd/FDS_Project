from django.db import connection
from django.contrib.auth.models import User
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.db.utils import OperationalError

# 사용자 생성 시 데이터베이스 생성
@receiver(post_save, sender = User)
def create_user_database(sender, instance, created, **kwargs):
    if created:
        db_name = f'user_{instance.username}_db'
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database {db_name} created.")
        except OperationalError:
            print(f"Database {db_name} already exists.")


# 사용자 삭제 시 데이터베이스 삭제
@receiver(post_delete, sender = User)
def delete_user_database(sender, instance, **kwargs):
    db_name = f'user_{instance.username}_db'
    with connection.cursor() as cursor:
        cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
    print(f"Database {db_name} deleted.")

# 사용자 삭제 시 Elasticsearch 인덱스 자동 삭제
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.contrib.auth.models import User
from elasticsearch import Elasticsearch

@receiver(post_delete, sender=User)
def delete_elasticsearch_index(sender, instance, **kwargs):
    es = Elasticsearch([{'host': '10.0.1.7', 'port': 9200}])
    index_name = f'user_{instance.username}_transactions'
    try:
        es.indices.delete(index=index_name)
        print(f"Successfully deleted index: {index_name}")
    except Exception as e:
        print(f"Failed to delete index: {str(e)}")

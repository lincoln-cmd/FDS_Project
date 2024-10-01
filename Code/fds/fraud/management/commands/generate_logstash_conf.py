import os
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Generate Logstash configuration for a specific user'

    def handle(self, *args, **kwargs):
        # 사용자 ID를 받아서 동적으로 Logstash 설정 파일 생성
        user_id = input("Enter the user ID: ")

        # 설정 파일 생성 경로
        conf_file_path = f"/etc/logstash/conf.d/logstash_{user_id}.conf"

        # 설정 파일의 내용
        logstash_conf = f"""
        input {{
          jdbc {{
            jdbc_driver_library => "/path/to/mysql-connector-java.jar"
            jdbc_driver_class => "com.mysql.jdbc.Driver"
            jdbc_connection_string => "jdbc:mysql://10.0.1.7:3306/user_{user_id}_transaction"
            jdbc_user => "fds001"
            jdbc_password => "fds001"
            statement => "SELECT * FROM fraud_transaction"
          }}
        }}

        output {{
          elasticsearch {{
            hosts => ["http://localhost:9200"]
            index => "user_{user_id}_transactions"
          }}
        }}
        """

        # 설정 파일 저장
        with open(conf_file_path, 'w') as f:
            f.write(logstash_conf)

        print(f"Logstash configuration for user {user_id} has been generated at {conf_file_path}")

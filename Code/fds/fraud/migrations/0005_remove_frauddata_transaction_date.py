# Generated by Django 3.2.12 on 2024-09-13 03:50

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('fraud', '0004_auto_20240913_0347'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='frauddata',
            name='transaction_date',
        ),
    ]

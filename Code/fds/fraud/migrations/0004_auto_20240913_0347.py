# Generated by Django 3.2.12 on 2024-09-13 03:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fraud', '0003_frauddata'),
    ]

    operations = [
        migrations.AddField(
            model_name='frauddata',
            name='lat',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='frauddata',
            name='long',
            field=models.IntegerField(default=0),
        ),
    ]

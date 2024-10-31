import sqlite3
from django.core.management.base import BaseCommand
from project3.models import PubMedArticle

class Command(BaseCommand):
    help = "Import data from pubmed.db into Django"

    def handle(self, *args, **options):
        # Connect to the SQLite database
        conn = sqlite3.connect('./pubmed.db')
        cursor = conn.cursor()
        
        # Fetch all records
        cursor.execute("SELECT tag, pmid, title, pubdate, abstract FROM pubmed_articles")  # Adjust table name if needed
        records = cursor.fetchall()

        for record in records:
            tag, pmid, title, pubdate, abstract = record
            
            # Create or update each article in Django
            PubMedArticle.objects.update_or_create(
                pmid=pmid,
                defaults={
                    'tag': tag,
                    'title': title,
                    'pubdate': pubdate,
                    'abstract': abstract
                }
            )

        conn.close()
        self.stdout.write(self.style.SUCCESS("Data imported successfully"))

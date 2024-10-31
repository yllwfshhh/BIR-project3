from django.db import models

# Create your models here.
class PubMedArticle(models.Model):
    tag = models.CharField(max_length=100)
    pmid = models.CharField(max_length=50, unique=True)
    title = models.TextField()
    pubdate = models.TextField()
    abstract = models.TextField()

    def __str__(self):
        return self.title

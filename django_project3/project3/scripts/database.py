import os
import sys
import django
import xml.etree.ElementTree as ET

# Add the Django project root directory to the path
sys.path.append('/home/project3/django_project3')  # Use the absolute path to `django_project3`
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_project3.settings')
django.setup()
from project3.models import PubMedArticle


def parse_pubmed_xml(xml_file, tag):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    pmid = root.find(".//PMID").text if root.find(".//PMID") is not None else "Unknown PMID"
    title = root.find(".//ArticleTitle").text if root.find(".//ArticleTitle") is not None else "No Title"
    pubdate = extract_pubdate(root)
    abstract_list = []
    for elem in root.iter('AbstractText'):
        if elem is not None:
            elem_text = ''.join(elem.itertext())
            abstract_list.append(elem_text)
    abstract = "\n".join(abstract_list)

    # Check for required fields
    if not pmid or not title:
        print(f"Skipping record due to missing PMID or title. PMID: {pmid}, Title: {title}")
        return
    
    # Save to database
    article, created = PubMedArticle.objects.update_or_create(
        pmid=pmid,
        defaults={
            "tag": tag,
            "title": title,
            "pubdate": pubdate,
            "abstract": abstract
        }
    )
    print(f"Processed article with PMID: {pmid}")

def extract_pubdate(root):
    pubdate_element = root.find(".//PubDate")
    # Check if PubDate exists
    if pubdate_element is not None:
        year = pubdate_element.find("Year").text if pubdate_element.find("Year") is not None else "Unknown Year"
        month = pubdate_element.find("Month").text if pubdate_element.find("Month") is not None else "Unknown Month"
        pubdate = f"{year}-{month}"
    else:
        pubdate = "Unknown Published Date"
    return pubdate

def insert_database(directory, tag):
    
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            xml_file = os.path.join(directory, filename)
            print(xml_file)
            parse_pubmed_xml(xml_file, tag)

insert_database("../../data/depression","depression")
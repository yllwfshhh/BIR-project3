import os
import xml.etree.ElementTree as ET
import time
import requests


def extract_pubmed_info(xml_file_name, query):
    # Parse the XML file
    tree = ET.parse(xml_file_name)
    root = tree.getroot()

    # Initialize a dictionary to store the extracted information
    info = {}
    article_title = root.find(".//ArticleTitle")
    print(article_title)

    if article_title is not None:
        info['ArticleTitle'] = article_title.text
    else:
        info['ArticleTitle'] = "No Title Found"

    # Add more fields to extract as needed (e.g., abstract, authors)
    abstract = root.find(".//AbstractText")
    if abstract is not None:
        info['Abstract'] = abstract.text
    else:
        info['Abstract'] = "No Abstract Found"

    return info


def download_pubmed(query,num,batch_size) :
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": num,
        "retmode": "xml"
    }
    search_response = requests.get(search_url, params=search_params)
    search_response.raise_for_status()
    search_xml = ET.fromstring(search_response.content)
    pmids = [id_elem.text for id_elem in search_xml.findall(".//Id")]

    if not pmids:
        print("No results found.")
        return
    
    # FOLDER PATH
    base_path = os.path.join("data",query)
    print("folder path:",base_path)

    if os.path.isdir(base_path):
        print(f"{query} folder exists")
    else:
        print(f"forder didn't exists, make a new  folder {query}")
        os.makedirs(os.path.join(base_path),exist_ok=True)
    
    fetch_url = f"{base_url}efetch.fcgi"

    for i in range(0, len(pmids), batch_size):
        # Split PMIDs into batches
        batch_pmids = pmids[i:i + batch_size]
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(batch_pmids),
            "retmode": "xml"
        }
        try:
            fetch_response = requests.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()

            articles = ET.fromstring(fetch_response.content).findall(".//PubmedArticle")
            
            for article in articles:
                pmid = article.find(".//PMID").text
                file_path = os.path.join(base_path, f"X{pmid}.xml")

                with open(file_path, 'wb') as xml_file:
                    xml_file.write(ET.tostring(article, encoding='utf-8'))

                print(f"{pmid} added")

        except requests.exceptions.HTTPError as e:
            print(f"Failed to fetch batch {i+1} due to {e}")
            
        time.sleep(0.1)
        
    return

query = "enterovirus"
num = 10
batch_size = 5

download_pubmed(query, num, batch_size)
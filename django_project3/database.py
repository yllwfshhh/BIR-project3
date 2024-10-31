import xml.etree.ElementTree as ET
import os
import re
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize

# nltk.download('punkt_tab')
# nltk.download('stopwords')

### DATABASE

def parse_pubmed_xml(xml_file,tag):
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
    return {"tag": tag, "pmid": pmid, "title": title, "pubdate":pubdate, "abstract": abstract}

def check_pubmed(xml_file,db_name):

    conn = sqlite3.connect(db_name)
    c = conn.cursor()   
    pmid = xml_file.split('X')[1].split('.')[0]
    c.execute("SELECT 1 FROM pubmed_articles WHERE pmid = ?", (pmid,))
    result = c.fetchone()
    conn.close()
    # Return True if the title exists, False otherwise
    return result is not None

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

def parse_all_xml_files(data_dir,db_name):
    articles = []
    for tag in os.listdir(data_dir):
        xml_dir = os.path.join(data_dir, tag)
        print(f"Current folder: {xml_dir}")
        for file_name in os.listdir(xml_dir):
            if file_name.endswith(".xml"):
                file_path = os.path.join(xml_dir, file_name)
                if not check_pubmed(file_path,db_name): 
                    articles.append(parse_pubmed_xml(file_path,tag))
                    print(f'{file_name} is added to data')
    return articles

def create_database(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS pubmed_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT,
            pmid TEXT UNIQUE,
            title TEXT,
            pubdate TEXT,
            abstract TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_to_db(db_name, articles):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    for article in articles:
        c.execute('''
            INSERT OR IGNORE INTO pubmed_articles (tag, pmid, title, pubdate, abstract)
            VALUES (?, ?, ?, ?, ?)
        ''', (article['tag'],article['pmid'], article['title'], article['pubdate'], article['abstract']))
    conn.commit()
    conn.close()

### Get Data

def get_abstract_sentences(db_name,tag):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    if tag:
        query  = '''
        SELECT abstract FROM pubmed_articles
        WHERE tag LIKE ?
        '''
        c.execute(query, ('%' + tag + '%',))
    else:
        query = "SELECT abstract FROM pubmed_articles"  # Get all abstracts
        c.execute(query)
    
    abstracts = c.fetchall() 
    conn.close()

    abstract_sentences = []
    for abstract_tuple in abstracts:
        abstract = abstract_tuple[0]  # Extract the abstract text
        if abstract:
            sentences = sent_tokenize(abstract)  # Split abstract into sentences
            abstract_sentences.extend(sentences)

    return abstract_sentences

def get_all_words(db_name,tag):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    if tag:
        query  = '''
        SELECT abstract FROM pubmed_articles
        WHERE tag LIKE ?
        '''
        c.execute(query, ('%' + tag + '%',))
    else:
        query = "SELECT abstract FROM pubmed_articles"  # Get all abstracts
        c.execute(query)
    
    abstracts = c.fetchall() 
    conn.close()

    # put all abstracts into one string
    all_text = " ".join([abstract[0] for abstract in abstracts if abstract[0] is not None])
    tokens = nltk.word_tokenize(all_text)
    words = [word.lower() for word in tokens if word.isalpha()] # normalize and filter
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words] # Filter out stop words from the word list

    return filtered_words


if __name__ == "__main__":

    dir = "/data/"  
    db_name = "./pubmed.db"
    create_database(db_name)
    articles = parse_all_xml_files(dir,db_name)
    insert_to_db(db_name, articles)
    # get_abstract_sentences(db_name,'')


    



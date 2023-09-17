import os
import requests
import json
import time
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import re

# Constants and Initializations
SQUABBLES_TOKEN = os.environ.get('SQUABBLES_TOKEN')
GIST_TOKEN = os.environ.get('GITHUB_TOKEN')
GIST_ID = os.environ.get('TLDRBOT_GIST')
FILE_NAME = 'tldrbot.json'
GIST_URL = f"https://gist.githubusercontent.com/amightybeard/{GIST_ID}/raw/{FILE_NAME}"
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Utility Functions
def fetch_gist_data():
    headers = {'Authorization': f'token {GIST_TOKEN}'}
    response = requests.get(GIST_URL, headers=headers)
    response.raise_for_status()
    communities_data = response.json()
    print(f"Fetched data from Gist: {communities_data}")
    return communities_data

def fetch_new_posts(community_name, last_processed_id):
    response = requests.get(f'https://squabblr.co/api/s/{community_name}/posts?page=1&sort=new')
    response.raise_for_status()
    posts_data = response.json()  # Ensure this is parsed as JSON
    posts = posts_data["data"] if "data" in posts_data else [] 
    new_posts = [post for post in posts if post['id'] > last_processed_id]
    print(f"Fetched new posts for community {community_name}: {new_posts}")
    return new_posts

def load_domain_blacklist():
    with open("includes/blacklist-domains.txt", "r") as file:
        # Strip whitespace and filter out empty lines
        return [line.strip() for line in file if line.strip()]

def is_domain_blacklisted(url, blacklist):
    domain = url.split("//")[-1].split("/")[0]
    return domain in blacklist

def scrape_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract meta description
        meta_description = soup.find('meta', attrs={"name": "description"})
        if meta_description:
            meta_description = meta_description["content"]
        
        # Extract article content
        article_content = soup.body.get_text(separator="\n").strip()
        
        return meta_description, article_content
    
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred for URL {url}: {err}")
        return None, None

    finally:
        time.sleep(3)  # Introducing a delay of 3 seconds between requests

def generate_overview(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=600, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def correct_text(text):
    # Placeholder function for now, can be expanded in the future
    corrections = {
        " netyards ": " net yards ",
        " stook ": " stood "
    }
    for error, correction in corrections.items():
        text = text.replace(error, correction)
    return text

def group_and_format_sentences(sentences):
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    clustering = AgglomerativeClustering(affinity="precomputed", linkage="average", n_clusters=None, distance_threshold=0.8)
    clustering.fit(1 - cosine_matrix)
    
    clustered_sentences = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clustered_sentences:
            clustered_sentences[label] = []
        clustered_sentences[label].append(sentences[idx])
    
    formatted_output = []
    for group, group_sentences in clustered_sentences.items():
        formatted_output.append(" - " + " ".join(group_sentences))
    return formatted_output

def process_and_format_summary(text):
    text = correct_text(text)
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return "\n".join(group_and_format_sentences(sentences))

def generate_key_points(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=1500, min_length=300, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return process_and_format_summary(summary_text)

def send_reply(post_hash_id, overview, key_points):
    headers = {'authorization': 'Bearer ' + SQUABBLES_TOKEN}
    content = (
        "This is the best TL;DR I could put together from this article:\n\n"
        "-----\n\n"
        "🔍 **Overview**:\n"
        f"{overview}\n\n"
        "📌 **Key Points**:\n"
        f"{key_points}\n\n"
        "-----\n\n"
        "I am a bot. Post feedback and suggestions to /s/ModBot. Want this bot in your community? DM @modbot with `!summarize community_name`."
    )
    resp = requests.post(f'https://squabblr.co/api/posts/{post_hash_id}/reply', data={"content": content}, headers=headers)
    resp.raise_for_status()
    return resp.json()

def update_gist(community_name, new_last_processed_id, communities_data):
    for community in communities_data:
        if community["community"] == community_name:
            community["last_processed_id"] = new_last_processed_id
            break
    headers = {
        'Authorization': f'token {GIST_TOKEN}',
        'Content-Type': 'application/json'
    }
    data = {
        "files": {
            FILE_NAME: {
                "content": json.dumps(communities_data)
            }
        }
    }
    response = requests.patch(f"https://api.github.com/gists/{GIST_ID}", headers=headers, json=data)
    response.raise_for_status()
    return response.json()

# Main Execution
def main():
    # Initialization
    communities_data = fetch_gist_data()
    domain_blacklist

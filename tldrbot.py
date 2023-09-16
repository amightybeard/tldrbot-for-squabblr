import os
import requests
import json
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer

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
    return response.json()

def fetch_new_posts(community_name, last_processed_id):
    response = requests.get(f'https://squabblr.co/api/s/{community_name}/posts?page=1&sort=new')
    response.raise_for_status()
    posts = response.json()
    new_posts = [post for post in posts if post['id'] > last_processed_id]
    return new_posts

def load_domain_blacklist():
    with open("includes/blacklist-domains.txt", "r") as file:
        # Strip whitespace and filter out empty lines
        return [line.strip() for line in file if line.strip()]

def is_domain_blacklisted(url, blacklist):
    domain = url.split("//")[-1].split("/")[0]
    return domain in blacklist

def scrape_content(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    meta_description = None
    meta_tag = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
    if meta_tag:
        meta_description = meta_tag["content"]
    paragraphs = soup.find_all("p")
    article_content = " ".join(p.text for p in paragraphs if len(p.text) > 100)
    return meta_description, article_content

def generate_overview(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=600, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_key_points(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=1500, min_length=300, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def send_reply(post_id, overview, key_points):
    headers = {'authorization': 'Bearer ' + SQUABBLES_TOKEN}
    content = f"🔍 TL;DR Overview:\n{overview}\n\n📌 Key Points:\n{key_points}"
    resp = requests.post(f'https://squabblr.co/api/posts/{post_id}/reply', data={"content": content}, headers=headers)
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
    domain_blacklist = load_domain_blacklist()
    
    # Processing
    for community in communities_data:
        new_posts = fetch_new_posts(community["community"], community["last_processed_id"])
        for post in new_posts:
            post_url = post["url_meta"]["url"]
            if is_domain_blacklisted(post_url, domain_blacklist):
                continue
            meta_description, article_content = scrape_content(post_url)
            overview = meta_description if meta_description else generate_overview(article_content)
            key_points = generate_key_points(article_content)
            send_reply(post["id"], overview, key_points)
            update_gist(community["community"], post["id"], communities_data)

if __name__ == "__main__":
    main()

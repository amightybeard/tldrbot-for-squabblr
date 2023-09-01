import requests
import os
import json
import logging
import io
from urllib.parse import urlparse
import csv
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer
from datetime import datetime

SQUABBLES_TOKEN = os.environ.get('SQUABBLES_TOKEN')
GIST_TOKEN = os.environ.get('GITHUB_TOKEN')
GIST_ID = 'fd0e432ea9f3f7d1869ea7a52e26d6fe'
FILE_NAME = 'tldrbot-timestamp.csv'
CSV_PATH = 'includes/communities.csv'
DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
logging.basicConfig(level=logging.INFO)

canned_message_header = """
This is the best TL;DR I could come up with for this article:

-----
"""

canned_message_footer = """

-----

I am a bot. Submit feedback to the /s/ModBot.
"""

def post_reply(post_id, summary):
    headers = {
        'authorization': 'Bearer ' + SQUABBLES_TOKEN
    }
    resp = requests.post(f'https://squabblr.co/api/posts/{post_id}/reply', data={
        "content": summary
    }, headers=headers)
    
    if resp.status_code == 200:
        logging.info(f"Successfully posted a reply for post ID: {post_id}")
    else:
        logging.warning(f"Failed to post a reply for post ID: {post_id}. Response: {resp.text}")
        
    return resp.json()

from io import StringIO

def get_last_timestamp(community_name):
    """
    Get the last timestamp for a specific community from the CSV in the Gist.
    """
    logging.info(f"Fetching last timestamp for community: {community_name}")
    
    # Use the raw URL to fetch the file content directly
    raw_gist_url = f"https://gist.githubusercontent.com/amightybeard/{GIST_ID}/raw/{FILE_NAME}"
    response = requests.get(raw_gist_url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch CSV from Gist. Status code: {response.status_code}")
        return None

    csv_content = response.text  # Get the CSV content directly
    
    reader = csv.DictReader(io.StringIO(csv_content))
    for row in reader:
        if row['Community'] == community_name:
            return datetime.strptime(row['Post Created Date'], DATE_FORMAT)
    return None

def save_last_timestamp(community, post_url, timestamp):
    logging.info(f"Saving last timestamp for community: {community}")
    
    # First, fetch the current content of the CSV from the Gist
    resp = requests.get(f'https://api.github.com/gists/{GIST_ID}', headers={'Authorization': f'token {GIST_TOKEN}'})
    data = resp.json()
    print(f"GitHub API Response: {data}")
    # Extract the CSV content from the Gist response
    csv_content = data['files'][FILE_NAME]['content']

    # Append the new entry to the CSV content
    new_entry = f"\n{community},{post_url},{timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}"
    updated_csv_content = csv_content + new_entry

    # Update the Gist with the new CSV content
    gist_update_data = {
        "files": {
            FILE_NAME: {
                "content": updated_csv_content
            }
        }
    }
    update_resp = requests.patch(f'https://api.github.com/gists/{GIST_ID}', headers={'Authorization': f'token {GIST_TOKEN}'}, json=gist_update_data)
    if update_resp.status_code == 200:
        print(f"Successfully updated the timestamp for {community} in the Gist.")
    else:
        print(f"Failed to update the Gist. Response code: {update_resp.status_code}. Response text: {update_resp.text}")

def read_domain_blacklist(filename="includes/blacklist-domains.txt"):
    with open(filename, 'r') as file:
        blacklist = [line.strip() for line in file]
    return blacklist

def format_as_bullet_points(summary):
    # summary = summary.lstrip('-').strip()  # Remove leading '-' and strip any whitespace

    # Split the summary into sentences
    # sentences = summary.split("[BREAK]")
    
    # Convert sentences into bullet points
    # bullet_points = "\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence.strip()])
    # return bullet_points

    return summary

def read_word_blacklist(filename="includes/blacklist-words.txt"):
    with open(filename, 'r') as file:
        blacklist = [line.strip() for line in file]
    return blacklist

from bs4 import BeautifulSoup
import requests
from transformers import BartTokenizer, BartForConditionalGeneration

# Initialize the BART model and tokenizer
MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

def extract_content_with_bs(url):
    """
    Extracts main content of an article using BeautifulSoup
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract title
    title = soup.title.string if soup.title else ""

    # Extract main content based on common tags used for main article content
    content_tags = ['p']
    content = [tag.get_text() for tag in soup.find_all(content_tags)]
    content = "\n".join(content)

    return title + "\n" + content

def get_summary(url):
    """
    Gets the summary of an article
    """
    # Extract content using BeautifulSoup
    article_content = extract_content_with_bs(url)

    # Summarize using BART
    inputs = tokenizer.encode("summarize: " + article_content, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


def get_latest_posts(username, community):
    last_timestamp = get_last_timestamp(community) or datetime.min
    data = requests.get(f'https://squabblr.co/api/s/{community}/posts?page=1&sort=new&').json()
    logging.info(f"Checking posts for community: {community}")

    domain_blacklist = read_domain_blacklist()
    logging.info(f"Loaded domain blacklist: {domain_blacklist}")

    for post in data.get('data', []):
        created_at = post['created_at']
        post_date = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%S.%fZ')
        if post_date > last_timestamp:
            post_id = post['hash_id']
            logging.info(f"New post found with ID: {post_id} and date: {post_date}")
            if post.get("url_meta"):
                url = post["url_meta"].get("url")
                post_domain = urlparse(url).netloc
                if post_domain in domain_blacklist:
                    continue  # Skip this post if its domain is blacklisted
                if url:
                    summary = get_summary(url)
                    word_blacklist = read_word_blacklist()
                    for word in word_blacklist:
                        summary = summary.replace(word, "")  # Remove blacklisted words/phrases from the summary

                    if summary:
                        # Convert the summary into bullet points
                        formatted_summary = format_as_bullet_points(summary)
                        response = f"{canned_message_header}\n{formatted_summary}\n{canned_message_footer}"
                        r = post_reply(post_id, response)
                        if r:
                            print("Posted: " + post_id)
                            # Save the post details to the CSV
                            save_last_timestamp(community, post["url"], post_date)
                        else:
                            print(post)
                            print("Failed Posting: " + post_id)
            else:
                logging.info(f"No new post since the last timestamp: {last_timestamp}")


if __name__ == "__main__":
    with open(CSV_PATH, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            get_latest_posts(row['Username'], row['Community'])

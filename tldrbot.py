import requests
import os
import json
import logging
import io
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer
from datetime import datetime
import re
from collections import Counter

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Constants
SQUABBLES_TOKEN = os.environ.get('SQUABBLES_TOKEN')
GIST_TOKEN =  os.environ.get('GITHUB_TOKEN')
GIST_ID = os.environ.get('TLDRBOT_GIST')
FILE_NAME = 'tldrbot.json'
GIST_URL = f"https://gist.githubusercontent.com/amightybeard/{GIST_ID}/raw/{FILE_NAME}"
DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Initialize BART model and tokenizer
MODEL_NAME = "facebook/bart-large-cnn"
MODEL = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
TOKENIZER = BartTokenizer.from_pretrained(MODEL_NAME)

canned_message_header = """
This is the best TL;DR I could come up with for this article:

-----
"""

canned_message_footer = """

-----

I am a bot. Post feedback to /s/ModBot.
"""

def post_reply(post_id, content):
    headers = {
        'authorization': 'Bearer ' + SQUABBLES_TOKEN
    }
    
    resp = requests.post(f'https://squabblr.co/api/posts/{post_id}/reply', data={
        "content": content
    }, headers=headers)
    
    if resp.status_code in [200, 201]:
        logging.info(f"Successfully posted a reply for post ID: {post_id}")
    else:
        logging.warning(f"Failed to post a reply for post ID: {post_id}.")

    # Log the response status and content
    logging.info(f"Response status from Squabblr API when posting reply: {resp.status_code}")
    logging.info(f"Response content from Squabblr API when posting reply: {resp.text}")
    
    return resp.json()

from io import StringIO

def fetch_last_processed_ids():
    try:
        resp = requests.get(GIST_URL)
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.RequestException as e:
        logging.error(f"Failed to fetch processed IDs from Gist. Error: {e}")
        return {}

def save_processed_id(community, post_id):
    data = fetch_last_processed_ids()
    data[community]['last_processed_id'] = post_id
    
    headers = {
        'Authorization': f'token {GIST_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "files": {
            FILE_NAME: {
                "content": json.dumps(data, indent=4)
            }
        }
    }

    try:
        resp = requests.patch(f"https://api.github.com/gists/{GIST_ID}", headers=headers, json=payload)
        resp.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to update processed ID for {community}. Error: {e}. Resp: {resp}")

def read_domain_blacklist(filename="includes/blacklist-domains.txt"):
    with open(filename, 'r') as file:
        blacklist = [line.strip() for line in file]
    return blacklist
    
def split_into_sentences(text):
    # Use regular expression to split sentences by common punctuation used at the end of sentences
    return re.split(r'(?<=[.!?])\s+', text)
    
def extract_content_with_bs(url):
    """
    Extracts main content of an article using BeautifulSoup
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Log the initiation of the request
    logging.info(f"Initiating request to URL: {url}")
    
    response = requests.get(url, headers=headers)
    
    # Log the response status code
    logging.info(f"Received response with status code: {response.status_code}")

    # Start parsing with BeautifulSoup
    logging.info(f"Starting content extraction for URL: {url}")
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract title
    title = soup.title.string if soup.title else ""
    
    # Log if the title was found or not
    if title:
        logging.info(f"Title found for URL: {url} - '{title}'")
    else:
        logging.warning(f"No title found for URL: {url}")

    # Extract main content based on common tags used for main article content
    content_tags = ['p']
    content = [tag.get_text() for tag in soup.find_all(content_tags)]
    
    # Log the number of paragraphs/content tags found
    logging.info(f"Found {len(content)} paragraphs/content tags for URL: {url}")

    content = "\n".join(content)

    # Log a snippet of the extracted content
    logging.info(f"Content snippet for URL: {url} - '{content[:100]}...'")

    return title + "\n" + content

def get_summary(article):
    try:
        logging.info(f"Starting the summary generation for article content")

        if not article or len(article.strip()) == 0:
            logging.error(f"No valid content provided.")
            return None

        inputs = TOKENIZER([article], max_length=1024, return_tensors='pt', truncation=True)

        logging.info(f"Content tokenized.")

        if not inputs or not hasattr(inputs, 'input_ids') or len(inputs.input_ids) == 0:
            logging.error(f"Failed to tokenize content.")
            return None

        logging.info(f"Starting the model generation process.")
        summary_ids = MODEL.generate(inputs.input_ids, num_beams=6, repetition_penalty=2.0, length_penalty=1.2, max_length=800, min_length=100, no_repeat_ngram_size=2)

        if len(summary_ids) == 0:
            logging.error(f"Failed to generate summary IDs. Model returned empty IDs.")
            return None

        summary = TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)
        logging.info(f"Summary generated.")

        # Extract main points
        main_points = get_main_points(article)
        
        # Remove points that are very similar to the summary
        main_points = [point for point in main_points if point not in summary]
    
        return summary, main_points

    except Exception as e:
        logging.error(f"Error in generating summary. Error: {e}")
        return None

def get_main_points(text, num_points=5, max_length=150):
    # Tokenize the article into sentences
    sentences = split_into_sentences(text)
    
    # Tokenize each sentence into words and count word frequency
    word_freq = Counter(re.findall(r'\w+', text.lower()))
    
    # Score sentences based on word frequency
    ranked_sentences = sorted(
        ((sum(word_freq[word] for word in re.findall(r'\w+', sentence.lower())), idx, sentence)
         for idx, sentence in enumerate(sentences)),
        reverse=True
    )
    
    # Extract top n sentences as main points
    main_points = []
    for i in range(min(num_points, len(ranked_sentences))):
        sentence = ranked_sentences[i][2]
        if len(sentence) <= max_length:  # If the sentence is short enough, use it as-is
            main_points.append(sentence)
        else:
            # If the sentence is long, truncate it
            truncated = sentence[:max_length-3] + "..."
            main_points.append(truncated)

    return main_points

def get_latest_posts():
    processed_ids = fetch_last_processed_ids()
    domain_blacklist = read_domain_blacklist()

    for community, data in processed_ids.items():
        last_processed_id = data['last_processed_id']

        logging.info(f"Checking posts for community: {community}")
        response = requests.get(f'https://squabblr.co/api/s/{community}/posts?page=1&sort=new&')

        if response.status_code != 200:
            logging.warning(f"Failed to fetch posts for community: {community}")
            continue

        posts = response.json()["data"]
        for post in posts:
            post_id = post['id']

            if post_id <= last_processed_id:
                continue

            url_meta = post.get("url_meta")
            if not url_meta:
                logging.warning(f"Post with ID: {post['hash_id']} lacks 'url_meta'. Skipping.")
                continue

            # Check post type and skip if it's an image
            post_type = url_meta.get("type")
            if post_type == "image":
                logging.info(f"Skipping post with ID: {post['hash_id']} as it is an image.")
                continue

            url = url_meta.get("url")
            post_domain = urlparse(url).netloc
            if post_domain in domain_blacklist:
                logging.info(f"Skipping post with URL: {url} as its domain is blacklisted.")
                continue

            logging.info(f"New post found with ID: {post_id} in community: {community}")
            content = extract_content_with_bs(url)
            
            if content:
                summary, main_points = get_summary(content)
                if summary and main_points:
                    r = post_reply(post['hash_id'], summary)
                    final_reply = f"{canned_message_header}\n{summary}\n\n**Main Points**:\n" + "\n".join([f"- {point}" for point in main_points]) + f"\n{canned_message_footer}"
                    post_reply(post['hash_id'], final_reply)

                    if 'id' in r:
                        logging.info(f"Successfully posted a reply for post ID: {post['hash_id']}")
                        save_processed_id(community, post_id)
                    else:
                        logging.warning(f"Failed to post a reply for post ID: {post['hash_id']}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_latest_posts()

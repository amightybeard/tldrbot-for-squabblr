import requests
import os
import json
import logging
import io
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Constants
SQUABBLES_TOKEN = os.environ.get('SQUABBLES_TOKEN')
GIST_TOKEN = os.environ.get('GITHUB_TOKEN')
GIST_ID = os.environ.get('TLDRBOT_GIST')
FILE_NAME = 'tldrbot-processed-ids.json'
CSV_PATH = 'includes/communities.csv'
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
        'Authorization': f'token {GIST_ID}',
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
        logging.error(f"Failed to update processed ID for {community}. Error: {e}")

def read_domain_blacklist(filename="includes/blacklist-domains.txt"):
    with open(filename, 'r') as file:
        blacklist = [line.strip() for line in file]
    return blacklist

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

def get_summary(url):
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()

        # Check the response
        if response.status_code != 200:
            logging.error(f"Failed to fetch content from URL: {url}. Status code: {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article = "\n".join([para.text for para in paragraphs])

        # Log a snippet of the content
        logging.info(f"Content snippet from URL {url}: {article[:100]}...")

        # Check the parsed content
        if not article or len(article.strip()) == 0:
            logging.error(f"No valid content fetched from URL: {url}")
            return None

        inputs = TOKENIZER([article], max_length=1024, return_tensors='pt', truncation=True)

        # Log the tokenized inputs
        logging.info(f"Tokenized inputs from URL {url}: {str(inputs)[:100]}...")

        # Check the tokenization results
        if not inputs or not hasattr(inputs, 'input_ids') or len(inputs.input_ids) == 0:
            logging.error(f"Failed to tokenize content from URL: {url}")
            return None

        summary_ids = MODEL.generate(inputs.input_ids, num_beams=6, length_penalty=1.0, max_length=500, min_length=100, no_repeat_ngram_size=2)
        
        # Log the number of summary IDs generated
        logging.info(f"Number of summary IDs generated for URL {url}: {len(summary_ids)}")

        if len(summary_ids) == 0:
            logging.error(f"Failed to generate summary IDs for URL: {url}. Model returned empty IDs.")
            return None

        summary = TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)

        # Log the shape of tokenized inputs
        logging.info(f"Shape of tokenized input_ids for URL {url}: {inputs.input_ids.shape}")

        # Check the model's expected input size
        model_input_size = MODEL.config.max_position_embeddings
        logging.info(f"Model's max input size: {model_input_size}")

        # Ensure the tokenized input does not exceed the model's maximum input size
        input_ids = inputs.input_ids[0][:model_input_size].unsqueeze(0)
        attention_mask = inputs.attention_mask[0][:model_input_size].unsqueeze(0)

        # Check if the input is too short for the model
        if len(input_ids[0]) < 5:  # Example minimum length
            logging.warning(f"Tokenized input is too short for URL {url}. Skipping.")
            return None

        # Attempt summary generation and catch potential errors
        try:
            summary_ids = MODEL.generate(input_ids, attention_mask=attention_mask, num_beams=6, length_penalty=1.0, max_length=500, min_length=100, no_repeat_ngram_size=2)
        except Exception as gen_error:
            logging.error(f"Error during summary generation for URL {url}: {gen_error}")
            return None
        
        return summary
    except Exception as e:
        logging.error(f"Error in generating summary for URL: {url}. Error: {e}")
        return None

def get_latest_posts():
    processed_ids = fetch_last_processed_ids()
    domain_blacklist = load_domain_blacklist()

    for community, data in processed_ids.items():
        last_processed_id = data['last_processed_id']

        logging.info(f"Checking posts for community: {community}")
        response = requests.get(f'https://squabblr.co/api/s/{community}/posts?page=1&sort=new&')

        if response.status_code != 200:
            logging.warning(f"Failed to fetch posts for community: {community}")
            continue

        posts = response.json()
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
            summary = generate_summary(content)
            if summary:
                r = post_reply(post['hash_id'], summary)
                if r.get('success', False):
                    logging.info(f"Successfully posted a reply for post ID: {post['hash_id']}")
                    save_processed_id(community, post_id)
                else:
                    logging.warning(f"Failed to post a reply for post ID: {post['hash_id']}.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_latest_posts()

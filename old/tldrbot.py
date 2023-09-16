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
from sklearn.feature_extraction.text import TfidfVectorizer

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

    # Remove header and footer content
    for header in soup.find_all('header'):
        header.decompose()
    for footer in soup.find_all('footer'):
        footer.decompose()

    # Extract title
    title_tag = soup.find('title')
    title = title_tag.text if title_tag else ''
    
    # Log if the title was found or not
    if title:
        logging.info(f"Title found for URL: {url} - '{title}'")
    else:
        logging.warning(f"No title found for URL: {url}")

    # Extract meta description
    meta_description = ""
    meta_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    if meta_tag:
        meta_description = meta_tag.attrs.get("content", "")

    # Extract main content based on common tags used for main article content
    content_tags = ['p']
    content_elements = soup.find_all(content_tags)
    
    # Extract text from each content element and filter out short or irrelevant content
    content = [el.text for el in content_elements if len(el.text.split()) > 5 and title not in el.text]
    
    # Log the number of paragraphs/content tags found
    logging.info(f"Found {len(content)} paragraphs/content tags for URL: {url}")

    # Join content and return
    full_content = '\n'.join(content)

    logging.info(f"Found {len(content)} paragraphs/content tags for URL: {url}")
    logging.info(f"Content snippet for URL: {url} - '{full_content[:100]}...'")

    return full_content, title, meta_description

def split_into_chunks(text, chunk_size=5):
    """
    Split the content into chunks of given size.
    """
    paragraphs = text.split('\n')
    chunks = [paragraphs[i:i+chunk_size] for i in range(0, len(paragraphs), chunk_size)]
    return ['\n'.join(chunk) for chunk in chunks]

def generate_summary(text, max_length=150):
    # Ensure the MODEL and TOKENIZER are available
    global MODEL, TOKENIZER
    
    inputs = TOKENIZER.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = MODEL.generate(inputs, max_length=max_length, min_length=50, length_penalty=5.0, num_beams=2, early_stopping=True)
    summary = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    
    return summary

def generate_comprehensive_summary(content):
    """
    Generate a summary by splitting the content into chunks and summarizing each chunk.
    """
    chunks = split_into_chunks(content)
    summaries = [generate_summary(chunk) for chunk in chunks]
    
    # Combine the summaries
    combined_summary = ' '.join(summaries)

    # Limit the combined summary to a maximum of 7 sentences
    sentences = split_into_sentences(combined_summary)
    if len(sentences) > 7:
        combined_summary = ' '.join(sentences[:7])

    # Post-process the summary to remove irrelevant lines
    cleaned_summary = post_process_summary(combined_summary)

    return cleaned_summary

def post_process_summary(summary):
    """
    Post-process the summary to remove any irrelevant or out-of-context lines.
    """
    lines = summary.split('. ')
    cleaned_lines = [line for line in lines if not line.startswith("summarize:")]
    
    return '. '.join(cleaned_lines)


def get_summary(article):
    try:
        logging.info(f"Starting the summary generation for article content")

        if not article or len(article.strip()) == 0:
            logging.error(f"No valid content provided.")
            return None

        # Generate a comprehensive summary by handling the text in chunks.
        summary = generate_comprehensive_summary(article)
        logging.info(f"Summary generated.")

        # Extract main points
        main_points = get_main_points(article)
        
        # Remove points that are very similar to the summary
        main_points = [point for point in main_points if point not in summary]
    
        return summary, main_points

    except Exception as e:
        logging.error(f"Error in generating summary. Error: {e}")
        return None, None

def get_main_points(text, num_points=5):
    """
    Extracts the main points from the given text using TF-IDF ranking.
    """
    # Tokenize the article into sentences
    sentences = split_into_sentences(text)
    
    # Use TF-IDF to rank sentences with tweaked parameters
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=3, ngram_range=(1,2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    # Sum the TF-IDF scores for each sentence to get an overall score for the sentence
    sentence_scores = tfidf_matrix.sum(axis=1).tolist()
    
    # Rank sentences based on their scores
    ranked_sentences = [sentences[idx] for idx, score in sorted(enumerate(sentence_scores), key=lambda x: x[1], reverse=True)]
    
    # Extract top n sentences as main points
    main_points = ranked_sentences[:num_points]

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
            
            content, title, meta_description = extract_content_with_bs(url)

            if content:
                content = content.replace(title, '').replace(meta_description, '')
                summary, main_points = get_summary(content)
                if summary and main_points:
                    # r = post_reply(post['hash_id'], summary)
                    final_reply = f"{canned_message_header}\n{summary}\n\n**Main Points**:\n" + "\n".join([f"- {point}" for point in main_points]) + f"\n{canned_message_footer}"
                    r = post_reply(post['hash_id'], final_reply)

                    if 'id' in r:
                        logging.info(f"Successfully posted a reply for post ID: {post['hash_id']}")
                        save_processed_id(community, post_id)
                    else:
                        logging.warning(f"Failed to post a reply for post ID: {post['hash_id']}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_latest_posts()

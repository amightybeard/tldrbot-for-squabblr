import os
import requests
import json
import logging

# Constants and Initializations
SQUABBLES_TOKEN = os.environ.get('SQUABBLES_TOKEN')
GIST_TOKEN = os.environ.get('GITHUB_TOKEN')
GIST_ID = os.environ.get('TLDRBOT_GIST')
FILE_NAME = 'tldrbot.json'
GIST_URL = f"https://gist.githubusercontent.com/amightybeard/{GIST_ID}/raw/{FILE_NAME}"
RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY')
RAPIDAPI_HOST = os.environ.get('RAPIDAPI_HOST')

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
    print(f"Fetched new posts for community {community_name}")
    return new_posts

def load_domain_blacklist():
    with open("includes/blacklist-domains.txt", "r") as file:
        # Strip whitespace and filter out empty lines
        return [line.strip() for line in file if line.strip()]

def is_domain_blacklisted(url, blacklist):
    for pattern in blacklist:
        if re.search(pattern, url):
            return True
    return False

def get_summary_from_tldrthis(post_url):
    """
    Fetches a summarized version of the content from the provided post_url using tldrthis.com.
    """
    # URL for tldrthis.com
    url = "https://tldrthis.com/tldr/process-text/"
    
    try:
        # Make a GET request to the URL with the post URL as a parameter
        response = requests.get(url, params={'text_url': post_url})
        
        # If the request was successful, extract and return the summary
        if response.status_code == 200:
            data = response.json()
            return " ".join(data[1])
        else:
            logging.error(f"Failed to fetch summary for URL {post_url}. Status code: {response.status_code}")
            return None

    except Exception as e:
        logging.error(f"Exception occurred while fetching summary for URL {post_url}. Error: {str(e)}")
        return None

def send_reply(post_hash_id, overview):
    headers = {'authorization': 'Bearer ' + SQUABBLES_TOKEN}
    content = (
        "This is the best TL;DR I could put together from this article:\n\n"
        "-----\n\n"
        f"{overview}\n\n"
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
                "content": json.dumps(communities_data, indent=4)
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
    print(f"Loaded domain blacklist: {domain_blacklist}")
    
    # Processing
    for community in communities_data:
        print(f"Processing community: {community['community']}")
        new_posts = fetch_new_posts(community["community"], community["last_processed_id"])

        if not new_posts:
            print(f"No new posts found for community {community['community']}.")
            continue
            
        for post in new_posts:
            print(f"Processing post with ID {post['id']} for community {community['community']}")

            if (
                "url_meta" not in post or 
                not post["url_meta"] or 
                "url" not in post["url_meta"] or 
                post["url_meta"]["type"] != "general"
            ):
                print(f"Skipping post with ID {post['id']} as it doesn't meet criteria.")
                continue
            
            post_url = post["url_meta"]["url"]
            if is_domain_blacklisted(post_url, domain_blacklist):
                continue
            meta_description, article_content = scrape_content(post_url)

            # Fetch the summary from tldrthis.com
            overview = get_summary_from_tldrthis(post_url)
        
            if not summary:
                logging.error(f"Failed to generate a summary for post with ID {post['id']}. Skipping.")
                continue
            
            # key_points = generate_key_points(article_content)
            print(f"Summaries generated for post with ID {post['id']} for community {community['community']}")
            send_reply(post['hash_id'], overview)
            print(f"Reply sent for post with ID {post['id']} for community {community['community']}")
            update_gist(community["community"], post["id"], communities_data)

        print("TL;DR bot processing complete.")

if __name__ == "__main__":
    main()

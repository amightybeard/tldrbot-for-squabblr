import os
import requests
import json
from bs4 import BeautifulSoup

# Constants
SQUABBLES_TOKEN = os.environ.get('SQUABBLES_TOKEN')
GIST_TOKEN = os.environ.get('GITHUB_TOKEN')
GIST_ID = os.environ.get('TLDRBOT_GIST')
FILE_NAME = 'tldrbot.json'
GIST_URL = f"https://gist.githubusercontent.com/amightybeard/{GIST_ID}/raw/{FILE_NAME}"

# Function to fetch and parse tldrbot.json from the Gist
def fetch_gist_data():
    headers = {
        'Authorization': f'token {GIST_TOKEN}'
    }
    response = requests.get(GIST_URL, headers=headers)
    response.raise_for_status()  # Will raise an error if the request failed
    return response.json()

# Function to fetch new posts for a community
def fetch_new_posts(community_name, last_processed_id):
    # Fetch the latest posts from the API
    response = requests.get(f'https://squabblr.co/api/s/{community_name}/posts?page=1&sort=new')
    response.raise_for_status()
    posts = response.json()
    
    # Filter out posts we've already processed
    new_posts = [post for post in posts if post['id'] > last_processed_id]
    
    return new_posts

# Load the domain blacklist from the file
def load_domain_blacklist():
    with open("includes/blacklist-domains.txt", "r") as file:
        # Strip whitespace and filter out empty lines
        return [line.strip() for line in file if line.strip()]
      
def main():
  # Step 1: Initialization
  try:
      communities_data = fetch_gist_data()
      print("Successfully fetched the data from tldrbot.json")
  except Exception as e:
      print(f"Error during initialization: {e}")
      return
      
  domain_blacklist = load_domain_blacklist()
  # Step 2: Fetching New Posts
  for community in communities_data:
      community_name = community["community"]
      last_processed_id = community["last_processed_id"]
      try:
          new_posts = fetch_new_posts(community_name, last_processed_id)
          print(f"Found {len(new_posts)} new posts for community: {community_name}")
      except Exception as e:
          print(f"Error fetching new posts for {community_name}: {e}")

  # Step 3: Processing Posts
  
  for post in new_posts:
      # ... (Domain blacklist check)
      
      try:
          meta_description, article_content = scrape_content(post_url)
          print(f"Scraped content from {post_url}")
      except Exception as e:
          print(f"Error scraping content from {post_url}: {e}")

if __name__ == "__main__":
    main()

# Example of scraping news articles using BeautifulSoup
import requests
from bs4 import BeautifulSoup

url = "https://www.guardian.ng"  # Example: Guardian Nigeria
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract headlines (simplified for demonstration)
headlines = soup.find_all('h2')
for headline in headlines:
    print(headline.get_text())

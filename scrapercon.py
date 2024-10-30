import concurrent.futures
from concurrent.futures import as_completed
import time
import os
import json
from scrapingbee import ScrapingBeeClient # Importing SPB's client
client = ScrapingBeeClient(api_key='') # Initialize the client with your API Key, and using screenshot_full_page parameter to take a screenshot!

MAX_RETRIES = 5 # Setting the maximum number of retries if we have failed requests to 5.
MAX_THREADS = 5
urls = []

company_name = ''
folder_name = ''

def scrape(url):
    for _ in range(MAX_RETRIES):
        response = client.get(url, params={'render_js':'True','extract_rules':{'text':'body','link':'a@href'}}) # Scrape!
        # response = client.get(url, params={'extract_rules':{'text':'body','link':'a@href'}})
        filename = url[8:].replace("/", "_")
        if len(folder_name+filename+".txt") > 255:
            filename = filename[0:(255-len(folder_name)-len(".txt"))]
        filename = filename + ".txt"
        if response.ok: # If we get a successful request
            text = response.content.decode('utf-8')
            object = json.loads(text)
            with open(folder_name+filename, "w", encoding="UTF-8") as f:
                f.write(object['text']) # Save the screenshot in the file "screenshot.png"
            with open(f'{company_name}_link_map.txt', "a") as fp:
                fp.write(f'"{folder_name+filename}": "{url}",\n')
            break # Then get out of the retry loop
        else: # If we get a failed request, then we continue the loop
            print(response)
    return url

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    executor.map(scrape,urls)
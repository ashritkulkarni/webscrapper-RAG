import asyncio
import os
import json
import argparse
from scrapingbee import ScrapingBeeClient # Importing SPB's client

API_KEY = ''
MAX_RETRIES = 5 # Setting the maximum number of retries if we have failed requests to 5.
MAX_THREADS = 5

main_queue = asyncio.Queue()
parsed_links_queue = asyncio.Queue()
parsed_links = set()
visited_urls = 0
filename_to_link_map = dict()

#argParser = argparse.ArgumentParser()
#argParser.add_argument("-n", "--company_name", help="Company name")
#argParser.add_argument("-u", "--company_url", help="Company url")
#args = argParser.parse_args()



company_name = ''
folder_name = ''
seed_url = ''
start_url = ''

print(company_name)
print(seed_url)

def get_url(session,url):
    global visited_urls
    result = {}
    for _ in range(MAX_RETRIES):
        response = session.get(url, params={'wait':'10000','extract_rules':{'text':'body','all_links':{"selector": "a","type": "list","output": "@href"}}})
        if response.ok:
            visited_urls += 1
            result = json.loads(response.content.decode('utf-8'))
            break
        else:
            result['text'] = "error"
            result['all_links'] = []
            print(response)
            print(response.content)
    return result

async def worker(session,links):
    while True:
        url = await main_queue.get()
        filename = url[8:].replace("/", "_")
        if len(folder_name+filename+".txt") > 255:
            filename = filename[0:(255-len(folder_name)-len(".txt"))]
        filename = filename + ".txt"
        if not os.path.isfile(folder_name+filename) and '.pdf' not in url and '.xls' not in url and '.doc' not in url and '.png' not in url and '.docx' not in url and '.pptx' not in url and '.jpg' not in url and '.ppt' not in url and '.PDF' not in url and '.eps' not in url and '.potx' not in url and '.gif' not in url and '.mp4' not in url and '.wmv' not in url:
            print(f'file not exists: {folder_name+filename}')
            result = get_url(session,url)
            with open(folder_name+filename, "w", encoding="UTF-8") as f:
                #print('url: '+ url)
                f.write(result['text'])
            with open(f'{company_name}_link_map.txt', "a") as fp:
                fp.write(f'"{folder_name+filename}": "{url}",\n')
            with open(f'{company_name}_links.txt', "a") as file:
                if url.strip() not in links:
                    file.write(url + '\n')
                    links.append(url.strip())
            for href in result['all_links']:
                print
                if href!='' and href is not None:
                    print('link: '+href)
                    if href.startswith('/') and 'help/' in href and ':' not in href:
                        parsed_links_queue.put_nowait(seed_url + href)
                        with open(f'{company_name}_links.txt', "a") as file:
                            fullUrl = seed_url + href
                            #print('fullUrl: '+ fullUrl)
                            if fullUrl.strip() not in links:
                                file.write(fullUrl + '\n')
                                links.append(fullUrl.strip())
        else:
            print(f'file exists: {folder_name+filename}')

        main_queue.task_done()

async def consumer():
    while True:
        url = await parsed_links_queue.get()

        if url not in parsed_links:
            parsed_links.add(url)
            main_queue.put_nowait(url)

        parsed_links_queue.task_done()


async def main():
    links = []


    #if not os.path.exists("text/"+local_domain+"/"):
    #        os.mkdir("text/" + local_domain + "/")
    if os.path.exists(folder_name):
        with open(f'{company_name}_links.txt', "r") as fq:
            links = fq.read().splitlines()

    session = ScrapingBeeClient(api_key=API_KEY)
    workers = {asyncio.create_task(worker(session,links)) for _ in range(16)}
    c = asyncio.create_task(consumer())

    # Create a directory to store the text files
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        parsed_links.add(start_url)
        main_queue.put_nowait(start_url)
    else:
        for link in links:
            #print(link.strip())
            main_queue.put_nowait(link.strip())

    print('Initializing...')
    await asyncio.sleep(5)

    while main_queue.qsize():
        await asyncio.sleep(0.1)

    await main_queue.join()
    await parsed_links_queue.join()

    #with open(folder_name+"link_map.txt", "w") as fp:
    #    json.dump(filename_to_link_map, fp)

asyncio.run(main())
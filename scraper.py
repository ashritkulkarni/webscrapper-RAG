import asyncio
import aiohttp
import urllib.parse
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import os
import json
import argparse

main_queue = asyncio.Queue()
parsed_links_queue = asyncio.Queue()
parsed_links = set()
visited_urls = 0
filename_to_link_map = dict()

#argParser = argparse.ArgumentParser()
#argParser.add_argument("-n", "--company_name", help="Company name")
#argParser.add_argument("-u", "--company_url", help="Company url")
#args = argParser.parse_args()

#company_name = f'{args.company_name}'
#folder_name = f'{args.company_name}_data/'
#seed_url = f'{args.company_url}'
#seed_url = ''

company_name = ''
folder_name = '/'
seed_url = ''
start_url = ''

print(company_name)
print(seed_url)

def remove_newlines(serie):
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = serie.replace('  ', ' ')
    serie = serie.replace('  ', ' ')
    return serie

async def get_url(session,url):
    global visited_urls
    proxies = {
    'http': 'http://D5PWG4UCEE5330APY7TE6V9S2IWVNU1CSW2EE4YC974R8T3GEPBE93QV66YBY00PLLVTV54T1MB47XVM:render_js=false&wait=5000@proxy.scrapingbee.com:8886',
    'https': 'https://D5PWG4UCEE5330APY7TE6V9S2IWVNU1CSW2EE4YC974R8T3GEPBE93QV66YBY00PLLVTV54T1MB47XVM:render_js=false&wait=5000@proxy.scrapingbee.com:8887',
    'socks5': 'socks5://D5PWG4UCEE5330APY7TE6V9S2IWVNU1CSW2EE4YC974R8T3GEPBE93QV66YBY00PLLVTV54T1MB47XVM:render_js=false&wait=5000@socks.scrapingbee.com:8888'}
    try:
        async with session.get(url, proxy=proxies['https']) as resp:
            visited_urls += 1
            return await resp.text()
    except aiohttp.ClientError as e:
        print(f'Bad URL: {url}, with exception: {e}')

async def worker(session,links):
    while True:
        url = await main_queue.get()
        #if "/en-US" in url or url == '':
        filename = url[8:].replace("/", "_")
        if len(folder_name+filename+".txt") > 255:
            filename = filename[0:(255-len(folder_name)-len(".txt"))]
        filename = filename + ".txt"
        if not os.path.isfile(folder_name+filename) and '.pdf' not in url and '.xls' not in url and '.doc' not in url and '.png' not in url and '.docx' not in url and '.pptx' not in url and '.jpg' not in url and '.ppt' not in url and '.PDF' not in url and '.eps' not in url and '.potx' not in url and '.gif' not in url and '.mp4' not in url and '.wmv' not in url:
            print(f'file not exists: {folder_name+filename}')
            soup = BeautifulSoup(await get_url(session,url), 'html.parser')
            with open(folder_name+filename, "w", encoding="UTF-8") as f:
                #print('url: '+ url)
                text = soup.get_text()
                f.write(remove_newlines(text))
            with open(f'{company_name}_link_map.txt', "a") as fp:
                fp.write(f'"{folder_name+filename}": "{url}",\n')
            with open(f'{company_name}_links.txt', "a") as file:
                if url.strip() not in links:
                    file.write(url + '\n')
                    links.append(url.strip())
            #filename_to_link_map[filename] = url
            for a in soup.select('a[href]'):
                href = a['href']
                if href.startswith('/') and ':' not in href:
                    parsed_links_queue.put_nowait(seed_url + href)
                    with open(f'{company_name}_links.txt', "a") as file:
                        fullUrl = seed_url + href
                        #print('fullUrl: '+ fullUrl)
                        if fullUrl.strip() not in links:
                            file.write(fullUrl + '\n')
                            links.append(fullUrl.strip())
        else:
            print(f'file exists: {folder_name+filename}')
        #else:
        #    print(f'foreign url: {url}')

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

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        workers = {asyncio.create_task(worker(session,links)) for _ in range(5)}
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
# pip install requests
import requests

url = ""
proxy = ""
proxies = {"http": proxy, "https": proxy}
response = requests.get(url, proxies=proxies, verify=False)
print(response.text)
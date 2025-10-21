# search_web/serper_client.py
import os, requests
API_KEY = os.environ.get("SERPER_API_KEY")  # set this in your environment

def serper_search(query: str, top_k=5):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "gl": "us", "num": top_k}
    r = requests.post(url, json=payload, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    # convert to standardized format
    results = []
    for item in data.get("organic", [])[:top_k]:
        results.append({"title": item.get("title"), "snippet": item.get("snippet"), "link": item.get("link")})
    return results

import webbrowser
from urllib.parse import quote_plus


def search_web(query: str, dry_run: bool = True) -> str:
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    if not dry_run:
        webbrowser.open(url)
    return f"Would search for {query}" if dry_run else f"Searching for {query}"

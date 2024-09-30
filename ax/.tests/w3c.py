import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from rich.progress import Progress
from rich.table import Table
from rich.console import Console
from lxml import etree

console = Console()

def fetch_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        console.print(f"[red]Error fetching {url}: {e}[/red]")
        return None

def extract_links(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for link in soup.find_all('a', href=True):
        absolute_url = urljoin(base_url, link['href'])
        links.add(absolute_url)
    return links

def check_link(url):
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        return False

def validate_html(html):
    parser = etree.HTMLParser()
    try:
        etree.fromstring(html, parser)
        return []
    except etree.XMLSyntaxError as e:
        return e.error_log

def main():
    parser = argparse.ArgumentParser(description="Check all links and HTML syntax on a website.")
    parser.add_argument('url', help="The base URL to start the check.")
    args = parser.parse_args()

    base_url = args.url
    html = fetch_page(base_url)
    
    if html is None:
        return

    links = extract_links(html, base_url)

    console.print(f"Found {len(links)} links on {base_url}")
    
    invalid_links = []
    with Progress() as progress:
        task = progress.add_task("Checking links...", total=len(links))
        for link in links:
            if not check_link(link):
                invalid_links.append(link)
            progress.advance(task)

    console.print("[bold]HTML Syntax Errors:[/bold]")
    errors = validate_html(html)
    if errors:
        table = Table(title="HTML Errors")
        table.add_column("Line", justify="right")
        table.add_column("Error", justify="left")
        for error in errors:
            table.add_row(str(error.line), error.message)
        console.print(table)
    else:
        console.print("[green]No HTML syntax errors found![/green]")

    if invalid_links:
        console.print("[bold red]Invalid Links:[/bold red]")
        for link in invalid_links:
            console.print(f"[red] {link} [/red]")
    else:
        console.print("[green]All links are valid![/green]")

if __name__ == "__main__":
    main()


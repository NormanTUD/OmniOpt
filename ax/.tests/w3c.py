import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from rich.progress import Progress
from rich.table import Table
from rich.console import Console
from lxml import etree
import html5lib

console = Console()

def fetch_page(url):
    """Fetch the HTML content of a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        console.print(f"[red]Error fetching {url}: {e}[/red]")
        return None

def extract_links(html, base_url):
    """Extract all unique links from the HTML content."""
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for link in soup.find_all('a', href=True):
        absolute_url = urljoin(base_url, link['href'])
        links.add(absolute_url)
    return links

def check_link(url):
    """Check if a link is reachable (returns status code 200)."""
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        return False

def validate_html(html):
    """Validate the HTML using html5lib and lxml."""
    try:
        parser = html5lib.HTMLParser(strict=True)
        document = parser.parse(html)
        return [], document
    except Exception as e:
        return [str(e)], None

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Check all links and HTML syntax on a website.")
    parser.add_argument('url', help="The base URL to start the check.")
    args = parser.parse_args()

    base_url = args.url
    html = fetch_page(base_url)
    
    if html is None:
        return

    # Extract links
    links = extract_links(html, base_url)
    console.print(f"Found {len(links)} links on {base_url}")

    # Check each link with progress bar
    invalid_links = []
    with Progress() as progress:
        task = progress.add_task("Checking links...", total=len(links))
        for link in links:
            if not check_link(link):
                invalid_links.append(link)
            progress.advance(task)

    # Validate the HTML syntax
    console.print("[bold]HTML Syntax Errors:[/bold]")
    errors, document = validate_html(html)
    if errors:
        table = Table(title="HTML Errors")
        table.add_column("Error", justify="left")
        for error in errors:
            table.add_row(error)
        console.print(table)
    else:
        console.print("[green]No major HTML syntax errors found![/green]")

    # Display invalid links
    if invalid_links:
        console.print("[bold red]Invalid Links:[/bold red]")
        for link in invalid_links:
            console.print(f"[red] {link} [/red]")
    else:
        console.print("[green]All links are valid![/green]")

if __name__ == "__main__":
    main()

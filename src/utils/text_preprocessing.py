from bs4 import BeautifulSoup


def clean_html(html_doc):
    """Clean html."""
    soup = BeautifulSoup(html_doc, 'html.parser')
    return soup.get_text()

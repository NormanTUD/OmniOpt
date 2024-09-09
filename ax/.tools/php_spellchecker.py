from bs4 import BeautifulSoup
import sys
import re
from spellchecker import SpellChecker
import emoji

# Liste von Regex-Mustern, die ignoriert werden sollen (z.B. technische Begriffe, Abkürzungen, usw.)
IGNORED_PATTERNS = [
    r'linux',
    r'slurm',
    r'dark-mode.',
    r'python3',
    r'botorch',
    r'botorch-modular',
    r'omniopt2',
    r'slurm',
    r'omniopt2',
    r'\d+',
    r'https?://\S+',
    r'\b[A-Z]{2,}\b',
    r'\b[A-Za-z0-9_-]+@[A-Za-z0-9._-]+\.[A-Za-z]{2,}\b'
]

def extract_visible_text_from_html(html_content):
    try:
        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()

        # Extract the visible text
        visible_text = soup.get_text(separator='\n')

        # Clean up unnecessary whitespace and empty lines
        clean_text = "\n".join([line.strip() for line in visible_text.splitlines() if line.strip()])
        return clean_text
    except Exception as e:
        print(f"Error processing HTML content: {e}")
        return None

def clean_word(word):
    # Remove punctuation and split hyphenated words
    word = re.sub(r'[^\w\s-]', '', word)  # Remove punctuation except hyphen
    return word.split('-')  # Split on hyphens to check each part separately

def filter_emojis(text):
    # Remove emojis and other non-alphanumeric characters
    return ''.join(char for char in text if not emoji.is_emoji(char))

def check_spelling(text):
    try:
        # Initialize the spell checker with the American English dictionary
        spell = SpellChecker(language='en')

        # Split the text into words
        words = text.split()

        # Filter out words that match any of the ignored patterns or contain emojis
        filtered_words = []
        for word in words:
            cleaned_word_parts = clean_word(word)
            for part in cleaned_word_parts:
                part_no_emoji = filter_emojis(part)
                if part_no_emoji and not any(re.match(pattern, part_no_emoji) for pattern in IGNORED_PATTERNS):
                    filtered_words.append(part_no_emoji)

        # Find words that are misspelled
        misspelled = spell.unknown(filtered_words)

        return misspelled
    except Exception as e:
        print(f"Error checking spelling: {e}")
        return None

if __name__ == "__main__":
    # Read HTML input from stdin (e.g., from PHP output)
    html_content = sys.stdin.read()

    # Extract the visible text from HTML content
    extracted_text = extract_visible_text_from_html(html_content)

    if extracted_text:
        # Perform spell check on the extracted text
        misspelled_words = check_spelling(extracted_text)

        if misspelled_words:
            print("Misspelled words:")
            print(", ".join(misspelled_words))
            sys.exit(len(misspelled_words))  # Exit with the number of misspelled words
        else:
            print("\nNo misspelled words found.")
            sys.exit(0)  # Exit with 0 if no misspellings were found
    else:
        print("No text was extracted.")
        sys.exit(1)  # Exit with 1 if no text was extracted


import re


def clean_text(text):
    """
    A function to clean text by removing special characters and unnecessary whitespace
    """
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove multiple whitespaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text

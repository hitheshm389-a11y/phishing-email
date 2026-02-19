# Helpful utilities for model inference and evaluation (kept simple)
import re

def find_urls(text):
    return re.findall(r'https?://[^\s]+', text)

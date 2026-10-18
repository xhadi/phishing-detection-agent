"""Text cleaning and feature extraction shared by train_model.ipynb and app.py.

Training and inference MUST go through these functions so the model always
sees input prepared the same way.

Design notes (see dataset/README.md for the evidence):
- Email: URLs and email addresses are removed, not counted. The Nazario phishing
  bodies had their links stripped by the dataset author while safe SpamAssassin
  mail kept them, so URL features would teach the model "has a link = safe".
- SMS: links are intact in every SMS source and are a real signal, so they are
  replaced with a placeholder token and counted. SMS scams often omit "http://"
  (e.g. 'wel01.us/r/rest05'), so bare domains with common TLDs count as links.
- Punctuation and case are dropped because Enron text is pre-tokenized and
  lowercase while the other sources are raw; otherwise style reveals the source.
- Message length is not a feature (it mostly separated SMS from email in v1).
"""
import re

import numpy as np

EMBEDDER_NAME = 'all-MiniLM-L6-v2'
CHANNELS = ('email', 'sms')

# TLDs seen in SmishTank links plus other common ones; limits false matches like 'e.g.' or '3.5'
_TLDS = (
    'com|net|org|info|biz|us|uk|co|io|me|ly|to|in|id|de|gy|gd|cc|ws|ru|cn|tk|ml|ga|cf|gq|'
    'top|xyz|shop|shopping|store|link|online|site|club|app|live|vip|icu|pro|sbs|cfd|bond|fun|'
    'win|click|life|world|today|support|homes|financial|network|directory|boutique|services'
)
URL_PATTERN = re.compile(
    r'(?:https?://|www\.)\S+'
    r'|\b(?:[a-z0-9-]+\.)+(?:' + _TLDS + r')\b(?:/\S*)?',
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
URL_TOKEN = 'urllink'

# Words that only identify which corpus a message came from (Enron staff/offices, Nazario's mailbox)
CORPUS_ARTIFACT_TOKENS = {'enron', 'ect', 'hou', 'vince', 'kaminski', 'jose', 'monkey'}

URGENCY_KEYWORDS = [
    'urgent', 'suspend', 'suspended', 'decline', 'hold', 'alert', 'verify', 'update',
    'immediate', 'immediately', 'restricted', 'validate', 'restore', 'expire', 'expires',
    'locked', 'confirm', 'unusual', 'within 24 hours',
]
URGENCY_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, URGENCY_KEYWORDS)) + r')\b', re.IGNORECASE)

STRUCTURAL_FEATURE_NAMES = {
    'email': ['urgency_count'],
    'sms': ['urgency_count', 'has_url', 'url_count'],
}


def _check_channel(channel):
    if channel not in CHANNELS:
        raise ValueError(f'channel must be one of {CHANNELS}, got {channel!r}')


def detokenize(text):
    """Rejoin URLs/emails split by pre-tokenized corpora, e.g. Enron's 'http : / / bin . site . com'.

    Without this, Enron URLs would survive cleaning as ordinary words.
    """
    return re.sub(r'(?<=\S) ([./:@_-]) (?=\S)', r'\1', text)


def normalize_text(text, channel):
    """Detokenize, strip HTML/emails, handle URLs per channel, lowercase, keep only words and numbers."""
    _check_channel(channel)
    text = detokenize(str(text))
    text = HTML_TAG_PATTERN.sub(' ', text)
    text = EMAIL_PATTERN.sub(' ', text)  # before URLs, so 'name@site.com' is not read as a link
    text = URL_PATTERN.sub(f' {URL_TOKEN} ' if channel == 'sms' else ' ', text)
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return ' '.join(t for t in tokens if t not in CORPUS_ARTIFACT_TOKENS)


def build_email_text(subject, body):
    """Combine the fields the app collects for an email. Sender is not used: Enron has no sender column."""
    subject = '' if subject is None or subject != subject else str(subject)  # handle None and NaN
    body = '' if body is None or body != body else str(body)
    return f'{subject}\n{body}'.strip()


def structural_features(raw_text, channel):
    clean = normalize_text(raw_text, channel)
    features = [min(len(URGENCY_PATTERN.findall(clean)), 10)]
    if channel == 'sms':
        url_count = clean.split().count(URL_TOKEN)
        features += [int(url_count > 0), min(url_count, 5)]
    return features


def build_features(raw_texts, embedder, channel, batch_size=64, show_progress_bar=False):
    """Return the model input matrix: [384 semantic dims | structural features for the channel]."""
    _check_channel(channel)
    raw_texts = list(raw_texts)
    vectors = embedder.encode(
        [normalize_text(t, channel) for t in raw_texts],
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
    )
    structural = np.array([structural_features(t, channel) for t in raw_texts], dtype=np.float32)
    return np.hstack((vectors, structural))

"""Convert an mbox export (Gmail Takeout, Nazario's corpus) into a training CSV with label, subject, body columns.

Example, for your own inbox (all safe):
    python mbox_to_csv.py dataset/Inbox.mbox dataset/emails/custom.csv --label 0 \
        --holdout dataset/emails/custom_holdout.csv --redact "Jane" "Doe"

- Bodies are plain text: the text/plain part when there is one, otherwise the HTML with style/script removed.
- Email addresses, links, phone numbers and the --redact words are removed, so the CSV holds less personal data.
- Only English messages are kept: the cleaning in phishing_features.py keeps only a-z and 0-9, so other scripts vanish
  and other languages give the English embedder little to work with.
- At most --max-per-sender messages per sender domain, picked at random, so one newsletter can't teach "this style = safe".
- With --holdout, a share of sender domains goes to a separate file. Those senders are never in training, which makes
  the file a fair check of false positives (train_model.ipynb section 9 scores it when it exists).
"""
import argparse
import csv
import email.utils
import html
import mailbox
import random
import re
from collections import defaultdict
from email.header import decode_header, make_header

MAX_BODY_CHARS = 4000  # the embedder reads ~200 words; the rest only matters to the urgency count

ENGLISH_STOPWORDS = set(
    'the a an and or of to in on for with is are was be your you we our this that it at by from as have has will can '
    'if not no do any all more please here about'.split()
)
MIN_ENGLISH_RATIO = 0.12  # share of words that are common English stopwords
MIN_WORDS = 6  # short one-time-code emails matter: the models flag them

HIDDEN_BLOCKS = re.compile(r'<(script|style|head|title)\b.*?</\1\s*>', re.IGNORECASE | re.DOTALL)
BLOCK_TAGS = re.compile(r'<\s*(br|/p|/div|/tr|/li|/h\d|/table)\b[^>]*>', re.IGNORECASE)
TAGS = re.compile(r'<[^>]+>')
URLS = re.compile(r'(?:https?://|www\.)\S*', re.IGNORECASE)
EMAILS = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
PHONES = re.compile(r'\+?\d[\d\s().-]{7,}\d')
SPACES = re.compile(r'[ \t ​-‏͏﻿]+')


def html_to_text(markup):
    text = HIDDEN_BLOCKS.sub(' ', markup)
    text = BLOCK_TAGS.sub('\n', text)
    return html.unescape(TAGS.sub(' ', text))


def decode_part(part):
    payload = part.get_payload(decode=True) or b''
    try:
        return payload.decode(part.get_content_charset() or 'utf-8', errors='replace')
    except LookupError:  # unknown charset name
        return payload.decode('utf-8', errors='replace')


def message_body(msg):
    """Text of the first text/plain or text/html part, whichever holds more text.

    Some senders put only "View in browser" in the plain part, and some put HTML in it, so both go through html_to_text.
    """
    parts = {}
    for part in msg.walk():
        if part.is_multipart() or part.get_content_disposition() == 'attachment':
            continue
        if part.get_content_type() in ('text/plain', 'text/html'):
            parts.setdefault(part.get_content_type(), html_to_text(decode_part(part)))
    return max(parts.values(), key=lambda text: len(text.split()), default='')


def header_text(value):
    try:
        return str(make_header(decode_header(str(value or ''))))
    except Exception:  # malformed encoded header
        return str(value or '')


def scrub(text, redact_pattern):
    text = URLS.sub(' ', text)
    text = EMAILS.sub(' ', text)
    text = PHONES.sub(' ', text)
    if redact_pattern:
        text = redact_pattern.sub(' ', text)
    text = SPACES.sub(' ', text)
    return re.sub(r'\s*\n\s*', '\n', text).strip()


def is_english(text):
    words = re.findall(r'[a-z]+', text.lower())
    return len(words) >= MIN_WORDS and sum(w in ENGLISH_STOPWORDS for w in words) / len(words) >= MIN_ENGLISH_RATIO


def sender_domain(msg):
    address = email.utils.parseaddr(str(msg.get('From', '')))[1].lower()
    domain = address.rsplit('@', 1)[-1]
    return '.'.join(domain.split('.')[-2:])  # email.playstation.com and txn-email.playstation.com are one sender


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mbox')
    parser.add_argument('output')
    parser.add_argument('--label', type=int, choices=(0, 1), required=True, help='0 = safe, 1 = threat')
    parser.add_argument('--max-per-sender', type=int, default=15)
    parser.add_argument('--holdout', help='CSV for the held-out sender domains')
    parser.add_argument('--holdout-fraction', type=float, default=0.2, help='share of sender domains held out')
    parser.add_argument('--redact', nargs='*', default=[],
                        help='words to remove, e.g. your first and last name (words starting with them go too)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Words starting with a redacted word go too, so usernames like 'jane42' are removed
    redact_pattern = (
        re.compile(r'\b(?:' + '|'.join(map(re.escape, args.redact)) + r')\w*', re.IGNORECASE) if args.redact else None
    )
    by_domain = defaultdict(list)
    seen = set()
    counts = defaultdict(int)
    for msg in mailbox.mbox(args.mbox, create=False):
        counts['messages'] += 1
        subject = scrub(header_text(msg['Subject']), redact_pattern).replace('\n', ' ')
        body = scrub(message_body(msg), redact_pattern)[:MAX_BODY_CHARS]
        if not is_english(f'{subject}\n{body}'):
            counts['not English or too short'] += 1
            continue
        if (subject, body) in seen:
            counts['exact duplicates'] += 1
            continue
        seen.add((subject, body))
        by_domain[sender_domain(msg)].append((args.label, subject, body))

    rng = random.Random(args.seed)
    domains = sorted(by_domain)
    rng.shuffle(domains)
    n_holdout = round(len(domains) * args.holdout_fraction) if args.holdout else 0
    splits = {'train': domains[n_holdout:], 'holdout': domains[:n_holdout]}
    for split, path in [('train', args.output), ('holdout', args.holdout)]:
        if not path:
            continue
        rows = []
        for domain in splits[split]:
            messages = by_domain[domain]
            rows += rng.sample(messages, min(len(messages), args.max_per_sender))
            counts[f'{split}: capped per sender'] += max(0, len(messages) - args.max_per_sender)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'subject', 'body'])
            writer.writerows(rows)
        print(f'{split}: {len(rows)} messages from {len(splits[split])} sender domains -> {path}')
    for name, n in counts.items():
        print(f'  {name}: {n}')


if __name__ == '__main__':
    main()

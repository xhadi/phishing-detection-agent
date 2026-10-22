# Datasets

The downloaded CSV files are not committed to git (see `.gitignore`). Download them into the folders below. Only `synthetic/` is committed.

```text
dataset/
├── emails/                    # Kaggle "Phishing Email Dataset" (Naser Abdullah Alam)
│   ├── Enron.csv
│   ├── Nazario.csv
│   ├── Nigerian_Fraud.csv
│   ├── SpamAssasin.csv        # (sic) original Kaggle file name
│   └── custom.csv             # optional: your own messages (never committed)
├── sms/
│   ├── phishing_sms.csv       # Mishra & Soni, Mendeley 2022
│   ├── SmishTank_dataset.csv  # SmishTank.com, user-submitted smishing 2021–2022
│   ├── custom.csv             # optional: your own messages (never committed)
│   ├── spam_sms.csv           # UCI SMS Spam Collection (NOT used, see below)
│   └── sms_from_LLMs.csv      # Balanced spam/smishing set, Mendeley 2025 (NOT used, see below)
├── synthetic/                 # committed: synthetic modern messages, both labels
│   ├── emails.csv
│   └── sms.csv
└── legacy/
    └── old_datasets_v1.zip    # Data used by the first model (do not use for training)
```

## Modern messages: synthetic and custom sources

All the downloaded sources are old (2000–2022), and none has modern safe messages. The v3 models learned that brands, orders, codes and links mean "threat". Two extra sources fill the gap. The notebook loads them first, so they win when duplicates are removed:

| File | Columns | Contents |
|---|---|---|
| `synthetic/emails.csv` | label, category, subject, body | 114 safe (orders, receipts, security notices, bank, travel, work tools, newsletters, personal) and 45 threats (CEO/gift-card fraud, bank-detail changes, callback scams, credential phishing, advance-fee, job and investment scams) |
| `synthetic/sms.csv` | label, category, text | 130 safe (one-time codes, delivery notices, most with links, appointments, bank alerts, personal chats) and 50 threats ("Hi mum" impersonation, boss gift cards, fraud call-backs, toll and delivery smishing) |
| `emails/custom.csv` | label, subject, body | Optional. Your own messages, for example exported from your inbox with personal data removed |
| `sms/custom.csv` | label, text | Optional. Same, for SMS |

Labels: `0` = safe, `1` = threat. Each label in a custom file needs either 20 or more rows or none at all; otherwise the stratified split stops with an error.

### From a Gmail export

`mbox_to_csv.py` turns an `.mbox` file (Gmail Takeout, or Nazario's yearly mailboxes with `--label 1`) into this format:

```bash
python mbox_to_csv.py dataset/Inbox.mbox dataset/emails/custom.csv --label 0 \
    --holdout dataset/emails/custom_holdout.csv --redact <first name> <last name>
```

It keeps English messages only, removes email addresses, links, phone numbers and the `--redact` words (and words starting with them, such as usernames), and keeps at most 15 messages per sender domain. With `--holdout`, 20% of sender domains go to a separate file that is never trained on. The notebook scores it as "own inbox, held-out senders" and prints only totals.

Everything in a Gmail inbox is labelled safe, so check that no phishing ended up in your inbox. `.mbox` files are git-ignored like the CSVs. You don't need to upload the `.mbox` to Drive, only the CSVs.

Caveats:
- The synthetic rows are written by an AI model, not collected. Both labels share one writing style, so the model can't use style alone to separate them. Still, check the per-source table: near-perfect scores on `Synthetic` with no gain in the reality check mean the model learned the style, not the content. Real messages in `custom.csv` are better whenever you can collect them.
- The synthetic rows use different brands, names and wording from `reality_check/`, so the reality check stays a fair test. The closest pair has cosine similarity 0.74 (email) and 0.81 (SMS), against the 0.90 near-duplicate threshold. The notebook prints a warning if a reality-check message comes too close to any training row.

## Emails

Source: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

| File | Columns | Rows | Labels | Use as |
|---|---|---|---|---|
| `Nazario.csv` | sender, receiver, date, subject, body, urls, label | 1,565 | all `1` | **Threat** — real credential phishing, 2015–2022 |
| `Nigerian_Fraud.csv` | sender, receiver, date, subject, body, urls, label | 3,332 | all `1` | **Threat** — advance-fee fraud, mostly 2003–2007 |
| `SpamAssasin.csv` | sender, receiver, date, subject, body, label, urls | 5,809 | `0`: 4,091 / `1`: 1,718 | **Safe** = label `0` only. Label `1` is generic spam, not phishing |
| `Enron.csv` | subject, body, label | 29,767 | `0`: 15,791 / `1`: 13,976 | **Safe** = label `0` only. Label `1` is generic spam, not phishing |

Caveats:
- `Enron.csv` has **no sender column**, and its text is pre-processed: 100% lowercase with spaces around punctuation (`re : nom / actual vols`). The other files are raw text. Normalize all sources the same way (lowercase, collapse spacing around punctuation), or the model learns "Enron style = safe" instead of phishing.
- Safe emails (~19.9k) outnumber threats (~4.9k). Down-sample Enron or use class weights.
- The first row of `Nazario.csv` is a mailbox system message ("DON'T DELETE THIS MESSAGE"), not phishing. Drop it.

## SMS

| File | Columns | Rows | Labels | Use as |
|---|---|---|---|---|
| `phishing_sms.csv` | LABEL, TEXT | 5,971 | `ham` 4,844 / `Smishing` 1,105 / `smishing` 22 | Labels are mixed case, so lowercase them before mapping |
| `SmishTank_dataset.csv` (latin-1) | message text in `MainText`, plus Sender, Url, Brand, Message Categories, VirusTotal counts | 1,062 | all smishing | **Threat**. The only modern smishing source (delivery fees, locked accounts, romance scams) |
| `spam_sms.csv` | v1, v2 (latin-1) | 5,569 | `ham` 4,822 / `spam` 747 | **Not used** |
| `sms_from_LLMs.csv` | LABEL, TEXT, URL, EMAIL, PHONE | 10,191 | `ham` / `smishing` / `spam` (3,397 each) | **Not used** |

Sources:
- https://data.mendeley.com/datasets/f45bkkt8pr (phishing_sms)
- https://arxiv.org/abs/2402.18430 (SmishTank, Timko & Rahman 2024)
- https://data.mendeley.com/datasets/vmg875v4xs (sms_from_LLMs)

Caveats:
- **`spam_sms.csv` is not used.** `phishing_sms.csv` already contains almost all of its messages, so only ~90 of its rows survived deduplication. It was dropped in v4 to simplify the notebook.
- **`sms_from_LLMs.csv` is not used.** Every one of its safe rows is an exact copy of `phishing_sms.csv`, and its threats are word-swapped copies of old UK spam ("we *sample* to contact you", similarity ≥ 0.9 to the original for 87% of rows). It adds no new information and lets copies of the same message straddle train and test.
- **SmishTank:** do not filter by the VirusTotal columns (`Detected`, `Malicious`, ...). Many obvious scams have 0 detections because the link was checked after it went offline. About a third of its links have no `http://` (e.g. `wel01.us/r/rest05`), so URL detection must match bare domains.
- SMS links are a genuine signal (in the text of 21% of Mishra threats and 86% of SmishTank, against 0.3% of safe SMS), unlike email.
- Real SMS campaigns repeat with small changes (a different phone number or name): half of the real threats have a near-twin with embedding similarity ≥ 0.93. The notebook removes near-duplicates at 0.90 after embedding.
- **No downloaded source has modern safe SMS** (one-time codes, delivery notifications, appointment reminders). Only `synthetic/sms.csv` and your `custom.csv` provide them.

## Rules for the training notebook

1. Deduplicate before splitting, both exact and near-duplicates. The v1 model's test set was 42% exact copies of training rows.
2. Keep a hand-written test set of modern messages, separate from these sources, as the real check. It lives in `reality_check/` (60 emails and 60 SMS, half threats) and is never used for training.

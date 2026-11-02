# A Comparative Textual Analysis of Low-Rated Reviews on Amazon

CA1 project for Text Analytics and Natural Language Processing
| Student name: Vanshika Sharma (x23198389)

This project asks whether bad reviews (1–2 stars) of **well-liked** products are written differently from bad reviews of **disliked** products. A product is well-liked if the reviews written **before** a bad review average 4 stars or more, and disliked if they average 2.5 stars or less. Only earlier reviews are used, because those are the only ones the reviewer could have seen.

- **Data:** Amazon Reviews 2023, All Beauty category (McAuley Lab, UC San Diego)
- **Groups:** 16,816 bad reviews of well-liked products vs 2,881 bad reviews of disliked products (19,697 in total)
- **Methods:** 10 writing features, regression with controls, Holm correction, Cohen's d, TF-IDF classifiers, LDA topic modelling and five robustness checks
- **Process:** KDD (Knowledge Discovery in Databases)

The thinking behind each main choice is in [`DECISION_LOG.md`](DECISION_LOG.md).

---

## 1. Folder structure

```
.
├── 01_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_wordlists.ipynb
├── 03a_ai_mentions_reasons.ipynb
├── 04_features.ipynb
├── 04a_ai_labelling.ipynb
├── 05_analysis.ipynb
├── 06_baseline.ipynb
├── 07_robustness.ipynb
├── 08_topics.ipynb
├── data/
│   ├── raw/              (the two Amazon files (compressed .gz, unzip first))
│   └── processed/        (created by the notebooks (only the split file is shared))
├── lexicons/             (the two self-created phrase lists (+ NRC file, downloaded separately))
├── ai_labels/            (prompts and AI word-check decisions)
├── manual_checks/        (hand-labelled checks)
├── results/              (tables (.csv) and figures/ created by the notebooks)
├── requirements.txt
├── README.md
└── DECISION_LOG.md
```

## 2. What each notebook does

| Notebook | What it does | Main output |
|---|---|---|
| `01_eda.ipynb` | Explores both raw files: ratings, review length, helpful votes, missing prices | figures in `results/figures/` |
| `02_preprocessing.ipynb` | Removes duplicates, cleans HTML, keeps one review per person per product, calculates the earlier-review average with a running total (checked with an assertion test), joins price | `data/processed/reviews_clean.parquet` |
| `03_wordlists.ipynb` | Builds the two self-created phrase lists (mentions of other reviewers, reasons) and draws a 400-review check sample | `lexicons/lexicon_mentions.txt`, `lexicons/lexicon_reasons.txt` |
| `03a_ai_mentions_reasons.ipynb` | Labels the 400-review sample with gpt-5.4-nano and compares it with hand labels | `ai_labels/chatgpt_api_labels.csv` |
| `04_features.ipynb` | Creates the 10 writing features (NRC, Empath, Certainty Lexicon, spaCy, self-created lists) | `data/processed/features.parquet` |
| `04a_ai_labelling.ipynb` | Checks the most frequent NRC and Empath words in real review snippets with gpt-5.4-nano and hand labels | `ai_labels/word_decisions.csv`, `ai_labels/removed_words.csv` |
| `05_analysis.ipynb` | Regression for each feature with controls and product-clustered errors, Holm correction, Cohen's d, experience and price, helpful votes (hurdle model), price check | `results/q1_*.csv`, `results/q2_*.csv`, `results/q3_*.csv`, `results/robustness_price.csv` |
| `06_baseline.ipynb` | Product-grouped 80/20 split, majority baseline, features-only models, TF-IDF + NB / SVM / LR, bootstrap confidence intervals | `data/processed/baseline_split.csv`, `results/baseline_*.csv` |
| `07_robustness.ipynb` | Repeats the main tests under five settings: original NRC lists, 10 or 20 earlier reviews, last-10 average, 50+ words | `results/robustness_*.csv` |
| `08_topics.ipynb` | LDA topic modelling (15 topics chosen by NPMI coherence), topic differences and topic controls | `results/topics_*.csv` |

**Run the notebooks in order (01 → 08).** Each one reads the files made by the one before it.

## 3. How to get the data

The two raw Amazon files are included in `data/raw/` as compressed `.gz` files. Unzip them once before running the notebooks:

```bash
cd data/raw
gunzip -k All_Beauty.jsonl.gz meta_All_Beauty.jsonl.gz
cd ../..
```

This creates `All_Beauty.jsonl` and `meta_All_Beauty.jsonl`, which the notebooks read.

Source: Amazon Reviews 2023, All Beauty category, McAuley Lab, UC San Diego (<https://amazon-reviews-2023.github.io/>). The copy is shared for non-commercial academic reuse only; please cite the dataset (below).

**NRC Emotion Lexicon:** download `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` from <https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm> and put it in `lexicons/`. It is not included here because its terms of use do not allow redistribution. The same steps and link are also in the comments at the top of `04_features.ipynb`.

The Certainty Lexicon and the spaCy model are downloaded inside the notebooks.


## 4. Setup

Python 3.13 and Jupyter Notebook were used.

```bash
# 1. create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install the libraries
pip install -r requirements.txt

# 3. install the spaCy English model
python -m spacy download en_core_web_sm

# 4. start Jupyter and run the notebooks in order
jupyter notebook
```

**Random seed:** 42 is used everywhere (splits, models, LDA), so the results can be repeated.

**OpenAI API (notebooks 03a and 04a only):** these two notebooks call the OpenAI API (gpt-5.4-nano). The API key is typed into a hidden box when the notebook runs and is **never saved** in any file. You do **not** need to run 03a or 04a to reproduce the results: their final decisions (`ai_labels/word_decisions.csv`, `ai_labels/removed_words.csv`) are already in the repository, so skip these two notebooks.

**Time:** notebooks 02 and 04 take the longest (spaCy reads every review). Saved `.parquet` files are reused if they already exist, so later runs are faster.

## 5. What is shared and what is not

| Shared | Not shared (and why) |
|---|---|
| All 10 notebooks, and the raw Amazon files (compressed, for academic reuse) | |
| `lexicons/lexicon_mentions.txt`, `lexicons/lexicon_reasons.txt` (self-created phrase lists) | NRC lexicon file (its terms of use do not allow redistribution; download link above and in the comments of notebook 04) |
| `lexicons/frequent_phrases.csv` (frequent phrases used to build the phrase lists) | |
| `ai_labels/chatgpt_api_labels.csv`, `ai_labels/chatgpt_review_labels.csv` (the AI labels) | |
| `ai_labels/word_decisions.csv`, `ai_labels/removed_words.csv` (which NRC/Empath words were kept or removed) | |
| `manual_checks/` and AI check samples (hand labels used to test the AI labels) | |
| `ai_labels/` prompt files | |
| `data/processed/baseline_split.csv` (product-grouped train/test split) | |
| `data/processed/features.parquet` and the `feat_*.parquet` files (the 10 writing features, so notebooks 05–08 run without re-running 04) | `data/processed/reviews_clean.parquet` |
| `results/` tables and figures | Saved topic model |
| `requirements.txt`, `README.md`, `DECISION_LOG.md` | |

With the shared train/test split, future models (for example the transformer models planned for the final 50% assessment) can be tested on exactly the same reviews and compared with the TF-IDF baseline.




# Decision Log

A record of the main decisions in this project and the reason for each one. The decisions follow the KDD stages.

---

## 1. Selection

| # | Decision | Why |
|---|---|---|
| 1 | Use a real dataset (Amazon Reviews 2023) and not synthetic data | The question is about how real customers write reviews, so real reviews were needed. |
| 2 | Use only the **All Beauty** category | It had enough bad reviews of both well-liked and disliked products, and the full 33-category dataset was too large to process. Personal interest in beauty products also helped with reading and checking reviews. |
| 3 | Compare **bad reviews with bad reviews** (well-liked vs disliked products) | Comparing a one-star review with all reviews mostly compares it with positive ones, so any difference could simply be because it is negative. Keeping both groups as complaints makes the comparison fair. |
| 4 | Decide each product's group using **only earlier reviews** (running total) | A reviewer cannot see reviews written later. Using the final average would mix in future information. |
| 5 | Require **at least 5 earlier reviews** | An average of fewer than five ratings is not believable (based on earlier research). |
| 6 | Bad review = 1–2 stars, 3 stars left out | Three-star reviews are a grey area between positive and negative. |
| 7 | Well-liked = earlier average of 4+ stars, disliked = 2.5 stars or less; products in between left out | Products in the middle have mixed earlier reviews, so neither group fits them. These cut-offs were set for this study. |
| 8 | Focus the study on **bad reviews only** | An earlier version also compared good reviews. Narrowing the scope made the question clearer and fitted the 6-page limit. |

## 2. Preprocessing

| # | Decision | Why |
|---|---|---|
| 9 | Remove duplicates and keep only one review per person per product | So that no person is counted twice for the same product. |
| 10 | **No stopword removal or stemming** for the writing features | Small words like "I", "not" and "because" are the features themselves. Stemming could stop words matching the word lists. |
| 11 | Remove stopwords **only for topic modelling** | Very common words hide the topics in LDA. |
| 12 | Keep only reviews with **25+ words** for the features | Short reviews give unstable percentages (one "not" in 10 words = 10% negation). 25 words balanced stable scores and keeping enough reviews. |
| 13 | Check the running-total average with an **assertion test** | To prove the average used only earlier reviews (all 410,866 rows matched). |
| 14 | Use log scale for experience, price and popularity | A few very active reviewers and very expensive products would skew the results. |

## 3. Transformation (features)

| # | Decision | Why |
|---|---|---|
| 15 | Build **LIWC-like** features from free tools | LIWC is paid software. |
| 16 | Use the **NRC Emotion Lexicon** for emotions (not VADER or Empath) | VADER only gives overall sentiment; Empath missed common words like "angry" and "disappointed". NRC covered the emotions needed. |
| 17 | Keep **Empath** for money and the **Certainty Lexicon** for certainty | These were the best available measures for those two features. |
| 18 | Use **spaCy** for negation and first-person "I" | spaCy reads the grammar of each sentence and doesn't read matching words. |
| 19 | **Check and clean the NRC and Empath lists** | Some NRC words do not fit beauty reviews (for example, "smell" is listed as an anger word). The most frequent words were checked in real review snippets, using a stopping rule so the checking effort went where most matches were. |
| 20 | Use **gpt-5.4-nano** to label word contexts, checked against hand labels | Checking thousands of snippets by hand was not possible in the time. Hand labels were used to measure how far the AI could be trusted (kappa 0.63). |
| 21 | Build two **self-created phrase lists** (mentions of other reviewers, reasons) | No existing lexicon measures these. |
| 22 | Use a **stratified 400-review sample** to check the phrase lists | Mention phrases are rare, so a random sample would contain too few of them. |
| 23 | Treat the **reason list as exploratory** | Agreement for reasons was low (kappa 0.38), so it is not presented as reliable. |
| 24 | Keep both the **original and cleaned** NRC scores | So the robustness check can test whether cleaning changed the results. |
| 25 | Leave certainty **empty (not zero)** when no certainty phrase is found | The certainty score is an average of the phrases found, so it cannot be calculated without any phrases. These reviews are left out of the certainty model only. |

## 4. Data mining

| # | Decision | Why |
|---|---|---|
| 26 | Move from t-tests / Mann-Whitney to **regression with controls** | Simple tests could not account for star rating, popularity, review age and verified purchase. Negative emotion looked significant in simple tests but not after controls. |
| 27 | **Cluster standard errors by product** | Reviews of the same product are not independent. |
| 28 | Check predictors with **Spearman correlation and VIF** | Overlapping predictors make regression unreliable; "repeat_reviewer" was removed because it overlapped with "log_experience". |
| 29 | Logistic regression for experience and price; **hurdle model** for helpful votes | Group membership is yes/no; most reviews get no votes, and vote counts vary more than a Poisson model allows. |
| 30 | **Product-grouped** 80/20 train-test split | So the same product never appears in both training and test data. |
| 31 | Start with a **majority-class baseline**, then features-only, then TF-IDF models | To see what each step adds. |
| 32 | **TF-IDF** with unigrams and bigrams (not bag-of-words, no trigrams) | TF-IDF lowers the weight of very common words; trigrams did not improve results. |
| 33 | Add **LDA topic modelling** | The strongest TF-IDF words were product words, so the groups were partly about different products. LDA tested this. |

## 5. Evaluation

| # | Decision | Why |
|---|---|---|
| 34 | **Holm correction** across the 10 tests | Many tests increase the chance of a false positive. |
| 35 | Report **Cohen's d** (adjusted difference / pooled SD) | With thousands of reviews almost any difference is significant, so the size of the difference matters more. |
| 36 | Use **macro-F1 and PR-AUC**, not accuracy | The groups are imbalanced (85/15), so the majority baseline already gets 86% accuracy. |
| 37 | Choose the number of topics by **NPMI coherence** (5–20 tried, 15 chosen) | A standard way to pick topics that make sense. |
| 38 | Repeat the main tests under **five robustness settings** and with **topic controls** | To check the findings do not depend on one choice. Negation and "I" held in all checks; anger and money did not. |
| 39 | Re-run the helpful-vote model **on the same priced reviews** with and without price | To separate the effect of price from the smaller sample. The odds ratio was already 1.22 before price was added, so the smaller sample explained most of the change. |

## 6. Ethics and sharing

| # | Decision | Why |
|---|---|---|
| 40 | Share a compressed copy of the raw Amazon data for academic reuse | So the notebooks can be run directly; the dataset is cited and used for non-commercial academic work only. |
| 41 | Never save the OpenAI API key | It is typed into a hidden box each time and not stored in any file. |
| 42 | Plan **BERT / RoBERTa** for the final 50% on the same split | So the transformer models can be compared fairly with this TF-IDF baseline. |


# Dataset card — sexism-detection v2

Binary label: 1 = sexist (hostility targeting women), 0 = not sexist
(includes clearly benign text, general-domain neutral text, and hateful
but non-sexist abuse — the model must separate sexism from generic hate).

| Source | Rows |
|---|---|
| agnews_rows_before | 30000 |
| agnews_rows_clean | 29979 |
| augmented_copies | 3618 |
| drop:duplicate | 32 |
| drop:few_tokens | 161 |
| drop:label_contradiction | 2 |
| drop:non_english | 341 |
| drop:non_latin | 10 |
| final:agnews_neutral | 6000 |
| final:mhs_benign | 15000 |
| final:mhs_hateful_nonsexist | 6397 |
| final:mhs_sexist | 3437 |
| hatecheck_eval_rows | 3728 |
| hatecheck_eval_sexist | 373 |
| mhs_annotator_rows | 135556 |
| mhs_bucket:mhs_benign_before | 17553 |
| mhs_bucket:mhs_benign_clean | 17321 |
| mhs_bucket:mhs_hateful_nonsexist_before | 6576 |
| mhs_bucket:mhs_hateful_nonsexist_clean | 6397 |
| mhs_bucket:mhs_sexist_before | 3549 |
| mhs_bucket:mhs_sexist_clean | 3437 |
| mhs_comments | 39565 |
| mhs_drop:ambiguous_gender_hate | 815 |
| split:test | 3083 |
| split:test_sexist | 343 |
| split:train | 28282 |
| split:train_sexist | 3387 |
| split:val | 3083 |
| split:val_sexist | 344 |

## Labeling rules
- MHS: score >= 0.5 & women-target >= 0.5 -> 1; score <= -1 -> 0;
  score >= 0.5 & women-target < 0.1 -> 0 (hateful non-sexist).
- AG News: neutral news headlines -> 0.
- HateCheck: hateful & target=women -> 1, else 0 (eval only).

## Anomaly filters
URL/mention masking, whitespace collapse, mojibake & non-Latin drop,
length/token bounds, langdetect=en, exact-duplicate drop,
cross-label contradiction drop.
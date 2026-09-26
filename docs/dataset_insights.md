# Dataset Insights and Findings

This document explains what we have learned from exploring our dataset. It is written so that anyone on the team can easily understand our data, our goals, and the challenges ahead.

## 1. Our Goal
Our primary goal is **business matching** (also known as entity resolution). We need to take a list of businesses from "Source 1" and find all records that refer to the same real-world business in "Source 2" and "Source 3". Since the data comes from different places, the names and addresses might be spelled differently, formatted weirdly, or contain errors, making this a complex matching puzzle.

## 2. Dataset Size
We are working with a massive amount of data. Here are the exact number of records in each of our six source files:

| File Name | Record Count |
| :--- | :--- |
| **train_source1.tsv** | 2,206,821 |
| **train_source2.tsv** | 5,034,616 |
| **train_source3.tsv** | 5,285,603 |
| **test_source1.tsv** | 1,732,544 |
| **test_source2.tsv** | 4,887,273 |
| **test_source3.tsv** | 5,082,316 |

## 3. Countries in the Data
The dataset contains businesses from specific countries, and there is a very important difference between the training data we use to teach our model, and the test data we are evaluated on.

| File Name | US | India | France |
| :--- | :--- | :--- | :--- |
| **train_source1.tsv** | 1,323,633 | 883,188 | 0 |
| **train_source2.tsv** | 3,016,817 | 2,017,799 | 0 |
| **train_source3.tsv** | 3,170,056 | 2,115,547 | 0 |
| **test_source1.tsv** | 663,106 | 809,986 | 259,452 |
| **test_source2.tsv** | 1,871,330 | 2,312,565 | 703,378 |
| **test_source3.tsv** | 1,945,701 | 2,405,000 | 731,615 |

* **Training Data**: Only has businesses in the **US** and **India**.
* **Test Data**: Has businesses in the **US**, **India**, and **France**.

**Why this matters:** The introduction of France in the test set means our model will see French addresses (which have different formats, languages, and structures) for the very first time during testing. If we build rules that only work for US zip codes or Indian states, our model may perform poorly on French data. 

## 4. Missing Data and Quality
We checked how many rows were missing a business name or address:

* **Missing Names:** There are **0** missing business names across all files. *Important note:* Just because a name isn't blank doesn't mean it is accurate. There could still be typos, generic placeholder names, or abbreviations.
* **Missing Addresses:** 
  * Source 1 (Train & Test): **0** missing.
  * train_source2: **168,967** missing.
  * train_source3: **175,916** missing.
  * test_source2: **129,408** missing.
  * test_source3: **136,098** missing.

*Important note:* Even though Source 1 has zero blank addresses, this does not mean it is perfectly clean. We should still expect typos or formatting errors. Because Sources 2 and 3 have missing addresses, we cannot rely on the address alone to match businesses.

## 5. Ground Truth Statistics
By looking at the actual correct matches for the training set (the "ground truth"), we learned how businesses connect across sources:

* **Zero Matches:** 123,247 businesses in Source 1 had absolutely no match in the other sources. Our model needs to be smart enough to say "no match found" instead of guessing incorrectly.
* **Average Matches:** On average, a Source 1 business has **3.46** matches.
* **Maximum Matches:** One business in the training set matched to **11** other records. *Note:* This 11 is just the maximum we observed in the training data, not a strict limit. A business in the test set could theoretically have more matches.

## 6. Hardware Limitations and Next Steps
During our scan, we checked the computer's memory (RAM):
* **Total RAM:** ~5.85 GB
* **Available RAM (Snapshot):** ~0.16 GB (164 MB)

**Implications and Next Steps:**
1. **Memory is low:** The available RAM was just a snapshot in time and needs to be rechecked, but it is clear we cannot load all 24+ million rows into memory at once.
2. **Chunking is not enough:** While reading files in "chunks" (small pieces) saves memory, chunking alone does not solve matching at this scale. If we compare every chunk of Source 1 to every chunk of Source 2, it will take far too long. 
3. **Efficient Candidate Generation:** Our immediate next step must be to build a smart filtering system (often called "blocking" or "candidate generation"). We need a fast way to group similar records together so we only run our complex matching model on businesses that are somewhat similar.
4. **Tooling:** Using specialized big-data tools like Dask is optional, not mandatory. Tool choice will depend on measured memory use and runtime; disk-backed indexing and batch processing may be needed.

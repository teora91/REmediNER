# Named Entity Recognition and Relation Extraction of Drug-Related Adverse Effects from Medical Case Reports (REUPLOAD Oct. 2025)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-orange.svg)](https://huggingface.co/transformers/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)  

**Authors:**  
- Matteo Radaelli (matteo.radaelli@ntnu.no)  
- Stefano Zanoni (s.zanoni@studenti.unipit.it)  

---

## Overview
The monitoring of drug safety (**Pharmacovigilance**) plays a pivotal role in healthcare decision-making. Extracting information from medical case reports helps ensure high-quality health care and maintain the integrity of the patient–physician relationship.  

With recent advances in NLP, it is possible to automatically identify relations between drugs and their side effects, called **Adverse Drug Events (ADEs)**, which is difficult to perform manually due to patients often undergoing multiple treatments.  

This project proposes a **joint model** that provides:
- **Named Entity Recognition (NER):** identifying drugs and effects in raw text.  
- **Relation Extraction (RE):** predicting potential relations between drugs and effects.  

---

## Named Entity Recognition (NER)
NER identifies two key entity types in this project:
- **DRUG**
- **EFFECT**

### Challenges
- Entities are semantically complex and require world knowledge.  
- Open-ended label categories and unbalanced data.  
- Ambiguity and polysemy.  

### Approach
- Entities annotated with **IOB tagging** (`B-DRUG`, `I-DRUG`, `B-EFFECT`, `I-EFFECT`, `O`).  
- Model: **BERT (bert-base-cased)** fine-tuned for token classification.  
- Loss function: **weighted cross-entropy** to mitigate class imbalance.  
- Optimizer: **AdamW**.  

**Hyperparameters**
```yaml
batch_size: 8
learning_rate: 2e-4
gradient_accumulation_steps: 4
epochs: 6
````

---

## Relation Extraction (RE)

Relation Extraction links drugs to their adverse effects.

### Approach

* Entities **masked** as `DRUG` and `EFFECT` to capture syntactic patterns.
* Dataset annotated with numeric labels to align drug–effect pairs (supports one-to-many mappings).
* Model: **BERT + BiLSTM + GELU head**.
* Optimizer: **AdamW**.

**Hyperparameters**

```yaml
batch_size: 8
learning_rate: 2e-4
gradient_accumulation_steps: 4
epochs: 6
```

---

## Dataset

* **Source:** ADE corpus.
* **Size:** 4,272 sentences and 6,821 ADE relations.
* **Structure:**

  * `text`: raw medical report sentence.
  * `drug`: annotated drug span.
  * `effect`: annotated effect span.
  * `span`: character-level indices.

### Data Augmentation

* Concatenated 2–4 sentences randomly to create longer instances.
* Final augmented dataset: **12,951 sentences**.
* Introduced implicit **negative relations** (since ADE corpus only had positive pairs).

---

## Preprocessing

* Removed ambiguous cases (drug and effect with same term).
* Lowercased all entities for normalization.
* Removed irrelevant characters, quotes, statistical values.
* Cleaned spacing and formatting.

---

## Results

### NER Model

| Metric    | Score |
| --------- | ----- |
| Precision | 0.733 |
| Recall    | 0.925 |
| F1 Score  | 0.806 |

### RE Model

| Metric    | Score |
| --------- | ----- |
| Precision | 0.884 |
| Recall    | 0.874 |
| F1 Score  | 0.878 |

> Metrics are macro-averaged across classes.

---

## Extra Notes

* Attempted **POS embeddings** (via spaCy `en_core_web_sm`) for RE.
* Result: decreased performance + higher computational cost → excluded.

---

## Instructions

Clone the repository:

```bash
git clone https://github.com/StefanoZanoni/REmediNER.git
cd REmediNER
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train NER model:

```bash
python src/train_ner.py
```

Train RE model:

```bash
python src/train_re.py
```

---

## Repository Structure

```
REmediNER/
│
├── data/                # Datasets (ADE, augmented data, etc.)
├── notebooks/           # Jupyter notebooks for exploration & experiments
├── src/                 # Source code for training and evaluation
│   ├── train_ner.py     # Script to train NER model
│   ├── train_re.py      # Script to train RE model
│   └── utils.py         # Preprocessing and helper functions
│
├── requirements.txt     # Dependencies
├── LICENSE              # License file (MIT)
└── README.md            # Project documentation
```

---

## References

* ADE Corpus: [https://huggingface.co/datasets/ade_corpus](https://huggingface.co/datasets/ade_corpus)
* HuggingFace Transformers

---

## 📬 Contact

For questions, contact:

* **Stefano Zanoni** – [s.zanoni@studenti.unipit.it](mailto:s.zanoni@studenti.unipit.it)
* **Matteo Radaelli** – [matteo.radaelli@ntnu.no](mailto:matteo.radaelli@ntnu.no)

GitHub Repo shared also with: [REmediNER](https://github.com/StefanoZanoni/REmediNER)

```


  

# 🧠 Fine-Tuning BioGPT for Medical Question Answering

This project fine-tunes Microsoft's BioGPT using LoRA (PEFT) on a cleaned dataset of patient questions and doctor responses from HealthCareMagic. It includes dataset preparation, PEFT integration, Ray Tune hyperparameter optimization, and final evaluation on a test set.

---

## 📁 Project Structure

```
project_root/
│
├── data/
│   ├── train_data.json
│   ├── val_data.json
│   └── test_data.json
│
├── tokenized/
│   ├── tokenized_train/
│   ├── tokenized_val/
│   └── tokenized_test/
│
├── outputs/
│   ├── ray_model_22-43-15/
│   └── ray_model_22-52-47/
│
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── model_selection_and_setup.ipynb
│   ├── peft_finetuning_and_logging.ipynb
│   └── raytune-peft-finetune-final.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🧪 Environment Setup Instructions (1 point)

### 1. Python Version

- Python 3.10 or higher is recommended.

### 2. Install requirements

```bash
pip install -r requirements.txt
```

---

## 📓 Notebook Descriptions 

- **data_preparation.ipynb**:
    - Loads and cleans raw JSON data.
    - Tokenizes using `AutoTokenizer`.
    - Formats for language modeling by mapping prompt+response and labels.

- **model_selection_and_setup.ipynb**:
    - Loads baseline BioGPT and computes loss/perplexity for comparison.
    - Sets up model and tokenizer.

- **peft_finetuning_and_logging.ipynb**:
    - Applies LoRA tuning using PEFT.
    - Evaluates model and logs metrics.

- **raytune-peft-finetune-final.ipynb**:
    - Conducts Ray Tune hyperparameter search across learning rate and weight decay.
    - Stores best model and training logs.

---

## ✅ Evaluation & Metrics

| Model              | Loss     | Perplexity       |
|-------------------|----------|------------------|
| Baseline BioGPT   | 14.49    | ~1,959,145       |
| Fine-tuned Model  | 15.79    | ~7,216,947       |

⚠️ Fine-tuned model shows higher perplexity, possibly due to:
- Short training duration (1 epoch)
- Dataset size or class imbalance
- Need for better hyperparameter tuning

---

## 💡 Notes

- Ray Tune Trial `71499_00000` gave best results:  
  - Loss = 3.37, Perplexity ≈ 29.3

- Use `tokenized_test`, `tokenized_val`, and `tokenized_train` folders directly with `datasets.load_from_disk`.

---

## 📦 Requirements

```
transformers==4.41.1
datasets==2.19.1
peft==0.10.0
accelerate==0.31.0
bitsandbytes==0.43.1
scikit-learn
pandas
numpy
matplotlib
seaborn
evaluate
rouge-score
nltk
ray[tune]
sacremoses
```

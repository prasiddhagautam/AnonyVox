# 📂 Voice Models Directory

This directory stores the Retrieval-based Voice Conversion (RVC) model files (`.pth` weights and `.index` mapping files) used by the AnonyVox voice changer.

To keep the repository lightweight and respect model licensing/privacy, actual voice model files are ignored in Git and will not be uploaded to GitHub.

---

## 📥 How to Add Your Own Models

1. Obtain your target RVC model (you can download community models or train your own).
2. Place both the `.pth` weights file and its corresponding `.index` index file directly inside this directory.

### Example Directory Structure:
```text
models/
├── README.md                 # This instruction file
├── your_custom_voice.pth     # Model weights
└── your_custom_voice.index   # Model index (for similarity matching)
```

---

## 🔍 Fuzzy Matching Logic

AnonyVox uses a token-based match search to bind `.pth` models to their respective `.index` files automatically. 
* It first checks for an exact filename match (e.g., `model_name.pth` and `model_name.index`).
* If not found, it performs keyword matching based on tokens in the filenames (e.g., matching `vocal_model_e300.pth` to `added_IVF138_Flat_nprobe_1_vocal_model_v2.index`).
* Keep your files named similarly to ensure auto-binding works flawlessly.

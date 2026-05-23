# CC-MMD — Clean, shareable notebook (variant-switchable)

This folder contains a cleaned, variant-switchable notebook derived from the original `CC_mmd_chinese.ipynb`.

Files
- `CC_mmd_chinese_clean.ipynb` — compact, language-agnostic notebook with: setup, feature extraction (cached), model, training, dev evaluation, and multi-view consensus inference + submission generator.

Quick overview
- Variant concept: the notebook uses a variant tag (not a hard language label). Call `set_variant('china')`, `set_variant('tamil')`, or `set_variant('malayalam')` to point the notebook to the correct data folder and output paths.
- Data layout expected under the dataset root (set when calling `set_variant`):
  - `train & dev/train/train.csv` and images under `train & dev/train/`
  - `train & dev/dev/dev.csv` and images under `train & dev/dev/`
  - `test/test.csv` and images under `test/`

How to use (notebook)
1. Open `CC_mmd_chinese_clean.ipynb` in Colab / Jupyter.  
2. Edit the `set_variant(...)` call or run it with your `data_root`: `set_variant('tamil', '/content/dataset')`.  
3. Run cells top-to-bottom to extract features (cached to `features/`), train with `train_and_evaluate(...)`, then run inference cell (multi-view consensus) to produce a CSV under `submission/<variant>/`.

Multi-view consensus
 - The notebook implements a 3-view voting scheme: raw text (v1), length-filtered text (v2), and a mandatory translation view (v3). Translations are cached to reduce API calls and the notebook will fall back to the raw text if translation fails after retries; to enable translations install `deep-translator`.
- Outputs saved:
  - CSV submission: `submission/<out_path>/<team>_taska_<out_path>_original.csv`
  - Votes JSON for analysis: `submission/<out_path>/<team>_votes_<out_path>.json`

Requirements
- Python 3.8+ recommended
- Core packages (install in your environment if not already present):
  - `torch`, `torchvision` (or appropriate stable wheel for your CUDA)
  - `sentence-transformers`
  - `transformers` (optional, for CLIP image features)
  - `deep-translator` (optional, for translation view)
  - `pandas`, `numpy`, `tqdm`, `scikit-learn`, `Pillow`

Notes
- Feature extraction caches tensors in `features/` to avoid repeating expensive operations.  
- The notebook is intentionally language-agnostic: the cleaning steps remove URLs/handles/hashtags and apply light token filtering that works across scripts.  
- This is a shareable, minimal pipeline meant for reproducible experiments and teaching — adapt model sizes, batch sizes, or feature encoders to match your compute.


**Dataset layout (required)**
- **Root:** Provide a dataset root folder and call `set_variant('<variant>', '/content/dataset')` in the notebook.
- **Expected structure under the dataset root:**
  - `train & dev/train/train.csv` and image files in `train & dev/train/`
  - `train & dev/dev/dev.csv` and image files in `train & dev/dev/`
  - `test/test.csv` and image files in `test/`
- **CSV requirements:**
  - `train.csv` and `dev.csv` must contain at least the columns: `image_id`, `transcriptions`, `original_labels` (labels may be 0/1 or strings; the notebook maps `'not'` occurrences to negative examples).
  - `test.csv` must contain at least: `image_id`, `transcriptions`.
- **Image files:** filenames in the CSV `image_id` column should match files in the corresponding folder (extension may be `.jpg`, `.png`, or omitted in the CSV — the notebook will try appending `.jpg` if needed).

**Quick validation snippet** (run inside the notebook or a short script to sanity-check your layout):
```python
import os
import pandas as pd

def validate_dataset(data_root):
  paths = {
    'train': os.path.join(data_root, 'train & dev', 'train', 'train.csv'),
    'dev': os.path.join(data_root, 'train & dev', 'dev', 'dev.csv'),
    'test': os.path.join(data_root, 'test', 'test.csv'),
  }
  for split, csv_path in paths.items():
    if not os.path.exists(csv_path):
      raise FileNotFoundError(f"Missing CSV for {split}: {csv_path}")
    df = pd.read_csv(csv_path)
    required = ['image_id','transcriptions']
    if split in ('train','dev'):
      required.append('original_labels')
    for c in required:
      if c not in df.columns:
        raise ValueError(f"Missing column '{c}' in {csv_path}")
    print(f"{split}: OK — {len(df)} rows; columns: {list(df.columns)}")
  print('Dataset layout looks correct.')

# Example: validate_dataset('/content/dataset')
```

Call example: `set_variant('malayalam', '/content/dataset')` or `set_variant('tamil', '/content/dataset')` before running cells.

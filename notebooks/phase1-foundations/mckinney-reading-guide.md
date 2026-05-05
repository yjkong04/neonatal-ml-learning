# McKinney Reading Guide — Tiered Priorities & Exercises

Reference while reading *Python for Data Analysis* (3rd ed.). Reading order: **Ch 4 → 5 → 6 → 7 → 8 → 11.** Ch 10 (groupby) deferred to Phase 5.

## Speed rules

- Type code as you read — don't copy-paste, don't transcribe prose into markdown.
- One-line comments only. No paragraph-length explanations of things the book already explains.
- Time-box each section to 10–15 min. If you're stuck, mark it and move on.
- Predict the output of every code snippet before running it. The wrong prediction is where the learning is.
- Skip Tier 3 entirely on the first pass. Come back only if a downstream task forces you.

---

## Chapter 4 — NumPy Basics

### Tier 1 — must do (used daily downstream)

| # | Concept | Why it matters for NOA-AI work |
|---|---------|-------------------------------|
| 1 | Slicing & views vs. copies | ECG windowing breaks silently if you mutate a view |
| 2 | Boolean indexing | Filtering noisy samples, masking images |
| 3 | Broadcasting | Per-lead baseline correction, normalization |
| 4 | `np.where` | Vectorized thresholding (e.g. brady/tachy labels) |
| 5 | Aggregations with `axis` | Per-feature vs per-sample stats — easy to get backwards |
| 6 | Reshaping (`.reshape`, `.T`, `np.newaxis`) | CNN inputs require specific tensor shapes |
| 7 | `np.random.default_rng` | Reproducible splits, augmentation, simulated signals |

### Tier 2 — skim (know it exists)

- Universal functions list (`np.exp`, `np.log`, `np.sqrt`, `np.maximum`)
- `np.unique`, `np.in1d`
- Fancy indexing with integer arrays

### Tier 3 — skip on first pass

- Structured arrays
- File I/O with `.npy` (you'll use `torch.save` / `pd.read_csv`)
- Linear algebra section (PyTorch handles this)
- Random walks example

### Exercises (~15–20 min total)

Do these in `week01-numpy-basics.ipynb` as you hit each section. One cell per exercise. Predict the output before running.

**1. Views vs. copies**
```
Make a 1D array of 100 zeros. Take arr[10:20], set it to 99.
Print the original — confirm positions 10–19 are now 99.
Repeat with arr[10:20].copy() — confirm the original is unchanged.
```

**2. Boolean indexing**
```
Given hr = np.array([120, -5, 145, 300, 95, 0, 160]) (simulated noisy heart rates),
keep only physiologic values (40 ≤ hr ≤ 200) using a boolean mask.
```

**3. Broadcasting**
```
Make a 2D array of shape (100, 12) — 100 samples × 12 ECG leads, random values.
Subtract the per-lead mean so each lead is centered at zero. Verify .mean(axis=0) ≈ 0.
```

**4. np.where**
```
Given hr = np.array([80, 110, 150, 180, 95]),
use nested np.where to label each as 'brady' (<100), 'normal' (100–160), or 'tachy' (>160).
```

**5. Axis aggregations**
```
Given arr of shape (50, 8), predict the shape of arr.mean(axis=0) and arr.mean(axis=1) before running.
Then verify.
```

**6. Reshaping for a CNN**
```
Make a 1D array of 1000 samples. Reshape to (10, 100) — 10 windows of 100 samples.
Then add a channel dim to get shape (10, 1, 100) — the format a 1D CNN expects.
```

**7. Reproducible random**
```
Use rng = np.random.default_rng(seed=42) to generate a 1000-sample noisy sine wave:
t = np.linspace(0, 10, 1000); signal = np.sin(2*np.pi*t) + rng.normal(0, 0.1, 1000).
Re-run with the same seed — confirm it's identical.
```

---

## Chapter 5 — pandas Basics

### Tier 1 — must do (required before Ch 7)

| # | Concept | Why it matters for NOA-AI work |
|---|---------|-------------------------------|
| 1 | Series and DataFrame construction | Mental model for every pandas operation downstream |
| 2 | `.loc` (label-based) vs `.iloc` (position-based) indexing | Getting rows/columns wrong silently is a common bug — know the difference cold |
| 3 | Selecting columns, filtering rows | The daily bread of data work |
| 4 | Handling missing data basics (`NaN`, `None`) | Almost all real patient data has gaps |
| 5 | Index alignment in arithmetic | When you add two Series, pandas aligns on index — surprising if you don't know it |

### Tier 2 — skim (know it exists)

- `pd.Series` from dict — useful for building lookup tables
- `drop`, `rename`, `reset_index` — housekeeping you'll need occasionally
- `value_counts` — quick sanity check on categorical columns

### Tier 3 — skip on first pass

- `MultiIndex` (hierarchical indexing) — complex; not needed until Ch 8+
- `stack` / `unstack` — come back when you actually need it

### Exercises (~15–20 min total)

Do these in a new `week01-pandas-fundamentals.ipynb`.

**1. Build and index a vitals DataFrame**
```
Construct a DataFrame with columns ['patient_id', 'hr', 'spo2', 'temp'] and 10 rows of made-up values.
Use .loc to select rows where hr > 100. Use .iloc to select the first 3 rows and first 2 columns.
Confirm the results are what you expected.
```

**2. Index alignment**
```
Make two Series: a = pd.Series([1, 2, 3], index=['a','b','c']) and b = pd.Series([10, 20], index=['b','c']).
Add them. Predict what happens to index 'a' before running.
```

---

## Chapter 6 — Data Loading *(skim — ~20 min)*

No tiered breakdown — just know these exist:

- `pd.read_csv` and its most common options: `index_col`, `parse_dates`, `na_values`, `dtype`
- `df.to_csv` for saving processed data
- `pd.read_parquet` — faster than CSV for large datasets; you'll use this in Phase 3+

Skip everything on Excel, HDF5, JSON, and web scraping — not relevant to the plan.

**No exercises for Ch 6** — you'll get the practice organically the first time you load a real dataset.

---

## Chapter 7 — Data Cleaning and Preparation

### Tier 1 — must do (used daily downstream)

| # | Concept | Why it matters for NOA-AI work |
|---|---------|-------------------------------|
| 1 | `dropna` / `isnull` / `notnull` | Real vitals tables have gaps — you need to find and decide what to do with them |
| 2 | `fillna` (forward-fill, backfill, constant) | ECG/SpO2 dropout: filling with last known value vs. interpolation is a clinical decision |
| 3 | Detecting and filtering outliers (boolean mask + `clip`) | Artifact rejection — same pattern as Ch 4 boolean indexing, now on DataFrames |
| 4 | `pd.cut` / `pd.qcut` — discretization and binning | Turning continuous HR/SpO2 into clinical categories (brady, normal, tachy) |
| 5 | `apply` / `map` on Series and DataFrames | Per-column transformations without loops — normalizing, converting units |
| 6 | Computing indicator/dummy variables (`pd.get_dummies`) | Encoding categorical columns (sex, GA group) before feeding to a model |

### Tier 2 — skim (know it exists)

- `drop_duplicates` — patient records often have repeated rows
- `replace` — recoding values (e.g. mapping `'M'/'F'` → `0/1`)
- `rename` — cleaning up column names from raw exports
- `np.random.permutation` for row shuffling (you'll use this in train/val splits)

### Tier 3 — skip on first pass

- Regular expressions / `re` module (come back when you need complex string parsing)
- Vectorized string methods (`str.contains`, `str.extract`) — not needed until you're parsing free-text fields
- `pd.factorize` (overlap with `get_dummies` — not worth two passes)

### Exercises (~15–20 min total)

Do these in `week01-pandas-fundamentals.ipynb` alongside the Ch 5 exercises.

**1. Missing vitals**
```
Build a DataFrame with columns ['hr', 'spo2', 'temp'] and 20 rows.
Randomly set ~15% of values to NaN using rng.choice.
Count missing values per column. Then fill hr/spo2 with forward-fill and temp with the column mean.
Verify no NaNs remain.
```

**2. Outlier clip**
```
Given spo2 = pd.Series([98, 102, 75, 99, -3, 95, 101, 88]),
identify values outside physiologic range (70–100) using a boolean mask.
Replace them with NaN, then forward-fill.
```

**3. Binning HR into clinical labels**
```
Given hr = pd.Series([55, 78, 105, 160, 190, 95, 130]),
use pd.cut with bins [0, 60, 100, 160, 300] and labels ['brady','low-normal','normal','tachy'].
```

**4. Dummy variables**
```
Given a DataFrame with a 'sex' column containing 'M'/'F' and a 'ga_group' column with 'preterm'/'term',
use pd.get_dummies with drop_first=True and confirm the output shape.
```

## Chapter 8 — Data Wrangling: Join, Combine, Reshape

*(populate when you start Ch 8)*

## Chapter 10 — Data Aggregation and Group Operations

*(Deferred to Phase 5. Not blocking for Phases 2–4. Come back when you need per-cohort aggregations for the fairness analysis.)*

## Chapter 11 — Time Series

*(populate when you start Ch 11 — most important chapter for downstream work)*

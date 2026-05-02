# McKinney Reading Guide — Tiered Priorities & Exercises

Reference while reading *Python for Data Analysis* (3rd ed.). Reading order: **Ch 4 → 7 → 8 → 10 → 11.**

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

## Chapter 7 — Data Cleaning and Preparation

*(populate when you start Ch 7)*

## Chapter 8 — Data Wrangling: Join, Combine, Reshape

*(populate when you start Ch 8)*

## Chapter 10 — Data Aggregation and Group Operations

*(populate when you start Ch 10)*

## Chapter 11 — Time Series

*(populate when you start Ch 11 — most important chapter for downstream work)*

# Requirements for Memorization Metrics

## Python Packages Needed

Most dependencies should already be available in your environment. If not, install:

```bash
# If using conda (recommended for your setup)
conda install scikit-learn

# Or using pip
pip install scikit-learn
```

## Existing Dependencies (Should Already Have)

- `torch` ✓
- `transformers` ✓  
- `pandas` ✓
- `numpy` ✓
- `matplotlib` ✓
- `seaborn` ✓
- `typer` ✓ (used in your other scripts)

## Verify Installation

```bash
python -c "from sklearn.neighbors import NearestNeighbors; print('✓ scikit-learn OK')"
```

If this fails, install scikit-learn as shown above.

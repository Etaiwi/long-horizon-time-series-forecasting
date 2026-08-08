# Data

The final project uses the preprocessed Weather benchmark distributed through
the official LTSF-Linear/Autoformer dataset collection.

Place the file at:

```text
data/raw/weather.csv
```

The CSV is not committed because it is a downloaded benchmark artifact. The
local file used for the reported experiments has:

- rows: 52,696;
- forecasting variables: 21;
- SHA-256: `34EE981D07313E51DA2A50BB600072C8AE4A69CB4B0651F4CB93A069D7A2BA63`.

The official DLinear repository links the benchmark collection in its dataset
instructions: <https://github.com/cure-lab/LTSF-Linear#datasets>.

ETTh1 remains available locally for the earlier reconstruction and exploratory
experiments, but Weather is the selected final-project dataset.

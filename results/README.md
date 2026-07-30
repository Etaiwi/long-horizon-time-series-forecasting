# Results

This directory will contain compact, reproducible outputs such as:

- metric summaries;
- training histories;
- final comparison tables;
- report-ready figures.

Large model checkpoints are excluded from Git and stored locally under
`checkpoints/`.

Each saved result will identify its dataset, model, forecast horizon,
configuration, and random seed. Repeated-seed summaries will report the mean and
standard deviation across runs.

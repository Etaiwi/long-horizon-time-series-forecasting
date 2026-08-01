# DLinear collaboration workflow

This directory is the shared handoff point for the two independent DLinear
improvement tracks.

## Ownership

- `etai_experiment_log.md` is updated only by Etai's branch.
- `partner_experiment_log.md` is updated only by the partner's branch.
- Do not edit the other person's log. Read it as a handoff document.

Recommended branch names:

- `dlinear-etai-improvements`
- `dlinear-partner-improvements`

Recommended experiment locations:

```text
notebooks/experiments/etai/
notebooks/experiments/partner/
results/dlinear/improvements/etai/
results/dlinear/improvements/partner/
```

Keeping different notebook and result paths prevents Git conflicts. Reusable
code should remain in `src/ts_project/`, but experimental variants should use
different filenames until the group selects the final method.

## How to share work

1. Start the personal branch from the latest `main`.
2. Commit and push after each meaningful experiment.
3. Record the hypothesis, configuration, validation result, and conclusion in
   the branch owner's experiment log.
4. Share the branch name and latest commit hash with the other person.
5. Inspect the other branch on GitHub, or fetch it locally without merging it.

For example, after fetching, Etai can read the partner's latest log without
switching branches:

```powershell
git show origin/dlinear-partner-improvements:collaboration/partner_experiment_log.md
```

The partner can read Etai's log in the same way:

```powershell
git show origin/dlinear-etai-improvements:collaboration/etai_experiment_log.md
```

## Experiment rule

Use training and validation results while developing an improvement. Do not
repeatedly choose ideas or hyperparameters using the test set. Evaluate the
locked final candidate on the test set using the same split and metrics as the
DLinear reconstruction.


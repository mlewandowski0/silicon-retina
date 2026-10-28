"""Put the rerun "DVS (Fixed 200 FPS)" runs into the Fig. 12 data and notebook.

Fetches train_loss_epoch / val_loss_epoch of the W&B runs DVS_fixed_200fps_run1..N (project
e2vid-saliency2) and writes them into paper-figures/analysis/comparison_{train,val}_loss.csv in the
W&B-export format used there ("Step", "<run> - <metric>", "__MIN", "__MAX"). The three original
fixed-rate runs, whose events were mistimed, are removed from the CSVs, and their names in
paper-figures/analysis_saliency.ipynb are replaced by the new ones (the git history keeps the old data).

Run from the repository root, with `wandb login` done (offline runs: `wandb sync` them first):

    python scripts/runpod/export_fig12_runs.py --entity <wandb entity> [--dry-run]
"""
import argparse
import csv
import json
import os

OLD_RUNS = ["FIREnetwork_20260212_025427", "FIREnetwork_20260211_200608", "FIREnetwork_20260211_131459"]
METRICS = {"train": "train_loss_epoch", "val": "val_loss_epoch"}


def fetch(entity, project, names):
    """{run name: {"train": {step: value}, "val": {step: value}}} from W&B."""
    import wandb
    api = wandb.Api()
    out = {}
    for name in names:
        runs = list(api.runs(f"{entity}/{project}", filters={"display_name": name}))
        if not runs:
            raise SystemExit(f"no W&B run named {name} in {entity}/{project}")
        # a restarted run leaves an unfinished duplicate: prefer finished, then the longest history
        runs.sort(key=lambda r: (r.state == "finished", r.lastHistoryStep or 0), reverse=True)
        run = runs[0]
        if len(runs) > 1:
            print(f"{name}: {len(runs)} runs with this name, using {run.id} ({run.state})")
        hist = {k: {} for k in METRICS}
        for row in run.scan_history(keys=["_step", *METRICS.values()]):
            for split, metric in METRICS.items():
                if row.get(metric) is not None:
                    hist[split][int(row["_step"])] = float(row[metric])
        print(f"{name}: {run.id} {run.state}, {len(hist['val'])} epochs, best val {min(hist['val'].values()):.4f}")
        out[name] = hist
    return out


def merge_csv(path, metric, runs_split):
    """Drop OLD_RUNS columns and add one column triple per new run, aligned on Step."""
    rows = list(csv.DictReader(open(path)))
    cols = [c for c in rows[0] if not any(c.startswith(f"{o} - ") for o in OLD_RUNS)]
    table = {int(r["Step"]): {c: r[c] for c in cols} for r in rows}
    for name, values in runs_split.items():
        base = f"{name} - {metric}"
        cols += [base, base + "__MIN", base + "__MAX"]
        for step, v in values.items():
            row = table.setdefault(step, {c: "" for c in cols})
            row[base] = row[base + "__MIN"] = row[base + "__MAX"] = repr(v)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(cols)
        for step in sorted(table):
            w.writerow([str(step) if c == "Step" else table[step].get(c, "") for c in cols])


def update_notebook(path, new_names):
    text = open(path).read()          # plain replacement keeps the file's formatting
    for old, new in zip(OLD_RUNS, new_names):
        text = text.replace(old, new)
    json.loads(text)                  # still valid JSON
    open(path, "w").write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", default="e2vid-saliency2")
    ap.add_argument("--runs", nargs="+", default=[f"DVS_fixed_200fps_run{i}" for i in (1, 2, 3)])
    ap.add_argument("--figures", default="paper-figures")
    ap.add_argument("--dry-run", action="store_true", help="fetch and summarise only")
    args = ap.parse_args()
    if len(args.runs) != len(OLD_RUNS):
        raise SystemExit(f"expected {len(OLD_RUNS)} runs to replace {OLD_RUNS}")

    runs = fetch(args.entity, args.project, args.runs)
    if args.dry_run:
        return
    for split, metric in METRICS.items():
        merge_csv(os.path.join(args.figures, "analysis", f"comparison_{split}_loss.csv"), metric,
                  {n: h[split] for n, h in runs.items()})
    update_notebook(os.path.join(args.figures, "analysis_saliency.ipynb"), args.runs)
    print("updated comparison_{train,val}_loss.csv and analysis_saliency.ipynb; "
          "now re-run the notebook from paper-figures/")


if __name__ == "__main__":
    main()

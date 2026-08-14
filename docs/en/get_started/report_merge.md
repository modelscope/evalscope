# Merge and Rename Reports

```{note}
Merge and Rename are extensions to the [Evaluation Report List](visualization.md#evaluation-report-list) page. This page assumes you're already familiar with the Web visualization tool - see [Visualization](visualization.md) for installation and the basics.
```

Each `evalscope eval` run creates one report per model, scoped to whatever datasets that run covered. If you evaluate the same model against different datasets across several separate runs (a common pattern - e.g. adding a new benchmark later, or splitting a long dataset list across nodes), you end up with several report rows for one model instead of one. **Merge** combines a hand-picked set of those reports into a single report; **Rename** relabels a report's model in place, most often used to fix a typo'd model name or to give a report a clearer name before merging it with others.

## Merge

### How to Merge

1. On the report list page, select 2 or more reports using the checkbox on each card.
2. Click **Merge** in the selection tray that appears at the bottom of the page (next to **Compare**).
3. A confirmation dialog lists the reports about to be merged. Confirm to proceed.
4. On success, the new merged report appears in the list and the original reports it was built from are deleted.

### Requirements

A merge is only valid when:

- **At least 2 reports** are selected.
- **All selected reports belong to the same model.** Reports for different models cannot be merged.
- **No dataset appears in more than one selected report.** Merge combines disjoint dataset coverage into one report - it does not average or pick a winner between two runs of the *same* dataset.

The **Merge** button is disabled (with an inline reason) whenever the current selection fails one of these checks - this is a client-side pre-check mirroring the backend's own validation, so you see why before submitting rather than after. The backend re-validates independently regardless of what the UI already checked, since the client-side check only has access to whatever summary data the report cards happen to show.

```{important}
Merge additionally rejects the request (409) if any selected report belongs to a task that is still running.
```

### What Happens

Merging creates a **new run directory** (`merged_<timestamp>_<random>`) and:

1. Copies each source report's dataset result files, predictions, and reviews into it.
2. Merges the source runs' task configs into one `task_config.yaml`, with `datasets`/`dataset_args` covering the full merged set.
3. Regenerates the combined `report.html`.
4. Deletes the original source reports.

If step 4 (cleanup) fails for one or more sources, the merge itself is **not** rolled back - it already succeeded - and a separate cleanup notice tells you which reports to remove by hand.

If anything fails *before* the merged report is fully written, the partially-created run directory is rolled back and nothing is left behind.

### Error Reference

| Cause | Status |
|---|---|
| Fewer than 2 reports selected | 400 |
| Selected reports belong to different models | 400 |
| Malformed report reference | 400 |
| A selected report's directory is missing, or has no dataset results | 404 |
| Two selected reports share a dataset | 409 |
| A selected report belongs to a task still executing | 409 |

## Rename

Rename relabels a report's `model_name` - the report itself and all of its underlying dataset/prediction/review files are moved and rewritten in place under the new name; nothing is copied or merged.

### How to Rename

1. Select **exactly one** report (Rename is only available for a single-report selection).
2. Click **Rename** in the selection tray.
3. A dialog pre-filled with the current model name opens - edit it and confirm.

### Requirements

- The new name must be non-empty, different from the current name, and a valid path segment (no `/` or other path separators - it becomes a directory name on disk).
- No other report for that new model name may already exist within the same run directory.
- The report must not belong to a task that is still executing.

### Error Reference

| Cause | Status |
|---|---|
| Empty `new_model_name` | 400 |
| `new_model_name` contains a path separator | 400 |
| `new_model_name` is unchanged from the current name | 400 |
| Malformed report reference | 400 |
| Report directory not found | 404 |
| The report belongs to a task still executing | 409 |
| A report for `new_model_name` already exists in this run | 409 |

## API Reference

Both actions are also available directly over HTTP, if you're scripting against the Web service rather than clicking through the UI (see [Visualization](visualization.md) for how to start `evalscope service`). A report reference is the flat `{run_id}/{model_id}` form used throughout the reports API.

**`POST /api/v1/reports/merge`**

```json
{
  "root_path": "/path/to/outputs",
  "refs": ["run1/my-model", "run2/my-model"]
}
```

`root_path` is optional and falls back to the service's configured output root. Returns `{"success": true, "run_id": "<merged-run-id>", "model_id": "my-model"}` on success.

**`POST /api/v1/reports/runs/{run_id}/models/{model_id}/rename`**

```json
{
  "root_path": "/path/to/outputs",
  "new_model_name": "my-model-v2"
}
```

Returns `{"success": true, "run_id": "run1", "model_id": "my-model-v2"}` on success.

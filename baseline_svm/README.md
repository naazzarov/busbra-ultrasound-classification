# baseline + SVM

The implementation lives in `src/busbra/` and uses the common nested runner.
Read [method definitions](../docs/METHODS.md) and [reference audit](../docs/REFERENCE_AUDIT.md).

```bash
uv sync --locked
uv run --locked busbra baseline --data /PRIVATE/PATH/BUSBRA_features_ready.csv --output private_outputs/baseline-run-001 --jobs 2
```

The output must be a new private directory. It includes the exact handoff CSV/JSON
schemas, complete inner-search audit, code and pinned package versions. The frozen
CSV must pass byte-hash validation. Never publish this output folder.

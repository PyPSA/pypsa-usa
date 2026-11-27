Closes # (if applicable).
<!--- PRs should generally have an issue; small fixes can go without. Use "Part of #" if it only resolves part of one. -->
<!--- Into `develop`, "Closes #" does not link or close the issue: link it in the Development sidebar, and close it by hand after the merge. -->

## Changes proposed in this Pull Request
<!--- Describe your changes in detail -->

## Testing
<!--- Which test tiers ran and their results: -->
<!--- Tier A `pytest -m fast` / Tier B `pytest -m integration` / equivalence harness -->

## Checklist

- [ ] The issue, if any, is linked, and a reviewer is requested (or asked for in a comment).
- [ ] I have read and understood every change, including any AI-generated parts ([AI Use](https://github.com/PyPSA/pypsa-usa/blob/develop/docs/source/ai-use.md)).
- [ ] I tested my contribution locally and it seems to work fine.
- [ ] `pre-commit` and `pytest -m fast` pass locally.
- [ ] Code and workflow changes are sufficiently documented.
- [ ] Changed dependencies are added to `workflow/envs/environment.yaml`.
- [ ] Changes in configuration options are added in `workflow/repo_data/config/config.default.yaml`.
- [ ] Changes in configuration options are also documented in `docs/source/configtables/*.csv`.

(ai-use)=
# AI Use

You are welcome to use AI tools (chat assistants, code completion, coding agents) when
contributing to PyPSA-USA. AI-assisted contributions follow the same
[contributing guide](contributing.md) and meet the same standards as any other. This page
adds the expectations that are specific to AI use. It is adapted from the
[PyPSA policy on AI-based contributions](https://docs.pypsa.org/stable/contributing/contributing/#ai-based-contributions).

## Expectations

1. **You are responsible for what you submit.** Read every changed line before requesting
   a review, and be able to explain it yourself. "The AI wrote it" does not answer a
   review comment.
2. **Write issues and pull requests for a human reader.** Descriptions, titles, and review
   replies should be short and read as if you wrote them.
3. **Keep the code simple.** AI tools tend to add unneeded abstractions, fallbacks, and
   comments. Remove them before review.
4. **Keep pull requests focused.** Solve one issue per pull request. Open new issues for
   other problems the tool finds, rather than fixing them along the way.
5. **Agree on large changes first.** Discuss the approach with a maintainer in the issue
   before generating a large change.
6. **Keep the documentation in step.** Update the affected pages in the same pull request,
   and search the docs for statements your change makes untrue.

Maintainers may close contributions that do not meet these expectations without a full
review.

## Verify the results

Plausible output is not evidence that a model change is correct. Before requesting a review:

- **Run it.** Report only tests and workflow runs that actually happened, including
  failures and anything you skipped.
- **Check numbers at the source.** Compare costs, efficiencies, units, and policy targets
  with the cited dataset or paper. Open every citation and link; AI tools invent them.
- **Show changes in results.** If your change alters values the pipeline produces, compare
  them before and after using real data, never synthetic data. See the pull request
  section of
  [`AGENTS.md`](https://github.com/PyPSA/pypsa-usa/blob/develop/AGENTS.md).

Do not give AI tools API keys, or data you do not have permission to share.

## Using a coding agent

The repository includes instructions that coding agents can read:

| File | Purpose |
|------|---------|
| [`AGENTS.md`](https://github.com/PyPSA/pypsa-usa/blob/develop/AGENTS.md) | Architecture, conventions, and pull request rules for all agents. `CLAUDE.md` imports it. |
| [`CONTEXT.md`](https://github.com/PyPSA/pypsa-usa/blob/develop/CONTEXT.md) | Definitions of domain terms. |
| [`.claude/skills/pr`](https://github.com/PyPSA/pypsa-usa/blob/develop/.claude/skills/pr/SKILL.md) | The `/pr` command for Claude Code, which checks a branch and opens a pull request. |

If your agent does not load `AGENTS.md` automatically, ask it to read the file first. When
an agent repeatedly gets something wrong, propose a fix to `AGENTS.md`.

Agents can draft issues, pull requests, and replies, but a person decides what is posted,
chooses the reviewer, and approves and merges the pull request. An AI review does not
replace the review by a maintainer.

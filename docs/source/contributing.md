# Contributing

Welcome to **PyPSA-USA**'s contributor's guide! The following information
will help make contributing easy for everyone involved.

This document focuses on getting any potential contributor familiarized with
the development processes, but
[other kinds of contributions](https://opensource.guide/how-to-contribute)
are also appreciated. If you are new to using [git](https://git-scm.com) or
have never collaborated in a project previously, please have a look at
[contribution-guide.org](https://www.contribution-guide.org/). Other resources
are also listed in the excellent
[guide created by FreeCodeCamp](https://github.com/FreeCodeCamp/how-to-contribute-to-open-source).
Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt,
[Python Software Foundation's Code of Conduct](https://www.python.org/psf/conduct/)
is a good reference in terms of behavior guidelines.

Thank you for contributing to PyPSA-USA!

## Issue Reports

If you experience bugs or general issues with PyPSA-USA, please have a
look on the [issue tracker](https://github.com/PyPSA/pypsa-usa/issues).
If you don't see anything useful there, please feel free to submit a new issue.

```{tip}
Don't forget to include the closed issues in your search.
Sometimes a solution was already reported, and the problem is considered
**solved**.
```

New issue reports should include information about your programming environment
(e.g., operating system, Python version) and steps to reproduce the problem.
Please try also to simplify the reproduction steps to a very minimal example
that still illustrates the problem you are facing. By removing other factors,
you help us to identify the root cause of the issue.


## Code Contributions

The following steps will walk through how to submit code changes.

```{seealso}
Before contributing, please see our
[installation instructions](about-install.md) and [usage guide](about-usage.md). If you
use AI tools, also read [AI Use](ai-use.md).
```

### 1. Submit an Issue

In general, a pull request should have an associated issue. Before you start, find or create one in the
[issue tracker](https://github.com/PyPSA/pypsa-usa/issues) and comment that you are
working on it. Discussing the approach there often raises additional considerations
and avoids unnecessary work. Small fixes can go without an issue.

### 2. Fork the repository

- Create an user account on [GitHub](https://github.com/) if you do not
already have one.

- Fork the project [repository](https://github.com/PyPSA/pypsa-usa)
by clicking on the **Fork** button near the top of the page. This creates a
copy of the code under your account on the repository service.

- Clone this copy to your local disk:

    ``` bash
    ~/repositories $ git clone https://github.com/<github_username>/pypsa-usa.git
    ~/repositories $ cd pypsa-usa
    ~/repositories/pypsa-usa $ git remote add upstream https://github.com/PyPSA/pypsa-usa.git
    ~/repositories/pypsa-usa $
    ```

### 3. Install developer dependencies

If you plan on contributing to the respository, please install developer dependencies.

#### 3a. Using UV

Run the following command from the activated `pypsa-usa` conda environment.

```console
uv pip install -r pyproject.toml --extra dev
```

#### 3b. Using conda/mamba

Run the following command from the activated `pypsa-usa` conda environment.

```console
conda env update --name pypsa-usa --file workflow/envs/dev.yaml --prune
```

### 4. Install pre-commit hooks:

Pre-commit hooks will format your code to match the styles used in PyPSA-USA.

```console
pre-commit install
```

### 5. Implement your changes

- Create a new branch from the latest `develop`, with a name in the form of `issue-###`
where `###` is the auto assigned issue number from GitHub.

    ```console
    ~/repositories/pypsa-usa $ git fetch upstream develop
    ~/repositories/pypsa-usa $ git checkout --no-track -b issue-### upstream/develop
    ```

   and start making changes. **Never work on `master` or `develop` directly!**

- Start your work on this branch. Don't forget to add
[docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
to new functions, modules and classes.

- When you’re done editing:

    ```console
    ~/repositories/pypsa-usa $ git add <MODIFIED FILES>
    ~/repositories/pypsa-usa $ git commit -m 'descripitve commit message'
    ~/repositories/pypsa-usa $ git push
    ```

   to record your changes in [git](https://git-scm.com).


### 5a. Run Tests

Before submitting your changes, make sure to run the test suite to ensure your modifications haven't introduced any bugs. To run all tests:

```bash
# Directly using pytest
pytest -v
```



### 6. Submit a Pull Request to the **`develop`** branch

All contributions go to `develop`. Only release pull requests, from `develop`, go to
`master`.

- If everything works fine, push your local branch to your fork with:

    ```console
    git push -u origin issue-###
    ```

- Go to the web page of your fork and click
[Create a Pull Request](https://github.com/PyPSA/pypsa-usa/pulls). Then make sure you are submitting the pull request from your_git/issue-xxx branch to the PyPSA/PyPSA-USA:develop branch.

- Complete the pull request template, link the issue (step 7), and request a reviewer
(step 8). Open the pull request as a draft if it is not ready for review.

- Communicate on the github Pull Request page to reconcile any changes to be made!

(linking-issues)=
### 7. Link the issue

If the pull request has an issue, put `Closes #123` at the top of the pull request description, with one keyword per issue
(`Closes #123, closes #124`). If the pull request resolves only part of an issue, write
`Part of #123` instead, and leave the issue open.

GitHub acts on closing keywords only when a pull request targets the default branch, which
for PyPSA-USA is `master`. What happens to the issue therefore depends on the base branch:

| Pull request into | Used for | Linked by `Closes #123`? | Issue closes |
|---|---|---|---|
| `develop` | All contributions | No. Also link it in the **Development** sidebar. | **By hand**, after the merge |
| `master` | Releases, by maintainers | Yes | Automatically, on merge |

For a pull request into `develop`:

- Link the issue under **Development** in the pull request sidebar. If you do not have
  permission, the reviewer does this.
- After merging, whoever merged the pull request closes each fully resolved issue by hand,
  with a comment naming the pull request. Releases do not follow a fixed schedule, so a
  closed issue means the change is on `develop`; it reaches `master` with the next release.
- If an issue should stay open until the release, say so in the issue. The maintainer
  preparing the release adds `Closes #123` to the release pull request, so the issue closes
  when it merges into `master`.

### 8. Request a reviewer

Every pull request needs a review from a maintainer other than its author. Request at
least one under **Reviewers** once the pull request is ready. If you do not have
permission, ask for a review in a comment, and a maintainer will assign someone.

## Updating Documentation

You can help improve PyPSA-USA docs by making them more readable and
coherent, or by adding missing information and correcting mistakes.

PyPSA-USA's documentation uses
[Sphinx](https://www.sphinx-doc.org/en/master/) as its main documentation
compiler. This means that the docs are kept in the same repository as the
project code, and that any documentation update is done in the same way was a
code contribution. We use Markdown language with
[MyST](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html)
extensions

When working on documentation changes, ensure you have completed the developer installation instructions, and install the documentation build requirements:

```console
uv pip install -r docs/requirements.txt
```

Then compile the documentation on your local machine:

```console
cd docs && make html && cd ..
```

And use Python's built-in web server for a preview in your web browser
(`http://localhost:8000`)

```console
python3 -m http.server --directory 'docs/build/html'
```

Two conventions to know when touching documentation:

- The configuration-reference pages pull YAML snippets straight from
  `workflow/repo_data/config/*.yaml` via `literalinclude` directives anchored on
  `# docs : <name>` comment markers. Each marker sits directly above the top-level
  key it names, and an include renders from its marker to the next one. If you add,
  move, or rename a config section, keep its marker with it — `pytest tests/docs`
  verifies that every include renders exactly the section it claims to.
- Workflow-generated documentation figures live in `docs/source/_static/generated/`
  and are re-rendered with `snakemake docs_figures` (needs built networks),
  `snakemake docs_simpl_resolutions` (needs the `""` and `<N>a` variants, plus an optional `<run>County` sibling run),
  `snakemake docs_rec_trading_zones` and `snakemake docs_tct_targets` (repo data only);
  the DAG images are re-rendered
  with `snakemake dag`. Don't hand-edit these.

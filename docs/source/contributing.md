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
you help us to identify the root cause of the issue. There are four templates
set up to help with this process:

## Documentation Improvements

You can help improve PyPSA-USA docs by making them more readable and
coherent, or by adding missing information and correcting mistakes.

PyPSA-USA's documentation uses
[Sphinx](https://www.sphinx-doc.org/en/master/) as its main documentation
compiler. This means that the docs are kept in the same repository as the
project code, and that any documentation update is done in the same way was a
code contribution. We use Markdown language with
[MyST](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html)
extensions

When working on documentation changes, first install the required dependencies
in your environment

```bash
~/pypsa-usa $ pip install sphinx sphinx-book-theme myst-parser
```

Then compile the documentation on your local machine

```bash
~/pypsa-usa $ cd docs && make html && cd ..
```

And use Python's built-in web server for a preview in your web browser
(`http://localhost:8000`)

```bash
~/pypsa-usa $ python3 -m http.server --directory 'docs/build/html'
```

## Code Contributions

The following steps will walk through how to submit code changes.

```{seealso}
Before contributing, please see our
[installation instructions](about-install.md) and working [tutorial](about-tutorial.md)
```

### 1. Submit an Issue

Before you work on any non-trivial code contribution it's best to first create
a report in the [issue tracker](https://github.com/PyPSA/pypsa-usa/issues)
to start a discussion on the subject. This often provides additional considerations
and avoids unnecessary work.

### 2. Fork the repository

- Create an user account on [GitHub](https://github.com/) if you do not
already have one.

- Fork the project [repository](https://github.com/PyPSA/pypsa-usa)
by clicking on the **Fork** button near the top of the page. This creates a
copy of the code under your account on the repository service.

- Clone this copy to your local disk:

    ``` bash
    ~/repositories $ git clone https://github.com/<github_username>/PyPSA/pypsa-usa.git
    ~/repositories $ cd pypsa-usa
    ~/repositories/pypsa-usa $
    ```

### 4. Implement your changes

- Create a new branch with a name in the form of `issue-###` where `###` is
the auto assigned issue number from GitHub.

    ```bash
    ~/repositories/pypsa-usa $ git checkout -b issue-###
    ```

   and start making changes. **Never work on the main branch!**

- Start your work on this branch. Don't forget to add
[docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
to new functions, modules and classes.

- When you’re done editing:

    ```bash
    ~/repositories/pypsa-usa $ git add <MODIFIED FILES>
    ~/repositories/pypsa-usa $ git commit -m 'descripitve commit message'
    ~/repositories/pypsa-usa $ git push
    ```

   to record your changes in [git](https://git-scm.com).

### 5. Submit your contribution

- If everything works fine, push your local branch to
[GitHub](https://github.com/PyPSA/pypsa-usa) with:

    ```bash
    git push -u origin my-feature
    ```

- Go to the web page of your fork and click
[Create a Pull Request](https://github.com/PyPSA/pypsa-usa/pulls) to
send your changes for review.

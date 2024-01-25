## py4DSTEM Contributor Guidelines

**Welcome to the py4DSTEM project!**

We are grateful for your interest in contributing to py4DSTEM, an open-source Python library for 4D-STEM data analysis. Your contributions will help us make py4DSTEM a more powerful and versatile tool for the scientific community.

This document outlines the guidelines and expectations for contributors to the py4DSTEM project. Please read it carefully before making any contributions.

### Contribution Types

There are many ways to contribute to py4DSTEM, including:

* **Reporting bugs:** If you encounter a bug in py4DSTEM, please file a bug report on GitHub. Be sure to provide as much detail as possible about the bug, including steps to reproduce it.

* **Submitting feature requests:** If you have a suggestion for a new feature for py4DSTEM, please submit a feature request on GitHub. Describe the feature in detail and explain how it would benefit users.

* **Improving documentation:** py4DSTEM's documentation is always in need of improvement. If you have suggestions for improving the documentation, please submit a pull request or open an issue on GitHub.

* **Developing new code:** If you are a developer, you can contribute to py4DSTEM by writing new code. Please follow the coding guidelines below.

### Coding Guidelines

* **Code style:** py4DSTEM uses the black code formatter and flake8 linter. All code must pass these checks without error before it can be merged. We suggest using `pre-commit` to help ensure any code commited follows these practices, checkout the [setting up developer environment section below](#install). We also try to abide by PEP8 coding style guide where possible.

* **Documentation:** All code should be well-documented, and use Numpy style docstrings. Use docstrings to document functions and classes, add comments to explain complex code both blocks and individual lines, and use informative variable names.

* **Testing:** Ideally all new code should be accompanied by tests using pyTest framework; at the least we require examples of old and new behaviour caused by the PR. For bug fixes this can be a block of code which currently fails and works with the proposed changes. For new workflows or extensive feature additions, please also include a Jupyter notebook demonstrating the changes for an entire workflow i.e. from loading the input data to visualizing and saving any processed results. 

* **Dependencies:** New dependencies represent a significant change to the package, and any PRs which add new dependencies will require discussion and agreement from the development team. If a new dependency is required, please prioritize adding dependencies that are actively maintained, have permissive installation requirements, and are accessible through both pip and conda.

### Pull Requests

When submitting a pull request, please:

* **Open a corresponding issue:** Before submitting a pull request, open an issue to discuss your changes and get feedback.

* **Write a clear and concise pull request description:** The pull request description should clearly explain the changes you made and why they are necessary.

* **Follow the coding guidelines:** Make sure your code follows the coding guidelines outlined above.

* **Add tests:** If your pull request includes new code, please add unit tests.

* **Address reviewer feedback:** Respond promptly to reviewer feedback and make changes as needed.

### Code Review

All pull requests will be reviewed by project maintainers before they are merged. The maintainers will provide feedback on the code and may ask for changes. Please be respectful of the maintainers' feedback and make changes as needed.

### Code of Conduct

py4DSTEM is committed to providing a welcoming and inclusive environment for all contributors. Please read and follow the py4DSTEM Code of Conduct.

### Acknowledgments

We are grateful to all of the contributors who have made py4DSTEM possible. Your contributions are invaluable to the scientific community.

Thank you for your interest in contributing to py4DSTEM! We look forward to seeing your contributions.


### Setting up the Developer Environment
<a id='install'></a>

1. **Fork the Git repository:** Go to the py4DSTEM GitHub repository and click the "Fork" button. This will create your own copy of the repository on your GitHub account.

2. **Clone the forked repository:** Open your terminal and clone your forked repository to your local machine. This will create a local copy of your fork of the repository on your computer's filesystem. Use the following command, replacing `<your-github-username>` with your GitHub username and <path> with the the directory where you'd like to copy the py4DSTEM repository onto your filesystem:

    ```bash
    cd <path>
    git clone https://github.com/<your-github-username>/py4DSTEM.git
    ```
**_extra tips_:** Github has an excellent [tutorial](https://docs.github.com/en/get-started/quickstart/fork-a-repo) you can follow to learn more about for steps one and two.

3. **Create a new branch:** Create a new branch for your development work. This will silo any edits you make to the new branch, allowing you to work and make edits to your working branch without affecting the main branch of the repository. Use the following command, replacing `<branch-name>` with a name for your branch.  Please use an informative name describing the purpose of the branch, e.g. `<fixing-virtual-image-bug>`:

   ```bash
   git checkout -b <branch-name>
   ```

4. **Create an anaconda environment:** Environments are useful for a number of reasons. They create an isolated computing environment for each project or task, making it easy to reproduce results, manage dependencies, and collaborate with others. There are a number of different ways to use environments, including anaconda, pipenv, virtualenv - we recommend using [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) for environment management, where you can create a new environment with:

   ```bash
   conda create -n py4dstem-dev python=3.10
   ```

You can then enter that environment using:

   ```bash
   conda activate py4dstem-dev
   ```


5. **Install py4DSTEM in editable mode:** Navigate to the cloned repository directory and install py4DSTEM in editable mode. This will allow you to make changes to the code and test them without having to reinstall the library. Use the following command:

   ```bash
   conda activate py4dstem-dev
   pip install -e .
   pip install flake8 black   # install linter and autoformatter
   ```

You can now make changes to the code and test them using your favorite Python IDE or editor.

6.  **_(Optional)_ - Using `pre-commit`**: `pre-commit` streamlines code formatting and linting. Essentially it runs black (an autoformatter) and flake8 (a linter) whenever a new commit is attempted on all staged files, and only allows the commit to proceed if they both pass. To use pre-commit, run:

    ```bash
    conda activate py4dstem-dev
    pip install pre-commit
    cd <path-to-your-fork-of-the-py4DSTEM-git-repo>   # go to your py4DSTEM repo
    pre-commit install
    ```
    
This will setup pre-commit to work on this repo by creating/changing a file in .git/hooks/pre-commit, which tells `pre-commit` to automatically run flake8 and black when you try to commit code.  It won't affect any other repos.

**_extra tips_:** 

```bash
# You can call pre commit manually at any time without committing
pre-commit run # will check any staged files 
pre-commit run -a # will run on all files in repo

# you can bypass the hook and commit files without the checks 
# (this isn't best practice and should be avoided, but there are times it can be useful)

git add file # stage file as usual 
git commit -m "you commit message" --no-verify # commit without running checks
git push # push to repo. 
```

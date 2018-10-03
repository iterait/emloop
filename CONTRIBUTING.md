# How to Contribute

There are two ways of contributing to cxflow.
One can either report a bug (or feature request) to the official [issue tracker](https://github.com/iterait/cxflow/issues) or implement a tested piece of code.
Do not hesitate to get in touch with us at cxflow@iterait.com.

## Creating an Issue
When you find a bug or feel like there is a feature you miss, please do not hesitate to report it to the [issue tracker](https://github.com/iterait/cxflow/issues).

A great issue should contain the following:

- English language
- Short and self-explaining title
- Exhaustive description of the bug or feature request
- (If applicable) Minimal (not) working example
- Summary
- Proper labels (e.g. *bug*, *test*, etc.)
- (Optionally) Assignees

## Developing

When you find an issue you would like to solve, please do not hesitate to do so.
Nevertheless, make sure there is no other developer assigned to this particular issue.

The first thing you need to do is to **[fork](https://guides.github.com/activities/forking/)** this repository to your account.
Then, create a **new branch** to which the changes will be committed.
Please make sure the branch is reasonably named.

Make sure you commit only small pieces of code and their commit messages are simple and explaining.
When implementing the desired functionality, please *do write* a **documentation strings** to all modules, classes, methods and functions.
Please use Python 3.5 type hints as much as possible in order to keep the code easily understandable.
Do not forget to stick to the Python **coding style** ([PEP 8](https://www.python.org/dev/peps/pep-0008/)).

Once the bug is fixed or the feature is implemented, please do not forget to run the **unit tests**.

```bash
python setup.py test
```

In the case of fixing a bug, please add the **regression tests** which address this particular bug.
Note that your code cannot be accepted without proper testing.

Once the code is finished and tests have successfully passed, it's time to create a **pull request**.
Be sure to create the pull request from the correct source branch into a new branch in the official repository.

Please name the pull request briefly and self-explainingly.
The pull request should contain a description of the changes, possibly with examples.
In case of solving some issues, the description of the pull request must contain a list of issues to be closed, e.g. `Closes #12`. 

When the pull request is submitted, the CI is run.
After all tests have successfully passed in all supported environments, the PR assignee or the master developer will code review the changes.
Then, the pull request might be merged.

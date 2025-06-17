[![PyPI version fury.io](https://badge.fury.io/py/deepforest-modal-app.svg)](https://pypi.python.org/pypi/deepforest-modal-app/)
[![Documentation Status](https://readthedocs.org/projects/deepforest-modal-app/badge/?version=latest)](https://deepforest-modal-app.readthedocs.io/en/latest/?badge=latest)
[![CI/CD](https://github.com/martibosch/deepforest-modal-app/actions/workflows/tests.yml/badge.svg)](https://github.com/martibosch/deepforest-modal-app/blob/main/.github/workflows/tests.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/deepforest-modal-app/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/deepforest-modal-app/main)
[![GitHub license](https://img.shields.io/github/license/martibosch/deepforest-modal-app.svg)](https://github.com/martibosch/deepforest-modal-app/blob/main/LICENSE)

# DeepForest modal app

Modal app for *serverless* [DeepForest](https://github.com/weecology/DeepForest) model inference, training/fine tuning of tree crown detection and species classification models.

See [an example notebook showcasing the features using the TreeAI Database](https://deepforest-modal-app.readthedocs.io/en/latest/treeai-example.html)

![comparison](https://github.com/martibosch/deepforest-modal-app/raw/main/docs/figures/comparison.png)
*Example annotations from the TreeAI Database (left), predictions with the DeepForest pre-trained tree crown model (center) and with the fine-tuned model (right).*

## Installation

```bash
pip install deepforest-modal-app
```

## Acknowledgements

- A big thank you to [Charles Frye](https://github.com/charlesfrye) and [Thomas Capelle](https://github.com/tcapelle) for helping me to get started with [Modal](https://modal.com).
- This package was created with the [martibosch/cookiecutter-geopy-package](https://github.com/martibosch/cookiecutter-geopy-package) project template.

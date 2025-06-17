[![PyPI version fury.io](https://badge.fury.io/py/deepforest-modal-app.svg)](https://pypi.python.org/pypi/deepforest-modal-app/)
[![Documentation Status](https://readthedocs.org/projects/deepforest-modal-app/badge/?version=latest)](https://deepforest-modal-app.readthedocs.io/en/latest/?badge=latest)
[![CI/CD](https://github.com/martibosch/deepforest-modal-app/actions/workflows/tests.yml/badge.svg)](https://github.com/martibosch/deepforest-modal-app/blob/main/.github/workflows/tests.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/deepforest-modal-app/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/deepforest-modal-app/main)
[![GitHub license](https://img.shields.io/github/license/martibosch/deepforest-modal-app.svg)](https://github.com/martibosch/deepforest-modal-app/blob/main/LICENSE)

# DeepForest modal app

Modal app for *serverless* [DeepForest](https://github.com/weecology/DeepForest) inference, training/fine tuning of tree crown detection and species classification models.

## Features

Execute all your pipeline (preprocessing, training/fine tuning, inference, postprocessing...) within the same *local* script/notebook:

- When running DeepForest inference and training/fine tuning of tree detection models, this library will handle setting up a [Modal *ephemeral* apps](https://modal.com/docs/guide/apps) in a GPU-enabled environment, execute the deep learning parts there and you will then retrieve the results (e.g., a geopandas data frame) as a *local* variable within your notebook
- The required data (e.g., aerial imagery) and model checkpoints are uploaded to persistent [Modal storage volumes](https://modal.com/docs/guide/volumes)
- Model checkpoints from HuggingFace Hub and PyTorch Hub are cached locally in a storage volume so uptime for ephemeral apps is minimal

See [an example notebook showcasing the features using the TreeAI Database](https://deepforest-modal-app.readthedocs.io/en/latest/treeai-example.html)

![comparison](https://github.com/martibosch/deepforest-modal-app/raw/main/docs/figures/comparison.png)
*Example annotations from the TreeAI Database (left), predictions with the DeepForest pre-trained tree crown model (center) and with the fine-tuned model (right).*

## Installation

This app requires [geopandas](https://github.com/geopandas/geopandas) in the local environment, which cannot be installed with pip. Until we have a working conda-forge recipe, the easiest solution is to *first* install geopandas using conda/mamba, e.g.:

```bash
conda install geopandas
```

and then install "deepforest-modal-app" using pip:

```bash
pip install deepforest-modal-app
```

## Acknowledgements

- A big thank you to [Charles Frye](https://github.com/charlesfrye) and [Thomas Capelle](https://github.com/tcapelle) for helping me to get started with [Modal](https://modal.com).
- This package was created with the [martibosch/cookiecutter-geopy-package](https://github.com/martibosch/cookiecutter-geopy-package) project template.

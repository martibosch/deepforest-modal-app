[![PyPI version fury.io](https://badge.fury.io/py/deepforest-modal-app.svg)](https://pypi.python.org/pypi/deepforest-modal-app/)
[![Documentation Status](https://readthedocs.org/projects/deepforest-modal-app/badge/?version=latest)](https://deepforest-modal-app.readthedocs.io/en/latest/?badge=latest)
[![CI/CD](https://github.com/martibosch/deepforest-modal-app/actions/workflows/tests.yml/badge.svg)](https://github.com/martibosch/deepforest-modal-app/blob/main/.github/workflows/tests.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/deepforest-modal-app/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/deepforest-modal-app/main)
[![GitHub license](https://img.shields.io/github/license/martibosch/deepforest-modal-app.svg)](https://github.com/martibosch/deepforest-modal-app/blob/main/LICENSE)

# DeepForest modal app

Modal app for *serverless* [DeepForest](https://github.com/weecology/DeepForest) [1] inference, training/fine tuning of tree crown detection and species classification models.

## Features

Execute all your pipeline (preprocessing, training/fine tuning, inference, postprocessing...) within the same *local* script/notebook:

- When running DeepForest inference and training/fine tuning of tree detection models, this library will handle setting up a [Modal *ephemeral* apps](https://modal.com/docs/guide/apps) in a GPU-enabled environment, execute the deep learning parts there and you will then retrieve the results (e.g., a geopandas data frame) as a *local* variable within your notebook
- Optimized defaults for the serverless infrastructure (i.e., different training and inference GPUs) and matching settings (batch sizes, number of workers, image pre-loading...) to improve performance. **TODO**: support for multi-GPU training coming shortly.
- The required data (e.g., aerial imagery) and model checkpoints are uploaded to persistent [Modal storage volumes](https://modal.com/docs/guide/volumes)
- Model checkpoints from HuggingFace Hub and PyTorch Hub are cached locally in a storage volume so uptime for ephemeral apps is minimal

![comparison](https://github.com/martibosch/deepforest-modal-app/raw/main/docs/figures/comparison.png)
*Example annotations from the TreeAI Database (left), predictions with the DeepForest pre-trained tree crown model (center) and with the fine-tuned model (right).*

## Examples

The following example notebooks use the TreeAI Database [2] to illustrate the features of this setup:

- [`getting-started.ipynb`](https://deepforest-modal-app.readthedocs.io/en/latest/getting-started.html): example notebook showcasing inference and training/fine-tuning (with the default settings).
- [`advanced-customizations.ipynb`](https://deepforest-modal-app.readthedocs.io/en/latest/advanced-customizations.html): shows how to use data augmentations, logging, callbacks and sharing checkpoints in HuggingFace Hub.
- [`crop-model.ipynb`](https://deepforest-modal-app.readthedocs.io/en/latest/crop-model.html): draft on multi-species classification using the DeepForest crop model.

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

## References

1. Weinstein, B. G., Marconi, S., Aubry‐Kientz, M., Vincent, G., Senyondo, H., & White, E. P. (2020). DeepForest: A Python package for RGB deep learning tree crown delineation. Methods in Ecology and Evolution, 11(12), 1743-1751.
1. Beloiu Schwenke, M., Xia, Z., Novoselova, I., Gessler, A., Kattenborn, T., Mosig, C., Puliti, S., Waser, L., Rehush, N., Cheng, Y., Xinliang, L., Griess, V. C., & Mokroš, M. (2025). TreeAI Global Initiative - Advancing tree species identification from aerial images with deep learning (TreeAI.V1.2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15351054

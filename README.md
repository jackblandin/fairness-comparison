This repository is a fork of https://github.com/algofairness/fairness-comparison, which includes development setup info and the corresponding paper.
The remainder of this README references the contributions made by our [Fairness Through Counterfactual Utilities](https://arxiv.org/abs/2108.05315) paper.

# Development Setup

```
# Create the conda environment from the config file
conda env create -f=conda.yaml

# Activate the conda environment
conda activate fairness-compairson

# Create an IPython kernel which will allow you to run the Jupyter Notebook in the conda environment
python3.6 -m ipykernel install --user --name fairness-compairson
```

# Reproducing Experiment Results

Start the jupyter notebook server
```
jupyter notebook

```
Then open localhost:8888 in web browser and start the [cfutil-experiments.ipynb](./notebooks/cfutil-experiments.ipynb) notebook with the `fairness-comparison` kernel.
Execute each of the cells in the notebook to rerun our experiments.
You can also modify the parameters within the notebook in order to conduct your own analysis.

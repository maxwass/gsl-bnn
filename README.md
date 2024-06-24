# Graph Structure Learning with Interpretable Bayesian Neural Networks

Repository containing code for the paper ["Graph Structure Learning with Interpretable Bayesian Neural Networks"](https://arxiv.org/abs/2406.14786)


<br>

<div align="center">
  <img src="figures/layer_and_unrolling.png" alt="The GSL neural network diagram", width="400">
  <br>
  <em>DPG: The Graph Structure Learning neural network with independently interpretable parameters.</em>
</div>

<br><br>

<div align="center">
  <img src="figures/bayes_dpg_workflow.png" alt="Bayesian workflow diagram", width="700">
  <br>
  <em>Bayesian Modeling Workflow.</em>
</div>

<br><br>

<div align="center">
  <img src="figures/predictive_check_histogram.png" alt="Predictive checking diagram", width="300">
  <br>
  <em>Prior and Posterior Predictive Checking.</em>
</div>

<br>

<div align="center">
  <img src="figures/synthetic_label_predMean_predStdv.png" alt="Predictive checking diagram">
  <br>
  <em>Subjective Evaluation on Synthetic Data.</em>
</div>

<br>


For a gentle introduction to performing Bayesian inference on synthetics with DPG, see the [notebook](notebooks/simple_dpg_example.ipynb). You may first need to create the synthetic data; refer to the synthetics [README](data/synthetic/README.md) and the data [generating file](src/synthetic_data_generator.py).

## Authors

- [Max Wasserman](mailto:maxw14k@gmail.com)

## Setup

Refer to [Setting Up the Project Environment](docs/setup.md) for instructions on how to configure your local environment to run

# Relational Deep Learning for Formula 1 (F1) Driver Position Prediction

This project explores relational deep learning on the RelBench `rel-f1` dataset to predict Formula 1 driver finishing position. The goal is to compare traditional flat-feature machine learning baselines with a graph-based relational deep learning model that preserves the original multi-table database structure.

## Project Overview

Real-world data is often stored across multiple connected tables rather than in a single flat dataset. In this project, I used the RelBench `rel-f1` benchmark, which contains Formula 1 data organized across relational tables such as drivers, races, results, constructors, standings, circuits, and qualifying records.

The prediction task is `driver-position`, a regression task where the objective is to predict a driver's average finishing position over a future time window.

## Main Result

The graph neural network achieved the best validation performance among all tested models.

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Mean Baseline | 4.3344 | 5.4255 | -0.3695 |
| Random Forest | 3.7063 | 4.5931 | 0.0185 |
| MLP | 3.7553 | 4.6405 | -0.0019 |
| GNN | 3.2502 | 4.0537 | 0.2355 |

The results suggest that preserving the relational structure of the database helped the graph neural network capture useful predictive information beyond what was available from manually engineered flat features.

## Methods

The project compares four approaches:

1. **Mean Baseline**  
   A simple baseline that predicts the average target value from the training set.

2. **Random Forest Regressor**  
   A classical machine learning model trained on manually engineered driver-level historical features.

3. **MLP Baseline**  
   A standard deep learning model trained on the same flat engineered features as the Random Forest model.

4. **Graph Neural Network**  
   A heterogeneous graph neural network based on the RelBench tutorial design. The relational database was converted into a graph where tables became node types and primary key / foreign key relationships became edge types.

## Dataset

- Dataset: RelBench `rel-f1`
- Domain: Formula 1 racing data
- Task: `driver-position`
- Problem type: Regression
- Main evaluation metric: Mean Absolute Error (MAE)

## Tools and Libraries

- Python
- PyTorch
- PyTorch Geometric
- RelBench
- TorchFrame
- scikit-learn
- pandas
- NumPy
- matplotlib
- Google Colab

## Repository Files

- `relbench_f1_driver_position.ipynb`  
  Full notebook containing data loading, graph construction, baseline models, GNN training, evaluation, and results.

- `relbench_f1_project_report.pdf`  
  Project report summarizing the motivation, methodology, results, limitations, and future work.

## Limitations

This project was designed to be reproducible in a standard Google Colab environment, so I used the smaller RelBench `rel-f1` dataset instead of a larger benchmark such as `rel-amazon`.

Text-based columns were simplified during graph construction to avoid compatibility issues with external text embedding components. Hyperparameter tuning was also limited to keep the project computationally efficient.

## Future Work

Possible extensions include:

- applying the pipeline to a larger RelBench dataset,
- improving text-column handling with richer embedding methods,
- performing more extensive hyperparameter tuning,
- testing alternative relational deep learning architectures.

## References

- RelBench documentation: https://relbench.stanford.edu/
- RelBench rel-f1 dataset: https://relbench.stanford.edu/datasets/rel-f1/
- Robinson, J., Ranjan, R., Hu, W., Huang, K., Han, J., Dobles, A., Fey, M., Lenssen, J. E., Yuan, Y., Zhang, Z., He, X., & Leskovec, J. (2024). RelBench: A benchmark for deep learning on relational databases. arXiv: https://arxiv.org/abs/2407.20060
- PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io/

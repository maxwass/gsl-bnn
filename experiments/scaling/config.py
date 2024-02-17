# config.py

from src import DATA_PATH, RESULTS_PATH



# General experiment settings
experiment_settings  = {
    ## data generation
    # parameters for the data generation
    'p': .25, # defining ER parameter
    'num_graphs': 100,
    'graph_sizes': [10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 1000],

    # where should the data be saved?
    'data_path': DATA_PATH + 'scaling/datasets.pkl',
    'results_path': RESULTS_PATH + 'scaling/',

    ## Modeling
    'map_num_train_samples': 10,
    'map_num_test_samples': 20,
    'map_dpg_depth': 200,

    # Optimization
    'maxiter': 5000, # change to larger, still not convergent here
    'learning_rate': .01,

    # MAP models to train
    'map_graph_sizes_to_train': [20, 50, 80, 100, 150, 200, 300],

    ## Evaluation/Plotting
    'size_generalization_graph_sizes': [250, 300, 400, 500, 1000],

    # misc
    'random_seed': 50,
    
    # ... other settings
}

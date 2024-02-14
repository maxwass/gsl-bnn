from src import SYNTHETIC_DATA_PATH, RESULTS_PATH


# General experiment settings
experiment_settings  = {
    # data
    'data_paths': 
        {'RG_.32': SYNTHETIC_DATA_PATH + 'RG_N=20_r=0.32_dim=2.pt', 
         'RG_.5': SYNTHETIC_DATA_PATH + 'RG_N=20_r=0.5_dim=2.pt',
         'ER_.5': SYNTHETIC_DATA_PATH + 'ER_N=20_p=0.5.pt',
         'BA_1': SYNTHETIC_DATA_PATH + 'BA_N=20_m=1.pt'}
         ,
    'num_signals': 'expected', # == float('inf'),
    'num_test_samples': 100,
    # posterior samples / point estimates
    'samples_path': RESULTS_PATH + 'iid_generalization/',
    'results_path': RESULTS_PATH + 'label_mismatch/',
    # misc
    'random_seed': 0,
    # ... other settings
}

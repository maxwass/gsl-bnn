These data sets are created in src/synthetic_data_generator.py. 

They are a dict with the raw binary adjacencies along  'analytic' euclidean distance matrix, as well as it's finite sample approximations.

Each file contains the following dictionary
 {'adjacencies': adjacencies, # raw binary graphs sampled using the graph_params
 'graph_params': graph_params
 **euclidean_distance_dict} # 'analytic' euclidean distance matrix (= infinite signals), as well as it's finite sample approximations

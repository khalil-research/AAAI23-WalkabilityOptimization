# Data-Driven Walkability Optimization

* prepare_data.py - Data Preprocessing. Obtain the shortest path pairs, existing amenity instances, residential locations, and candidate allocation location for each NIA.

* optimize.py - Run the optmization models.
>  python optimize.py MODEL_NAME NIA_ID --k_array k_{grocery}, k_{restaurant}, k_{school}
>  
  * Options for MODEL_NAME are:
    * OptMultipleDepth: MILP, MultiChoice case
    * OptMultiple: MILP, SingleChoice case
    * OptMultipleDepthCP: CP, MultiChoice case
    * OptMultipleCP: CP, SingleChoice case
    * GreedyMultipleDepth: Greedy, MultiChoice case
    * GreedyMultiple: Greedy, SingleChoice case

* results_summary.py - Get model compairson and emprical evaluation statistics.


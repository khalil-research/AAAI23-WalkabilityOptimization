# Data-Driven Walkability Optimization

Instances for computational experiments in this projects are created from Neigbhourhood Improvement Areas (NIAs) at the City of Toronto. The Pedestrian Netowrk data (pednet.zip) can be obtained from the publicly available dataset created by the City of Toronto: https://github.com/gcc-dav-official-github/dav_cot_walkability/tree/master/data. Information on NIAs is publicly available from The City of Toronto’s Open Data Portal: https://www.toronto.ca/city-government/data-research-maps/neighbourhoods-communities/neighbourhood-profiles/nia-profiles/

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


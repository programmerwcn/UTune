## Key Features

### Operator-level CAM Predictor (node_estimation.py)

class OperatorModel

### End-to-end index tuning process (index_selection.py)

def index_selection_9 in index_selection.py: end-to-end index tuning process

* Index Benefit Estimation (def cost_estimation_4)
  Estimate index benefit with operator-level model and uncertainty-aware correction

  Operator Model in node_estimation.py
* Index Selection (def candidate_enumeration_3)
  Uncertainty-aware index selection
* PostgreSQL Integration

  Online integration with PostgreSQL to create indexes and execute queries, with 'postgres_dbms.py'
* Operator Model Update
  def detect_error_nodes_4: get error nodes from actual feedback

  def update_model_with_replaybuffer_3: model update

### Detailed Implementation of Query Plan Analysis (query_plan_node.py)

detailed implementation of functions in end-to-end tuning process

## Running

Run the index selection process using the following command:

```
python node_estimation/index_selection.py --workload_configs "[(0,20),(0,20),(0,20),(0,20),(0,20),(0,20),(0,20),(0,20),(0,20),(0,20),(0,20),(0,20),(0,20),(0,20),(0,20),(0,20)]" --exp_id "tpch_static" --selection_policy 'Boltzman' --rerun True --epsilon 0.5 --decay_rate 0.9 --db "tpch" --index_config 8 --lam0 0.5
```

Configure db info in the file db_config.json

All queries are stored in tpc_h_static_100_pg.json file.

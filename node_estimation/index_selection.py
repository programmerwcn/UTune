import itertools
import json
import sys
import os
import random
import logging
import torch
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from postgres_dbms import PostgresDatabaseConnector
import query_plan_node
import datetime
import argparse
import ast
import time
import numpy as np

MAX_INDEX_WIDTH = 2
MAX_INDEXES_PER_TABLE = 3
PARENT_PATH = "xx"

class Query():
    def __init__(self,id, query_string, predicates, payload):
        self.id = id
        self.query_string = query_string
        self.predicates = predicates
        self.payload = payload

class ActualQuery():
    """
    id: str
    query_id: int
    index_names: list of index_name
    plan: plan of the query
    time: actual time of the query
    cost: cost estimation of the query
    """
    def __init__(self, query_id, plan, time, indexes = None):
        self.id = str(query_id) + '_' 
        self.index_names = []
        if indexes:
            self.id += '_'.join(indexes)
            for index in indexes:
                self.index_names.append(index)
        self.query_id = query_id
        self.plan = plan
        self.time = time
        
    
    def get_effective_indexes(self):
        """
        Analyze query plan to get effective indexes
        """
        plan_tree = query_plan_node.build_query_plan_tree(self.plan)
        dfs_nodes = query_plan_node.dfs_traverse(plan_tree)
        for node in dfs_nodes:
            if node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan']:
                index_name = node.attributes['Index Name']
                self.index_names.append(index_name)
        self.id = str(self.query_id) + '_indexed' 

class WhatIfQuery():    
    """
    query_id: int
    index_ids: list of index
    plan: plan of the query
    cost: cost estimation of the query
    refined_cost: cost estimation after refinement
    """
    def __init__(self, query_id, plan, cost, refined_cost, indexes = None):
        self.query_id = query_id
        self.index_ids = []
        self.plan = plan
        self.cost = cost
        self.refined_cost = refined_cost
        self.id = str(query_id) + '_' 
        if indexes:
            self.id += '_'.join(indexes)
            for index in indexes:
                self.index_ids.append(index)

class ErrorNode():
    def __init__(self, index_id, query_id, cam):
        self.index_id = index_id
        self.query_id = query_id
        self.cam = cam
    
    def to_json(self):
        return {
            "index_id": self.index_id,
            "query_id": self.query_id,
            "cam": self.cam
        }
    
class CandidateIndex():
    """
    id: str
    index_name: str
    whatif_queries:[]
    actual_queries:[]
    estimated_reward: int
    actual_reward: int
    """
    def __init__(self, id, index_name, table, columns, is_include, includes, cluster):
        self.id = id
        self.index_name = index_name
        self.table = table
        self.columns = columns
        self.is_include = is_include
        self.includes = includes
        self.cluster = cluster
        self.actual_reward = 0
        self.estimated_reward = 0
        self.whatif_queries = []
        self.effective_queries = []
        self.uncertainty = 0
        self.potential = 0

    def add_query(self, query_id, what_if_plan):
        self.queries[query_id] = what_if_plan

    def remove_query(self, query_id):
        if query_id in self.queries:
            del self.queries[query_id]

    def get_reward(self):
        reward = 0
        for query_id in self.queries:
            reward += self.queries[query_id].get('origin_cost',0) - self.queries[query_id].get('refined_estimation', 0)
        return reward

    def get_actual_reward(self):
        reward = 0
        for query_id in self.queries:
            if 'index_time' in self.queries[query_id]:
                reward += self.queries[query_id].get('origin_time',0) - self.queries[query_id]['index_time']
        return reward

    # def get_estimated_reward(self):
    #     reward = 0
    #     for query_id in self.qu

class IndexConfig():
    def __init__(self, index_ids):
        self.index_ids = index_ids
        self.reward = 0
        self.uncertainty = 0
        self.total_value = 0
    
    def add_index(self, index):
        self.indexes.add(index)

    

def clear_indexes():
    with open('db_config.json', 'r') as f:
        db_conf = json.load(f)
    database_connector = PostgresDatabaseConnector(db_conf, autocommit=True)
    database_connector.drop_indexes()


def get_index_name(index_cols, table_name, include_cols=()):
        if include_cols:
            include_col_names = ','.join(tuple(map(lambda x: x[0:4], include_cols))).lower()
            index_name =  table_name.lower() + '#' + ','.join(index_cols).lower() + '#' + include_col_names
        else:
            index_name = table_name.lower() + '#' + ','.join(index_cols).lower()
        return index_name

def get_index_name_btree(index_cols, table_name, include_cols=()):
    if include_cols:
            include_col_names = ','.join(tuple(map(lambda x: x[0:4], include_cols))).lower()
            index_name =  table_name.lower() + '_' + '_'.join(index_cols).lower() + '_' + include_col_names
    else:
            index_name = table_name.lower() + '_' + '_'.join(index_cols).lower()
    return index_name

def candidate_2_btee(candidate_index):
    index_def = candidate_index.split('#')
    table_name = index_def[0]
    columns = index_def[1].split(',')
    index_name = candidate_index.replace('#', '_').replace(',', '_')
    return index_name


def gen_candidates_from_predicates(query_id, predicates, payloads, is_includes = False):
    """
    This method take predicates (a dictionary of lists) as input and
    creates the generate arms for all possible column combinations

    :param connection: SQL connection
    :param query_obj: Query object
    :return: list of bandit arms
    """
    candidate_indexes = {}

    # generate permutation for predicates
    for table_name, table_predicates in predicates.items():
        col_permutations = list()
        index_width = len(table_predicates)
        if index_width > MAX_INDEX_WIDTH:
            index_width = MAX_INDEX_WIDTH
        for j in range(1, (index_width + 1)):
            col_permutations = col_permutations + list(itertools.permutations(table_predicates, j))
        for col_permutation in col_permutations:
            arm_name = get_index_name(col_permutation, table_name)
            candidate_indexes[arm_name] = {'table': table_name.lower(), 'columns': [co.lower() for co in col_permutation]}
            candidate_indexes[arm_name]['is_include'] = 0
            if len(table_predicates) == len(col_permutation):
                candidate_indexes[arm_name]['cluster'] = table_name.lower() + '_' + str(query_id) + '_all'
            # candidate_indexes.append(arm_name)

    # generate permutation for payloads
    for table_name, table_payloads in payloads.items():
        if len(table_payloads) > MAX_INDEX_WIDTH:
                continue
        col_permutation = table_payloads
        arm_name = get_index_name(col_permutation, table_name.lower())
        candidate_indexes[arm_name] = {'table': table_name.lower(), 'columns': [co.lower() for co in col_permutation]}
        candidate_indexes[arm_name]['is_include'] = 0
        candidate_indexes[arm_name]['cluster'] = table_name.lower() + '_' + str(query_id) + '_all'
        # candidate_indexes.append(arm_name)
    # include
    if is_includes:
        for table_name, table_predicates in predicates.items():
            includes = list()
            if table_name in payloads:
                includes = sorted(list(set(payloads[table_name]) - set(table_predicates)))
            if includes:
                col_permutations = list(itertools.permutations(table_predicates, len(table_predicates)))
                for col_permutation in col_permutations:
                    arm_name = get_index_name(col_permutation, table_name.lower(), includes)
                    candidate_indexes[arm_name] = {'table': table_name.lower(), 'columns': [co.lower() for co in col_permutation], 'includes': includes}
                    candidate_indexes[arm_name]['cluster'] = table_name.lower() + '_' + str(query_id) + '_all'
                    candidate_indexes[arm_name]['is_include'] = 1
                    # candidate_indexes.append(arm_name)
    return candidate_indexes



def gen_candidates_3(workload, existing_candidates):
    """
    Generate candidates based on input workload
    workload: {query_id: Query object}
    return: {index_id: CandidateIndex object}
    Add what-if queries that is not related with candidate generation (effective queries)
    """
    # Get query list
    is_include = False
    all_candidates = {}
    for query_id, query in workload.items():
            candidate_indexes = gen_candidates_from_predicates(query_id, query.predicates, query.payload, is_include)
            # query['candidate_indexes'] = candidate_indexes
            for index, info in candidate_indexes.items():
                index_name = get_index_name_btree(info['columns'], info['table'], info.get('includes', []))
                if index in all_candidates:
                    all_candidates[index].whatif_queries.append(query_id)
                else:
                    candidate_index = CandidateIndex(index, index_name, info['table'], info['columns'], is_include, info.get('includes', []), info.get('cluster', ''))
                    candidate_index.whatif_queries.append(query_id)
                    all_candidates[index] = candidate_index
    # Add what-if queries that is not related with candidate generation (effective queries)
    for index, candidate in all_candidates.items():
        if index in existing_candidates:
            for query_id in existing_candidates[index].whatif_queries:
                if query_id in workload and query_id not in candidate.whatif_queries:
                    candidate.whatif_queries.append(query_id)
        else:
            existing_candidates[index] = candidate
    return all_candidates


def detect_error_nodes_4(selected_ids, candidates, actual_queries, whatif_queries, database_connector, workload):
    """
    Error node detection
    Detect both what-if errors and effective query errors.
    """
    error_nodes = {}
    for selected_id in selected_ids:
        candidate = candidates[selected_id]
        for query_id in candidate.whatif_queries:
            whatif_id = str(query_id) + '_' + candidate.id
            whatif_query = whatif_queries[whatif_id]
            whatif_cost = whatif_query.cost
            flag = False
            for effective_id in candidate.effective_queries:
                if query_id == actual_queries[effective_id].query_id:
                    actual_query = actual_queries[effective_id]
                    flag = True
                    break
            if not flag:
                actual_query = actual_queries[str(query_id) + '_']
            origin_query = actual_queries[str(whatif_query.query_id) + '_']
            origin_cost = origin_query.plan['Total Cost']
            origin_time = origin_query.time
            actual_time = actual_query.time
            actual_improvement = (origin_time - actual_time) / origin_time
            estimated_improvement = (origin_cost - whatif_cost) / origin_cost
            # if abs(actual_improvement - estimated_improvement) > 0.1:
            origin_plan = origin_query.plan
            whatif_plan = whatif_query.plan
            ind_plan = actual_query.plan
            origin_tree = query_plan_node.build_query_plan_tree(origin_plan)
            whatif_tree = query_plan_node.build_query_plan_tree(whatif_plan)
            ind_tree = query_plan_node.build_query_plan_tree(ind_plan)
            inaccurate_nodes, multi, refined_cost = query_plan_node.find_inaccurate_node_8(origin_tree, whatif_tree, ind_tree, candidates[selected_id].table, candidates[selected_id].columns)
            error_node = ErrorNode(selected_id, whatif_query.query_id, multi)
            if inaccurate_nodes:
                error_nodes[whatif_query.id] = {"whatif_query": whatif_query,
                "whatif_node":inaccurate_nodes[1].to_json(),
                    "multi": multi,
                    "new_estimation": refined_cost,
                    "error_node": error_node}
                logging.info(f"Error detected in whatif query {whatif_query.id} with error node {inaccurate_nodes[1].to_json()} \n multi: {multi} \n refined cost:{refined_cost}")
        # For effective queries not in what-if queries
        for effective_id in candidate.effective_queries:
            actual_query = actual_queries[effective_id]
            actual_query_id = actual_query.query_id
            if actual_query_id not in candidate.whatif_queries:
                whatif_id = str(actual_query_id) + '_' + candidate.id
                whatif_plan = database_connector.get_ind_plan(workload[actual_query_id].query_string, [candidate.id], "hypo")
                cost = whatif_plan['Total Cost']
                whatif_query = WhatIfQuery(actual_query_id, whatif_plan,cost, cost, [candidate.id])
                whatif_queries[whatif_id] = whatif_query
                candidate.whatif_queries.append(actual_query_id)
                logging.info(f"Query {actual_query_id} is not in what-if queries in index {selected_id}, but in effective queries. Adding it to what-if queries.")
                origin_query = actual_queries[str(whatif_query.query_id) + '_']
                origin_cost = origin_query.plan['Total Cost']
                origin_time = origin_query.time
                actual_time = actual_query.time
                actual_improvement = (origin_time - actual_time) / origin_time
                estimated_improvement = (origin_cost - whatif_cost) / origin_cost
                # if abs(actual_improvement - estimated_improvement) > 0.1:
                origin_plan = origin_query.plan
                whatif_plan = whatif_query.plan
                ind_plan = actual_query.plan
                origin_tree = query_plan_node.build_query_plan_tree(origin_plan)
                whatif_tree = query_plan_node.build_query_plan_tree(whatif_plan)
                ind_tree = query_plan_node.build_query_plan_tree(ind_plan)
                inaccurate_nodes, multi, refined_cost = query_plan_node.find_inaccurate_node_8(origin_tree, whatif_tree, ind_tree, candidates[selected_id].table, candidates[selected_id].columns)
                error_node = ErrorNode(selected_id, whatif_query.query_id, multi)
                if inaccurate_nodes:
                    error_nodes[whatif_query.id] = {"whatif_query": whatif_query,
                    "whatif_node":inaccurate_nodes[1].to_json(),
                        "multi": multi,
                        "new_estimation": refined_cost,
                        "error_node": error_node}
                    logging.info(f"Error detected in whatif query {whatif_query.id} with error node {inaccurate_nodes[1].to_json()} \n multi: {multi} \n refined cost:{refined_cost}")
    return error_nodes



def update_model(model_path, error_nodes, parameters, candidates):
    update_vecs = []
    update_labels = []
    candidate_name_2_info = {candidate.id: {'columns': candidate.columns,'table': candidate.table, 'includes':candidate.includes} for candidate in candidates.values()}
    multi_2_label = {label: index for index, label in enumerate(parameters.classes)}
    for what_if_query_id, info in error_nodes.items():
        query_id = what_if_query_id.split('_',1)[0]
        candidate_index = what_if_query_id.split('_',1)[1]
        what_if_plan = info['whatif_query'].plan
        what_if_tree = query_plan_node.build_query_plan_tree(what_if_plan)
        nodes = query_plan_node.feature_extractor(what_if_tree, parameters)
        error_node = info['whatif_node']
        error_tree = query_plan_node.build_query_plan_tree(error_node)
        error_node = query_plan_node.feature_extractor(error_tree, parameters)[0]
        multi = info['multi']
        for node in nodes:
            if node.node_type == error_node.node_type and node.index_name == error_node.index_name:
                operator_vec, extrainfo_vec, condition1, condition2, hash_condition = query_plan_node.node_encoding(node, parameters, candidate_name_2_info)
            
                operator_vec = torch.tensor(operator_vec, dtype=torch.float32)
                extrainfo_vec = torch.tensor(extrainfo_vec, dtype=torch.float32).view(-1)
                condition1 = torch.tensor(condition1, dtype=torch.float32).view(-1)
                # condition1_list = condition1.detach().numpy().tolist()
                # outputs = []
                # for index, element in enumerate(condition1_list):
                #     if element != 0:
                #         outputs.append(index)
                # output_results[query_index] = {
                #     "condition1":outputs}
                condition2 = torch.tensor(condition2, dtype=torch.float32).view(-1)
                update_vecs.append([
                    operator_vec,
                    extrainfo_vec,
                    condition1,
                    condition2,
                ])
                update_labels.append(multi_2_label[multi])
                break
    query_plan_node.update_model(model_path, parameters, update_vecs, update_labels, save_path=model_path, num_epochs=200)


def update_model_with_replaybuffer_4(model_paths, error_nodes, parameters, candidates, replaybuffers):
    """
    Update model with replay buffer
    model: operator-level model
    """
    candidate_name_2_info = {candidate.id: {'columns': candidate.columns,'table': candidate.table, 'includes':candidate.includes} for candidate in candidates.values()}
    multi_2_label = {label: index for index, label in enumerate(parameters.classes)}
    label_2_multi = {index: label for index, label in enumerate(parameters.classes)}
    vectors = {operator_type:[] for operator_type in parameters.operator_model_types}
    labels = {operator_type:[] for operator_type in parameters.operator_model_types}
    errors = {operator_type:[] for operator_type in parameters.operator_model_types}
    error_nodes_by_operator = {operator_type:[] for operator_type in parameters.operator_model_types}
    candidates_by_operator = {operator_type:{} for operator_type in parameters.operator_model_types}
    labels_of_candidates = {operator_type:{} for operator_type in parameters.operator_model_types}
    for what_if_query_id, info in error_nodes.items():
        query_id = what_if_query_id.split('_',1)[0]
        candidate_index = what_if_query_id.split('_',1)[1]
        what_if_plan = info['whatif_query'].plan
        what_if_tree = query_plan_node.build_query_plan_tree(what_if_plan)
        nodes = query_plan_node.feature_extractor(what_if_tree, parameters)
        error_node = info['whatif_node']
        error_tree = query_plan_node.build_query_plan_tree(error_node)
        error_node = query_plan_node.feature_extractor(error_tree, parameters)[0]
        multi = info['multi']
        for node in nodes:
            if node.node_type == error_node.node_type and node.index_name == error_node.index_name:
                operator_vec, extrainfo_vec, condition1, condition2, hash_condition = query_plan_node.node_encoding(node, parameters, candidate_name_2_info)
                
            
                operator_vec = torch.tensor(operator_vec, dtype=torch.float32)
                extrainfo_vec = torch.tensor(extrainfo_vec, dtype=torch.float32).view(-1)
                condition1 = torch.tensor(condition1, dtype=torch.float32).view(-1)
      
                condition2 = torch.tensor(condition2, dtype=torch.float32).view(-1)
                # check conflict
                conflict_found = False
                if node.index_name not in candidates_by_operator[node.node_type]:
                    candidates_by_operator[node.node_type][node.index_name] = [[operator_vec,extrainfo_vec,
                    condition1,
                    condition2]]
                    labels_of_candidates[node.node_type][node.index_name] = [multi_2_label[multi]]
                else:
                    existing_vecs = vectors[node.node_type]
                    existing_labels = labels[node.node_type]
                    for i, (existing_vector, existing_label) in enumerate(zip(existing_vecs, existing_labels)):
                        # Check if the combination of vectors and label matches an existing entry
                        if (torch.equal(existing_vector[0], operator_vec) and
                            torch.equal(existing_vector[1], extrainfo_vec) and
                            torch.equal(existing_vector[2], condition1) and
                            torch.equal(existing_vector[3], condition2)):

                            conflict_found = True

                            # Handle the conflict by updating the label with the smaller one
                            current_label = multi_2_label[multi]
                            if abs(1-multi) < abs(1-label_2_multi[existing_label]):
                                # Update the label to the smaller one
                                labels[node.node_type][i] = current_label
                                errors[node.node_type][i].cam = multi
                                logging.info(f"Conflict detected for candidate {node.index_name}, label updated to {current_label} at index {i}")
                            break
                if not conflict_found:
                    vectors[node.node_type].append([
                        operator_vec,
                        extrainfo_vec,
                        condition1,
                        condition2,
                    ])
                    labels[node.node_type].append(multi_2_label[multi])
                    errors[node.node_type].append(info['error_node'])
                    error_nodes_by_operator[node.node_type].append(what_if_query_id)
                break
    for operator_type in parameters.operator_model_types:
        if len(vectors[operator_type]) != 0:

            query_plan_node.update_model_with_replaybuffer_3(model_paths[operator_type], parameters, errors[operator_type], vectors[operator_type], labels[operator_type], replaybuffers[operator_type], save_path=model_paths[operator_type], num_epochs=400)
            accuracy, entrophy, mismatch_info = query_plan_node.evaluate_estimation_model_2(model_paths[operator_type],  vectors[operator_type], labels[operator_type], parameters)
            for error in mismatch_info:
                logging.info(f"Error index: {error_nodes_by_operator[operator_type][error['index']]}")






def cost_estimation_4(model_paths, whatif_queries, actual_queries, candidates):
    """
    Cost estimation by candidate index.
    model: operator-level model
    """
    candidate_name_2_info = {candidate.index_name:{'columns': candidate.columns, 'table': candidate.table, 'includes': candidate.includes} for candidate in candidates.values()}
    plans_to_estimate = []
    for whatif_id, whatif_query in whatif_queries.items():
        # if 'supplier#s_name' in whatif_id:
        plans_to_estimate.append(whatif_query.plan)
    estimation_results = query_plan_node.plan_estimation_test_7(plans_to_estimate, model_paths, candidate_indexes=candidate_name_2_info)
    total_value = 0
    for actual_query in actual_queries.values():
        total_value += actual_query.plan['Total Cost']
    for i in range(len(estimation_results)):
        whatif_query = list(whatif_queries.values())[i]
        whatif_query.refined_cost = estimation_results[i][0]
        query_id = whatif_query.query_id
        origin_query = actual_queries[str(query_id) + '_']
        for candidate_id in whatif_query.index_ids:
            candidates[candidate_id].estimated_reward += origin_query.plan['Total Cost'] - whatif_query.refined_cost
            candidates[candidate_id].uncertainty = estimation_results[i][1]
            candidates[candidate_id].potential += origin_query.time 
            candidates[candidate_id].estimated_rewards[query_id] = origin_query.plan['Total Cost'] - whatif_query.refined_cost
    for candidate in candidates.values():
        candidate.estimated_reward_2 = 10 * candidate.estimated_reward / total_value
    return candidates


def update_indexes(old_ids, new_ids, db_connector):
    """
    Update indexes from old_ids to new_ids
    """
    ids_to_drop = []
    ids_to_create = []
    for index_id in old_ids:
        if index_id not in new_ids:
            ids_to_drop.append(index_id)
    for index_id in new_ids:
        if index_id not in old_ids:
            ids_to_create.append(index_id)
    logging.info(f"indexes to create: {ids_to_create}")
    db_connector.drop_chosen_indexes(ids_to_drop)
    db_connector.create_indexes(ids_to_create, mode='actual')


def execution(old_ids, selected_ids, db_connector, workload):
    """
    Create selected indexes and execute workload.
    """
    actual_query_list = []
    # Update indexes 
    start_time = time.time()   
    update_indexes(old_ids, selected_ids, db_connector)
    execution_time = time.time() - start_time
    logging.info(f"Index update finished with time {execution_time}")
    # Execute workload
    for query_id, query in workload.items():
        result, plan = db_connector.exec_query_txt(query.query_string)
        actual_query = ActualQuery(query_id, plan, result)
        actual_query.get_effective_indexes()
        actual_query_list.append(actual_query)
        logging.info(f"Query {query_id} executed with indexes {actual_query.index_names} and time {actual_query.time}")
    return actual_query_list
        
    
def greedy_search_indexes(workload, actual_queries, candidates, budget, model_path, whatif_queries, db_connector, lam = 0.5, epsilon = 0.5, policy = 'greedy'):
    """
    Greedy search indexes based on workload and candidates
    """
    selected_ids = []
    current_time = 0
    workload_size = len(workload)
    remaining_budget = min(budget, len(candidates))
    candidate_name_2_info = {candidate.index_name:{'columns': candidate.columns, 'table': candidate.table, 'includes': candidate.includes} for candidate in candidates.values()}
    while remaining_budget > 0:
        current_candidates_configs = []
        whatif_query_list = []
        for index in candidates:
            if index not in selected_ids:
                config_ids = selected_ids + [index]
                config = IndexConfig(config_ids)
                current_candidates_configs.append(config)
                for query in workload.values():
                    whatif_id = str(query.id) + '_' + '_'.join(config.index_ids)
                    if whatif_id not in whatif_queries:
                        whatif_plan = db_connector.get_ind_plan(query.query_string, config.index_ids, "hypo")
                        cost = whatif_plan['Total Cost']
                        whatif_query = WhatIfQuery(query.id, whatif_plan, cost, cost, config.index_ids)
                        whatif_queries[whatif_id] = whatif_query
                        # current_whatif_queries[whatif_id] = whatif_query
                        whatif_query_list.append(whatif_query)
                    else:
                        whatif_query_list.append(whatif_queries[whatif_id])
        estimation_results = query_plan_node.plan_estimation_test_5([whatif_query.plan for whatif_query in whatif_query_list], model_path, candidate_indexes=candidate_name_2_info)
        for i, config in enumerate(current_candidates_configs):
            for j in range(workload_size):
                query_id = list(workload.keys())[j]
                origin_query = actual_queries[str(query_id) + '_']
                reward = origin_query.plan['Total Cost'] - estimation_results[i*workload_size + j][0]
                uncertainty = estimation_results[i*workload_size + j][2]
                config.reward += reward
                # config.total_value += (1+ lam * dropout) * reward
                # config.uncertainty += dropout
        
        choice = random.random()
        if random.random() < epsilon:
            best_config = random.choice(current_candidates_configs)
        else:
            # select config in current candidates with the highest total value
            best_config = max(current_candidates_configs, key=lambda x: x.reward)
            # best_config = current_candidates[estimation_results.index(max(estimation_results))]
        logging.info(f"Selected indexes: {best_config.index_ids} with reward {best_config.reward} and choice {choice}, epsilon {epsilon}")
        selected_ids = best_config.index_ids
        remaining_budget -= 1
    return selected_ids
        

def two_phase_greedy(workload, actual_queries, candidates, budget, model_path, whatif_queries,  db_connector, lam = 0.5, epsilon = 0.5, policy = 'greedy'):
    """
    Two-phase greedy search for index tuning.
    Phase 1: Optimize indexes per query.
    Phase 2: Optimize for the entire workload.
    """
    logging.info(f"Two-phase greedy search started with policy {policy} and epsilon {epsilon}")
    query_candidates = {query_id: {} for query_id in workload.keys()}
    refined_candidates = {}
    for key, candidate in candidates.items():
        for query_id in candidate.whatif_queries:
            query_candidates[query_id][key] = candidate
    # Phase 1: per-query greedy search
    for query in workload.values():
       
        query_best_indexes = greedy_search_indexes({query.id:query}, actual_queries, query_candidates[query.id], budget, model_path, whatif_queries, db_connector, lam, epsilon, policy)
        for id in query_best_indexes:
            if id not in refined_candidates:
                refined_candidates[id] = candidate
    # Phase 2: Global optimization using refined candidates
    final_indexes = greedy_search_indexes(workload, actual_queries, refined_candidates, budget, model_path, whatif_queries, db_connector, lam, epsilon, policy)
    return final_indexes


  

def removed_covered_tables(candidate_rewards, candidates, chosen_id, table_count):
        """
        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :param table_count: count of indexes already chosen for each table
        :return: reduced arm list
        """
        reduced_candidate_rewards = {}
        for index_id in candidate_rewards.keys():
            chosen_table = candidates[chosen_id].table
            candidate_table = candidates[index_id].table
            if not (chosen_table == candidate_table and table_count[chosen_table] >= MAX_INDEXES_PER_TABLE):
                reduced_candidate_rewards[index_id] = candidate_rewards[index_id]
        return reduced_candidate_rewards

def removed_covered_clusters(candidate_rewards, candidates, chosen_id):
    reduced_candidate_rewards = {}
    for index_id in candidate_rewards.keys():
       chosen_table = candidates[chosen_id].table
       candidate_table = candidates[index_id].table
       if not (chosen_table == candidate_table and candidates[index_id].cluster and candidates[chosen_id].cluster and candidates[chosen_id].cluster == candidates[index_id].cluster):
            reduced_candidate_rewards[index_id] = candidate_rewards[index_id]
    return reduced_candidate_rewards


def removed_covered_queries(candidate_rewards, candidates, chosen_id):
    # When covering index is selected for a query we gonna remove all other arms from that query
    if candidates[chosen_id].is_include == 0:
        return candidate_rewards
    reduced_candidates_rewards = {}
    for index_id in candidate_rewards.keys():
        covered_query_ids = candidates[chosen_id].whatif_queries.keys()
        chosen_table = candidates[chosen_id].table
        candidate_table = candidates[index_id].table
        if chosen_table == candidate_table:
            for query_id in candidates[index_id].whatif_queries.keys():
                if query_id not in covered_query_ids:
                    reduced_candidates_rewards[index_id] = candidate_rewards[index_id]
                    break
        else:
            reduced_candidates_rewards[index_id] = candidate_rewards[index_id]
    return reduced_candidates_rewards
        

def update_existing_candidates(candidates, existing_candidates):
    for id, candidate in candidates.items():
        existing_candidates[id] = candidate


def index_selection_9(config):
    """
    Take index create time into account. log out index creation time for each index. do not duplicate index creation, but enable/disable existing indexes.
    add ORDERBY into consideration.
    """
    # Preparation
    # EXP_ID = "tpch_session"
    EXP_ID = config['exp_id']
    db = config['db']
    epsilon = config['epsilon']
    lam0 = config['lam0']
    decay_rate = config['decay_rate']
    index_config = config['index_config']
    workload_configs = config['workload_configs']
    t0 = config['t0']
    boltz_decay_rate = config['boltz_decay_rate']
    max_explore = config['max_explore_rounds']
    directory = f"{parent_dir}/records/{EXP_ID}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    if db == 'tpcds':
        db_conf = {"postgresql":{
            "database":'tpcds',
            "host":'localhost',
            "port":'5434',
            "user":'XXX',
            "password":'XXX'
        }}
        query_workload_path = f"{parent_dir}/workload_ds.json"
        candidate_workload_path = f"{parent_dir}/ds_static_100_filtered.json"
        schema_path = f"{parent_dir}/tpcds_schema.json"
        parameter_path = f"{parent_dir}/records/parameters_tpcds.json"
    elif db == 'tpch':
        db_conf = {"postgresql":{
            "database":'tpch',
            "host":'localhost',
            "port":'5434',
            "user":'XXX',
            "password":'XXX'
        }}
        candidate_workload_path = f"{parent_dir}/tpc_h_static_100_pg.json"
        query_workload_path = f"{parent_dir}/workload.json"
        schema_path = f"{parent_dir}/tpch_schema.json"
        parameter_path = f"{parent_dir}/node_estimation/parameters.json"
    elif db == 'job':
        db_conf = {"postgresql":{
            "database":'imdb',
            "host":'localhost',
            "port":'5434',
            "user":'XXX',
            "password":'XXX'
        }}
        candidate_workload_path = f"{parent_dir}/job_static.json"
        parameter_path = f"{parent_dir}/parameters_job.json"
    database_connector = PostgresDatabaseConnector(db_conf, autocommit=True)
    round_num = len(workload_configs)  
    query_plans = {}
    candidates = {}
    # Load parameters
    with open(parameter_path, 'r') as file:
        data = json.load(file)
    parameters = query_plan_node.Parameters.from_dict(data)
    multi_2_label = {label: index for index, label in enumerate(parameters.classes)}
    model_paths = {operator_type: directory + f'model_{operator_type}.pth' for operator_type in parameters.operator_model_types}
    # Set logging
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = directory + 'logging.log'
    logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path),  # Log to the unique file
    ]
)
    # Load workload
    workload = {}
    with open(candidate_workload_path, 'r') as f:
        line = f.readline()
        while line:
            query_info = json.loads(line)
            query = Query(int(query_info['id']), query_info['query_string'], query_info['predicates'], query_info['payload'])
            workload[query.id] = query
            line = f.readline()
    
    actual_queries_records = {}
    whatif_queries_records = {}
    old_ids = []
    seen_query_ids = []

    replay_buffers = {operator_type: None for operator_type in parameters.operator_model_types} 
    # Execute queries to get original cost  
    logging.info(f"Initial execution started")
    if config['rerun']:
        execution_results = {}
        for query in workload.values():
            actual_id = str(query.id) + '_'
            database_connector.exec_query_txt(query.query_string, cost_evaluation=True)
            time, plan = database_connector.exec_query_txt(query.query_string, cost_evaluation=True)
            actual_query = ActualQuery(query.id, plan, time)
            actual_queries_records[actual_id] = actual_query
            execution_results[query.id] = {'time': time, 'plan': plan}
            logging.info(f"Query {query.id} executed with time {actual_queries_records[actual_id].time}")
        with open(config['workload_execution_path'], 'w') as f:
            json.dump(execution_results, f)
        logging.info(f"Initial execution finished")
    else:
        with open(config['workload_execution_path'], 'r') as f:
            execution_results = json.load(f)
            for query_id, result in execution_results.items():
                actual_query = ActualQuery(query_id, result['plan'], result['time'])
                actual_queries_records[str(query_id) + '_'] = actual_query
                logging.info(f"Query {query_id} executed with time {actual_queries_records[str(query_id) + '_'].time}")

    existing_candidates = {}
    sota_records = {}
    created_index_dict = {}
    for round in range(round_num):
        logging.debug(f'Round {round} started')
        current_actual_queries = {}
        current_whatif_queries = {}
        # load workload
        workload_config = workload_configs[round]
        current_workload = {}
        origin_total_time = 0
        unseen_query_num = 0
        for query_idx in range(workload_config[0], workload_config[1] + 1):
            query_idx = query_idx % len(workload)
            query_id = list(workload.keys())[query_idx]
            if query_id not in seen_query_ids:
                seen_query_ids.append(query_id)
                unseen_query_num += 1
            current_workload[query_id] = workload[query_id]
            current_actual_queries[str(query_id) + '_'] = actual_queries_records[str(query_id) + '_']
            origin_total_time += current_actual_queries[str(query_id) + '_'].time
        logging.info(f"Round {round} workload initial time: {origin_total_time}")
  
        origin_total_time = 0
        for query in current_workload.values():
            actual_id = str(query.id) + '_'
            if actual_id not in actual_queries_records:
                database_connector.exec_query_txt(query.query_string, cost_evaluation=True)
                time, plan = database_connector.exec_query_txt(query.query_string, cost_evaluation=True)
                actual_query = ActualQuery(query.id, plan, time)
                actual_queries_records[actual_id] = actual_query
            current_actual_queries[actual_id] = actual_queries_records[actual_id]
            origin_total_time += current_actual_queries[actual_id].time
            logging.info(f"Query {query.id} executed with time {actual_queries_records[actual_id].time}")
        logging.info(f"Workload initial time: {origin_total_time}")

        # Generate candidates
        candidates = gen_candidates_3(current_workload,existing_candidates)
        candidate_name_2_id = {candidate.index_name: candidate.id for candidate in candidates.values()}
        for candidate in candidates.values():
            if candidate.estimated_size == 0:
                candidate.estimated_size = database_connector.estimate_index_size(candidate.id, "hypo") / 1024.0 / 1024.0
                # logging.info(f"Index {candidate.id} estimated size: {candidate.estimated_size}")
            for query_id in candidate.whatif_queries:
                whatif_id = str(query_id) + '_' + candidate.id
                if whatif_id not in whatif_queries_records:
                    whatif_plan = database_connector.get_ind_plan(workload[query_id].query_string, [candidate.id], "hypo")
                    cost = whatif_plan['Total Cost']
                    whatif_query = WhatIfQuery(query_id, whatif_plan,cost, cost, [candidate.id])
                    whatif_queries_records[whatif_id] = whatif_query
                current_whatif_queries[whatif_id] = whatif_queries_records[whatif_id]
        if round == 0:
            replay_capacity = len(candidates)
            for key in replay_buffers.keys():
                replay_buffers[key] = query_plan_node.ReplayBuffer(replay_capacity)

        # Estimation
        candidates = cost_estimation_4(model_paths, current_whatif_queries, current_actual_queries, candidates, parameters)
        # Candidate enumeration
        epsilon = epsilon * decay_rate
        t = (int)((1-unseen_query_num/len(current_workload)) * round)
        lam = lam0 * (decay_rate ** t)
        temperature = t0 * np.exp(-boltz_decay_rate * t)
        if workload_config in sota_records and sota_records[workload_config]['explore_times'] >= max_explore:
            selected_ids = sota_records[workload_config]['indexes']
        else:
            selected_ids = candidate_enumeration_3({index: candidate.estimated_reward for index, candidate in candidates.items()}, candidates, policy=config['selection_policy'], epsilon=epsilon, index_config=index_config, lam = lam, temperature=temperature,budget_type=config['budget_type'],storage_budget=config['storage_budget'])
        # selected_ids = ["lineitem#l_partkey,l_suppkey"]
        # Execution
        # selected_ids = ['lineitem#l_partkey,l_suppkey']
        actual_query_list, index_sizes = execution(old_ids, selected_ids, database_connector, current_workload)
        # actual_query_list, created_index_dict, index_sizes = execution_2(old_ids, selected_ids, database_connector, current_workload, created_index_dict)
        for index_key in index_sizes:
            candidates[index_key].estimated_size = index_sizes[index_key] / 1024.0 / 1024.0
        total_execution_time = 0
        for actual_query in actual_query_list:
            total_execution_time += actual_query.time
            current_actual_queries[actual_query.id] = actual_query
            for index_name in actual_query.index_names:
                if index_name in candidate_name_2_id:
                    candidate_id = candidate_name_2_id[index_name]
                    candidates[candidate_id].effective_queries.append(actual_query.id)
                    candidates[candidate_id].actual_reward += current_actual_queries[str(actual_query.query_id) + '_'].time - actual_query.time
        logging.info(f"Total execution time: {total_execution_time}")
        if workload_config not in sota_records:
            sota_records[workload_config] = {'total_time': total_execution_time, 'indexes': selected_ids, 'explore_times':1}
        else:
            if total_execution_time < sota_records[workload_config]['total_time']:
                sota_records[workload_config]['total_time'] = total_execution_time
                sota_records[workload_config]['indexes'] = selected_ids 
            sota_records[workload_config]['explore_times'] += 1
        # Detect Errors
        error_nodes = detect_error_nodes_4(selected_ids, candidates, current_actual_queries, current_whatif_queries, database_connector, current_workload, parameters)
                
        # Update model
        if error_nodes:
            update_model_with_replaybuffer_4(model_paths, error_nodes, parameters, candidates, replaybuffers=replay_buffers)
        
        old_ids = selected_ids

        update_existing_candidates(candidates, existing_candidates)
        

        logging.info(f"Round {round} finished")




def boltzman_sample(candidate_rewards, temperature = 0.5):
    """
    Boltzman Sampling
    """
    rewards = np.array([values['total_value'] for values in candidate_rewards.values()])
    exp_rewards = np.exp(rewards / temperature)
    probs = exp_rewards / np.sum(exp_rewards)
    selected_index = np.random.choice(list(candidate_rewards.keys()), p=probs)
    logging.info(f"Selected index: {selected_index} with reward {candidate_rewards[selected_index]} and probability {probs[list(candidate_rewards.keys()).index(selected_index)]}")
    return selected_index


def candidate_enumeration_3(candidate_rewards, candidates, policy = 'UCB', epsilon = 0.5, index_config = 5, lam = 0.5, temperature = 0.5):
    """
    Select indexes based on reward with Boltzman & temperature
    explore_value = confidence_interval + benefit_improvement
    """
    alpha = 0.5
    rewards = []
    explore_values = []
    for candidate_key in candidate_rewards:
        candidate = candidates[candidate_key]
        total_value = (1+ lam * candidate.uncertainty) * candidate.estimated_reward
        total_value_2 = (1+ lam * candidate.uncertainty) * candidate.estimated_reward_2
        # explore_value = alpha * candidate.uncertainty + (1-alpha) * candidate.potential
        # reward = candidate.estimated_reward + lam * explore_value
        candidate_rewards[candidate_key] = {'estimated_reward': candidate.estimated_reward, 'estimated_reward_2': candidate.estimated_reward_2, 'explore_value': total_value_2, 'total_value': total_value_2, 'uncertainty': candidate.uncertainty, 'lam': lam, 'estimated_rewards': candidate.estimated_rewards}
        # rewards.append(reward)
        # explore_values.append(explore_value)
    
    # if policy == 'UCB':
    #     rewards = [candidate.estimated_reward + lam * explore_values[i] for i, candidate in enumerate(candidates.values())]

    table_count = {}
    selected_indexes = []
    logging.info(f"temperature is {temperature}")
    for i in range(index_config):
        if policy == 'UCB':
            # candidate_keys = list(candidate_rewards.keys())
            selected_index = max(candidate_rewards, key=lambda index: candidate_rewards[index]['explore_value'])
            logging.info(f"Selected index: {selected_index} with reward {candidate_rewards[selected_index]} ")
        elif policy == 'Boltzman':
            selected_index = boltzman_sample(candidate_rewards, temperature)
        else: 
            choice = random.random()
            if choice < epsilon:
                # Select a random index with probability epsilon
                selected_index = random.choice(list(candidate_rewards.keys()))
            else:
                # Select the best index with probability 1 - epsilon
                selected_index = max(candidate_rewards, key=lambda index: candidate_rewards[index]['estimated_reward'])
            logging.info(f"Selected index: {selected_index} with estimated reward {candidate_rewards[selected_index]['estimated_reward']} and choice {choice}, epsilon {epsilon}")
                # selected_index = max(candidate_rewards, key=lambda index: candidate_rewards[index])
        selected_indexes.append(selected_index)
        
        
        # Update table count
        table = candidates[selected_index].table
        if table in table_count:
            table_count[table] += 1
        else:
            table_count[table] = 1
        del candidate_rewards[selected_index]
        candidate_rewards = removed_covered_tables(candidate_rewards, candidates, selected_index, table_count)
        candidate_rewards = removed_covered_clusters(candidate_rewards, candidates, selected_index)
        candidate_rewards = removed_covered_queries(candidate_rewards, candidates, selected_index)

    return selected_indexes



def parse():
    parser = argparse.ArgumentParser(description='Index Selection')
    parser.add_argument('--selection_policy', type=str, default='Boltzman')
    parser.add_argument('--exp_id', type=str, default='tpch_test')
    parser.add_argument('--workload_configs', type=str, default='[(10,20),(10,20),(10,20)]')
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.9)
    parser.add_argument('--db', type=str, default='tpch')
    parser.add_argument('--index_config', type=int, default=3)
    parser.add_argument('--rerun', type=str, default='False')
    parser.add_argument('--workload_execution_path', type=str, default=f'{PARENT_PATH}/records/tpch_workload_results.json')
    parser.add_argument('--lam0', type=float, default=0.5)
    parser.add_argument('--t0', type=float, default=1.0)
    parser.add_argument('--boltz_decay_rate', type=float, default=0.1)
    parser.add_argument('--max_explore_rounds', type=int, default=12)
    args = parser.parse_args()
    config = {
        'exp_id': args.exp_id,
        'workload_configs': ast.literal_eval(args.workload_configs),
        'epsilon': args.epsilon,
        'decay_rate': args.decay_rate,
        'db': args.db,
        'index_config': args.index_config,
        'rerun': True if args.rerun == 'True' else False,   
        'workload_execution_path': args.workload_execution_path,
        'selection_policy': args.selection_policy,
        'lam0':args.lam0,
        't0':args.t0,
        'boltz_decay_rate': args.boltz_decay_rate,
        'max_explore_rounds': args.max_explore_rounds
    }
    return config
    
    

    
if __name__ == '__main__':
    config = parse()
    exp_id = config['exp_id']
    directory = f'{PARENT_PATH}/records/' + exp_id + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_path = directory + 'logging.log'
    logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='a'),  # Log to the unique file
    ]
)
    clear_indexes()
    index_selection_9(config)
    # test_update()
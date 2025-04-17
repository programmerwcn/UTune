import sys
import os

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from node_operations import *
from predicate_features import *
import numpy as np

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

import json
import psycopg2
from decimal import Decimal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import node_estimation
import torch.nn.functional as F
import logging
import random

parent_path = "xx"

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.error_nodes = []  # Store error nodes corresponding to experiences

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def is_conflict(self, error_node, existing_node):
        return error_node.index_id ==  existing_node.index_id and error_node.query_id == existing_node.query_id and error_node.cam != existing_node.cam
    
    def add_2(self, experience, error_node):
        # First check for conflict in the buffer
        for idx, exp in enumerate(self.buffer):
            if self.is_conflict(error_node, self.error_nodes[idx]):
                self.buffer[idx] = experience  # Replace the old one
                self.error_nodes[idx] = error_node  
                logging.info(f"Replaced experience of {self.error_nodes[idx].to_json()} with {error_node.to_json()}")
                return

        # No conflict found, do normal circular buffer logic
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.error_nodes.append(error_node)
        else:
            self.buffer[self.position] = experience
            self.error_nodes[self.position] = error_node
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
class QueryPlanNode:
    def __init__(self, node_type, parent = None, **kwargs):
        self.node_type = node_type
        self.attributes = kwargs
        self.children = []
        self.parent = parent

    def add_child(self, child_node):
        self.children.append(child_node)
    
    def to_json(self):
        return {
            "Node Type": self.node_type,
            **self.attributes,
            "Plans": [child.to_json() for child in self.children]
        }

    def __repr__(self, level=0):
        indent = "  " * level
        output = f"{indent}{self.node_type} "
        for key, value in self.attributes.items():
            output += f"\n{indent}  {key}: {value}"
        if self.children:
            output += "\n" + "\n".join([child.__repr__(level + 1) for child in self.children])
        return output
    

    def cal_cost(self, cost_type = 'Total Cost'):
        # if self.node_type == 'Index Scan' and self.attributes.get('Parent Relationship') == 'Inner' and self.parent.node_type == 'Nested Loop':
        #     # Look for outer relation
        #     for child in self.parent.children:
        #         if child.attributes.get('Parent Relationship',None) == 'Outer':
        #             outer_plan_rows = child.attributes.get('Plan Rows',0)
        #             outer_actual_rows = child.attributes.get('Actual Rows',0)
        #     # Calculate cost
        #     if cost_type == 'Total Cost':
        #         cost = outer_plan_rows * self.attributes.get(cost_type,0)
        #     else:
        #         cost = outer_actual_rows * self.attributes.get(cost_type,0)
        # else:
        cost = self.attributes.get(cost_type,0)
        return cost
    
    def cal_cost_2(self, cost_type = 'Total Cost'):
        # Computes the cost by considering the cost of scanning the outer relation and rescanning the inner relation for each outer tuple.
        if self.node_type == 'Nested Loop':
            # Look for outer relation
            for child in self.children:
                if child.attributes.get('Parent Relationship',None) == 'Outer':
                    outer_node = child
            # Look for inner relation
            for child in self.children:
                if child.attributes.get('Parent Relationship',None) == 'Inner':
                    inner_node = child
            # Calculate cost
            if cost_type == 'Total Cost':
                cost = inner_node.attributes.get(cost_type,0) * outer_node.attributes.get('Plan Rows') + outer_node.attributes.get(cost_type,0) 
            elif cost_type == 'Actual Total Time':
                cost = inner_node.attributes.get(cost_type,0) * outer_node.attributes.get('Actual Rows') + outer_node.attributes.get(cost_type,0)
            elif cost_type == 'Calculation Cost':
                cost = inner_node.cal_cost_2(cost_type) * outer_node.attributes.get('Plan Rows') + outer_node.cal_cost_2(cost_type)

    def search_node(self, node_type, attributes):
        if self.node_type == node_type:
            is_node = True
            for key, value in attributes.items():
                if key not in self.attributes or self.attributes.get(key) != value:
                    is_node = False
                    break
            if is_node:
                return self
        for child in self.children:
            result = child.search_node(node_type, attributes)
            if result:
                return result
        return None

    def update_cost(self, cost_type = 'Total Cost', mode = 'test'):
        """
        Update the cost of a node based on its children's cost.
        Nested loop = inner_cost * outer_rows + outer_cost
        Limit = startup_cost + child(total_cost - startup_cost) * parent(tuples to fetch) / child (rows)
        """
        new_cost_type = cost_type + ' New'

        # 2 child nodes
        if len(self.children) == 2:
            if self.node_type == 'Hash Join' or self.node_type == 'Nested Loop':
                # Look for outer relation
                for child in self.children:
                    if child.attributes.get('Parent Relationship',None) == 'Outer':
                        outer_node = child
                # Look for inner relation
                for child in self.children:
                    if child.attributes.get('Parent Relationship',None) == 'Inner':
                        inner_node = child
                origin_inner_cost = inner_node.attributes.get(cost_type,0)
                origin_outer_cost = outer_node.attributes.get(cost_type,0)
                # Nested Loop: cost = inner_cost * out_rows + outer_cost
                if self.node_type == 'Nested Loop':         
                    if cost_type == 'Total Cost':
                        outer_rows = outer_node.attributes.get('Plan Rows',0)
                    elif cost_type == 'Actual Total Time':
                        outer_rows = outer_node.attributes.get('Actual Rows',0)
                    new_cost = self.attributes.get(cost_type,0) + (inner_node.attributes.get(new_cost_type, origin_inner_cost) - origin_inner_cost) * outer_rows + (outer_node.attributes.get(new_cost_type, origin_outer_cost) - origin_outer_cost)
                elif self.node_type == 'Hash Join':
                    new_cost = self.attributes.get(cost_type,0) + inner_node.attributes.get(new_cost_type, origin_inner_cost) - origin_inner_cost + outer_node.attributes.get(new_cost_type,origin_outer_cost) - origin_outer_cost
            else:
                origin_child_cost_1 = self.children[0].attributes.get(cost_type,0)
                origin_child_cost_2 = self.children[1].attributes.get(cost_type,0)
                new_cost = self.attributes.get(cost_type,0) + self.children[0].attributes.get(new_cost_type, origin_child_cost_1) - origin_child_cost_1 + self.children[1].attributes.get(new_cost_type, origin_child_cost_2) - origin_child_cost_2
        # 1 child node
        elif len(self.children) == 1:
            child_node = self.children[0]
            origin_child_cost = child_node.attributes.get(cost_type,0)
            if self.node_type == 'Limit':
                if cost_type == 'Total Cost':
                    multi = float(self.attributes.get('Plan Rows', 1)) / child_node.attributes.get('Plan Rows', 1)
                elif cost_type == 'Actual Total Time':
                    multi = float(self.attributes.get('Actual Rows', 1)) / child_node.attributes.get('Actual Rows', 1)
                new_cost = self.attributes.get(cost_type,0) + (child_node.attributes.get(new_cost_type, origin_child_cost) - origin_child_cost) * multi
            else:
                new_cost = self.attributes.get(cost_type,0) + child_node.attributes.get(new_cost_type, origin_child_cost) - origin_child_cost
                #     new_cost = self.attributes.get(cost_type,0) + child_node.attributes.get(new_cost_type, origin_child_cost) - child_node.attributes.get('Startup Cost',0) * self.attributes.get('Tuples') / child_node.attributes.get('Plan Rows',0)
                # elif cost_type == 'Actual Total Time':
            #         new_cost = self.attributes.get(cost_type,0) + child_node.attributes.get(new_cost_type, origin_child_cost) - child_node.attributes.get('Startup Cost',0) * self.attributes.get('Tuples') / child_node.attributes.get('Actual Rows',0)
            # new_cost = self.attributes.get(cost_type,0) + child_node.attributes.get(new_cost_type, origin_child_cost) - origin_child_cost

        elif len(self.children) == 3:
            if self.node_type == 'Hash Join' or self.node_type == 'Nested Loop':
                # Look for outer relation & inner relation
                for child in self.children:
                    if child.attributes.get('Parent Relationship',None) == 'Outer':
                        outer_node = child
                    elif child.attributes.get('Parent Relationship',None) == 'Inner':
                        inner_node = child
                    else:
                        third_node = child
                origin_inner_cost = inner_node.attributes.get(cost_type,0)
                origin_outer_cost = outer_node.attributes.get(cost_type,0)
                origin_third_cost = third_node.attributes.get(cost_type,0)
                # Nested Loop: cost = inner_cost * out_rows + outer_cost
                if self.node_type == 'Nested Loop':         
                    if cost_type == 'Total Cost':
                        outer_rows = outer_node.attributes.get('Plan Rows',0)
                    elif cost_type == 'Actual Total Time':
                        outer_rows = outer_node.attributes.get('Actual Rows',0)
                    new_cost = self.attributes.get(cost_type,0) + (inner_node.attributes.get(new_cost_type, origin_inner_cost) - origin_inner_cost) * outer_rows + (outer_node.attributes.get(new_cost_type, origin_outer_cost) - origin_outer_cost) + (third_node.attributes.get(new_cost_type, origin_third_cost) - origin_third_cost)
                elif self.node_type == 'Hash Join':
                    new_cost = self.attributes.get(cost_type,0) + inner_node.attributes.get(new_cost_type, origin_inner_cost) - origin_inner_cost + outer_node.attributes.get(new_cost_type,origin_outer_cost) - origin_outer_cost + third_node.attributes.get(new_cost_type, origin_third_cost) - origin_third_cost
            else:
                origin_child_cost_1 = self.children[0].attributes.get(cost_type,0)
                origin_child_cost_2 = self.children[1].attributes.get(cost_type,0)
                origin_child_cost_3 = self.children[2].attributes.get(cost_type,0)
                new_cost = self.attributes.get(cost_type,0) + self.children[0].attributes.get(new_cost_type, origin_child_cost_1) - origin_child_cost_1 + self.children[1].attributes.get(new_cost_type, origin_child_cost_2) - origin_child_cost_2 + self.children[2].attributes.get(new_cost_type, origin_child_cost_3) - origin_child_cost_3

        if mode == 'apply':
            for child in self.children:
                if new_cost_type in child.attributes:
                    child.attributes[cost_type] = child.attributes[new_cost_type]

        self.attributes[new_cost_type] = new_cost

        if self.parent:
            self.parent.update_cost(cost_type = cost_type, mode = mode)

    def update_cost_2(self, cost, cost_type = 'Total Cost', mode = 'test'):
        """
        Update the cost of a node based on its children's cost.
        Update startup and endtime
        Nested loop = inner_cost * outer_rows + outer_cost
        Limit = startup_cost + child(total_cost - startup_cost) * parent(tuples to fetch) / child (rows)
        """
        if cost_type == 'Total Cost':
            origin_start = "Startup Cost"
            origin_end = "Total Cost"
            new_start = "Startup Cost New"
            new_end = "Total Cost New"
            tuples = "Plan Rows"
        elif cost_type == 'Actual Total Time':
            origin_start = "Actual Startup Time"
            origin_end = "Actual Total Time"
            new_start = "Actual Startup Time New"
            new_end = "Actual Total Time New"
            tuples = "Actual Rows"  
        origin_cost = self.attributes.get(origin_end,0) - self.attributes.get(origin_start,0)
        
        # Leaf node: calculate node cost
        if len(self.children) == 0:
            self.attributes[new_start] = self.attributes[origin_start]
            self.attributes["Node Cost"] = cost
            self.attributes[new_end] = self.attributes[new_start] + cost
        # 1 child node
        
        elif len(self.children) == 1:
            child_node = self.children[0]
            origin_child_cost = child_node.attributes.get(origin_end,0) - child_node.attributes.get(origin_start,0)
            new_child_cost = child_node.attributes.get(new_end, child_node.attributes[origin_end] ) - child_node.attributes.get(new_start, child_node.attributes[origin_start])
            if self.node_type in ['Hash', 'Sort']: # wait until the child node is finished
                self.attributes[new_start] = self.attributes[origin_start] + (child_node.attributes.get(new_end, child_node.attributes[origin_end]) - child_node.attributes[origin_end])
            else:
                self.attributes[new_start] = self.attributes[origin_start] + child_node.attributes.get(new_start, child_node.attributes[origin_start]) - child_node.attributes[origin_start]
            if self.node_type == 'Limit': # Limit node
                multi = float(self.attributes.get(tuples, 1)) / child_node.attributes.get(tuples, 1)
                new_cost = origin_cost + (new_child_cost - origin_child_cost) * multi
                self.attributes[new_end] = self.attributes[new_start] + new_cost
            else:
                new_cost =  origin_cost + new_child_cost - origin_child_cost
                self.attributes[new_end] = self.attributes[new_start] + new_cost
        # 2 child nodes
        elif len(self.children) == 2:
            if self.node_type == 'Nested Loop':
                for child in self.children:
                    if child.attributes.get('Parent Relationship',None) == 'Outer':
                        outer_node = child
                    elif child.attributes.get('Parent Relationship',None) == 'Inner':
                        inner_node = child
                origin_inner_cost = inner_node.attributes.get(origin_end,0) - inner_node.attributes.get(origin_start,0)
                origin_outer_cost = outer_node.attributes.get(origin_end,0) - outer_node.attributes.get(origin_start,0)
                new_inner_cost = inner_node.attributes.get(new_end, inner_node.attributes[origin_end]) - inner_node.attributes.get(new_start,inner_node.attributes[origin_start])
                new_outer_cost = outer_node.attributes.get(new_end, outer_node.attributes[origin_end]) - outer_node.attributes.get(new_start,outer_node.attributes[origin_start])
                self.attributes[new_start] = self.attributes.get(origin_start,0) + (outer_node.attributes.get(new_start,outer_node.attributes[origin_start]) + inner_node.attributes.get(new_start,inner_node.attributes[origin_start])- outer_node.attributes[origin_start] - inner_node.attributes[origin_start])
                if inner_node.node_type == 'Materialize':
                    new_cost = origin_cost + (new_outer_cost - origin_outer_cost) + new_inner_cost - origin_inner_cost
                else:
                    new_cost = origin_cost + (new_inner_cost - origin_inner_cost) * outer_node.attributes.get(tuples,1) + (new_outer_cost - origin_outer_cost)
                self.attributes[new_end] = self.attributes[new_start] + new_cost
            else:
                origin_child_cost_1 = self.children[0].attributes.get(origin_end,0) - self.children[0].attributes.get(origin_start,0)
                origin_child_cost_2 = self.children[1].attributes.get(origin_end,0) - self.children[1].attributes.get(origin_start,0)
                new_child_cost_1 = self.children[0].attributes.get(new_end, self.children[0].attributes[origin_end]) - self.children[0].attributes.get(new_start, self.children[0].attributes[origin_start])
                new_child_cost_2 = self.children[1].attributes.get(new_end, self.children[1].attributes[origin_end]) - self.children[1].attributes.get(new_start, self.children[1].attributes[origin_start])
                self.attributes[new_start] = self.attributes.get(origin_start, 0) + self.children[0].attributes.get(new_start, self.children[0].attributes[origin_start]) + self.children[1].attributes.get(new_start, self.children[1].attributes[origin_start]) - self.children[0].attributes.get(origin_start,0) - self.children[1].attributes.get(origin_start,0)
                new_cost = origin_cost + new_child_cost_1 - origin_child_cost_1 + new_child_cost_2 - origin_child_cost_2
                self.attributes[new_end] = self.attributes[new_start] + new_cost
        # 3 child nodes
        elif len(self.children) == 3:
            if self.node_type == 'Nested Loop':
                for child in self.children:
                    if child.attributes.get('Parent Relationship',None) == 'Outer':
                        outer_node = child
                    elif child.attributes.get('Parent Relationship',None) == 'Inner':
                        inner_node = child
                    else:
                        third_node = child
                origin_inner_cost = inner_node.attributes.get(origin_end,0) - inner_node.attributes.get(origin_start,0)
                origin_outer_cost = outer_node.attributes.get(origin_end,0) - outer_node.attributes.get(origin_start,0)
                origin_third_cost = third_node.attributes.get(origin_end,0) - third_node.attributes.get(origin_start,0)
                new_inner_cost = inner_node.attributes.get(new_end, inner_node.attributes[origin_end]) - inner_node.attributes.get(new_start,inner_node.attributes[origin_start])
                new_outer_cost = outer_node.attributes.get(new_end, outer_node.attributes[origin_end]) - outer_node.attributes.get(new_start,outer_node.attributes[origin_start])
                new_third_cost = third_node.attributes.get(new_end, third_node.attributes[origin_end]) - third_node.attributes.get(new_start,third_node.attributes[origin_start])
                self.attributes[new_start] = self.attributes[origin_start] + (outer_node.attributes.get(new_start,outer_node.attributes[origin_start]) + inner_node.attributes.get(new_start,inner_node.attributes[origin_start]) + third_node.attributes.get(new_start,third_node.attributes[origin_start]) - outer_node.attributes[origin_start] - inner_node.attributes[origin_start] - third_node.attributes[origin_start])
                if inner_node.node_type == 'Materialize':
                    new_cost = origin_cost + (new_outer_cost - origin_outer_cost) + new_inner_cost - origin_inner_cost + (new_third_cost - origin_third_cost)
                else:
                    new_cost = origin_cost + (new_inner_cost - origin_inner_cost) * outer_node.attributes.get(tuples,1) + (new_outer_cost - origin_outer_cost) + (new_third_cost - origin_third_cost)
                self.attributes[new_end] = self.attributes[new_start] + new_cost
            else:
                origin_child_cost_1 = self.children[0].attributes.get(origin_end,0) - self.children[0].attributes.get(origin_start,0)
                origin_child_cost_2 = self.children[1].attributes.get(origin_end,0) - self.children[1].attributes.get(origin_start,0)
                origin_child_cost_3 = self.children[2].attributes.get(origin_end,0) - self.children[2].attributes.get(origin_start,0)
                new_child_cost_1 = self.children[0].attributes.get(new_end, self.children[0].attributes[origin_end]) - self.children[0].attributes.get(new_start, self.children[0].attributes[origin_start])
                new_child_cost_2 = self.children[1].attributes.get(new_end, self.children[1].attributes[origin_end]) - self.children[1].attributes.get(new_start, self.children[1].attributes[origin_start])
                new_child_cost_3 = self.children[2].attributes.get(new_end, self.children[2].attributes[origin_end]) - self.children[2].attributes.get(new_start, self.children[2].attributes[origin_start])
                self.attributes[new_start] = self.attributes[origin_start] + self.children[0].attributes.get(new_start, self.children[0].attributes[origin_start]) + self.children[1].attributes.get(new_start, self.children[1].attributes[origin_start]) + self.children[2].attributes.get(new_start, self.children[2].attributes[origin_start]) - self.children[0].attributes.get(origin_start,0) - self.children[1].attributes.get(origin_start,0) - self.children[2].attributes.get(origin_start,0)
                new_cost = origin_cost + new_child_cost_1 - origin_child_cost_1 + new_child_cost_2 - origin_child_cost_2 + new_child_cost_3 - origin_child_cost_3
                self.attributes[new_end] = self.attributes[new_start] + new_cost
        # More than 3 child nodes
        else:
            new_start_cost = self.attributes[origin_start]
            new_cost = origin_cost
            new_end_cost = self.attributes[origin_end]
            for child in self.children:
                new_start_cost += child.attributes.get(new_start, child.attributes[origin_start]) - child.attributes[origin_start]
                new_end_cost += child.attributes.get(new_end, child.attributes[origin_end]) - child.attributes[origin_end]
            self.attributes[new_start] = new_start_cost
            self.attributes[new_end] = new_end_cost
        
        if mode == 'apply':
            for child in self.children:
                if new_end in child.attributes:
                    child.attributes[origin_end] = child.attributes[new_end]
                    child.attributes[origin_start] = child.attributes[new_start]

        if self.parent:
            self.parent.update_cost_2(cost, cost_type = cost_type, mode = mode)



    

class Parameters():
    def __init__(self, condition_max_num, tables_id, columns_id, physic_ops_id, column_total_num,
                 table_total_num, physic_op_total_num, condition_op_dim,  condition_word_dim, compare_ops_id, bool_ops_id,
                 bool_ops_total_num, compare_ops_total_num, schema, min_max_info, classes, word2vec_model_path,operator_model_types, table_row_info, index_max_col_num=2):
        self.condition_max_num = condition_max_num
        self.tables_id = tables_id
        self.columns_id = columns_id
        self.physic_ops_id = physic_ops_id
        self.column_total_num = column_total_num
        self.table_total_num = table_total_num
        self.physic_op_total_num = physic_op_total_num
        self.condition_op_dim = condition_op_dim
        self.condition_word_dim = condition_word_dim
        self.compare_ops_id = compare_ops_id
        self.bool_ops_id = bool_ops_id
        self.bool_ops_total_num = bool_ops_total_num
        self.compare_ops_total_num = compare_ops_total_num
        self.schema = schema
        self.min_max_info = min_max_info
        self.index_max_col_num = index_max_col_num
        self.classes = classes
        self.num_classes = len(classes)
        self.word2vec_model_path = word2vec_model_path,
        self.word2vec_model = Word2Vec.load(word2vec_model_path)
        self.operator_model_types = operator_model_types
        self.table_row_info = table_row_info


    def to_dict(self):
        return {
            'condition_max_num': self.condition_max_num,
            'tables_id': self.tables_id,
            'columns_id': self.columns_id,
            'physic_ops_id': self.physic_ops_id,
            'column_total_num': self.column_total_num,
            'table_total_num': self.table_total_num,
            'physic_op_total_num': self.physic_op_total_num,
            'condition_op_dim': self.condition_op_dim,
            'condition_word_dim': self.condition_word_dim,
            'compare_ops_id': self.compare_ops_id,
            'bool_ops_id': self.bool_ops_id,
            'bool_ops_total_num': self.bool_ops_total_num,
            'compare_ops_total_num': self.compare_ops_total_num,
            'schema': self.schema,
            'min_max_info': self.min_max_info,
            'index_max_col_num': self.index_max_col_num,
            'classes': self.classes,
            'word2vec_model_path': self.word2vec_model_path,
            'operator_model_types': self.operator_model_types,
            'table_row_info': self.table_row_info

        }

    def store(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            condition_max_num=data['condition_max_num'],
            tables_id=data['tables_id'],
            columns_id=data['columns_id'],
            physic_ops_id=data['physic_ops_id'],
            column_total_num=data['column_total_num'],
            table_total_num=data['table_total_num'],
            physic_op_total_num=data['physic_op_total_num'],
            condition_op_dim=data['condition_op_dim'],
            condition_word_dim=data['condition_word_dim'],
            compare_ops_id=data['compare_ops_id'],
            bool_ops_id=data['bool_ops_id'],
            bool_ops_total_num=data['bool_ops_total_num'],
            compare_ops_total_num=data['compare_ops_total_num'],
            schema=data['schema'],
            min_max_info=data['min_max_info'],
            classes = data['classes'],
            index_max_col_num=data['index_max_col_num'],
            word2vec_model_path = data['word2vec_model_path'],
            operator_model_types = data['operator_model_types'],
            table_row_info = data['table_row_info']
            
        )


        


            


def build_query_plan_tree(plan, parent = None):
    # Create the root node
    node_type = plan.get("Node Type")
    # parent_relationship = plan.get("Parent Relationship")
    attributes = {k: v for k, v in plan.items() if k not in ["Node Type", "Plans"]}

    root_node = QueryPlanNode(node_type, parent, **attributes)

    # Recursively process the "Plans" key if it exists
    if "Plans" in plan:
        for child_plan in plan["Plans"]:
            child_node = build_query_plan_tree(child_plan, root_node)
            root_node.add_child(child_node)

    return root_node


def compare_nodes(original_node, what_if_node, ind_node):
    '''
    Compare if the cost of original node - what-if node is consistent with the actual total time of original node - index node.
    '''
    tolerance = 0.05
    if not original_node or not what_if_node or not ind_node:
        return 'Missing node data',0

    original_total_cost = original_node.cal_cost('Total Cost')
    what_if_total_cost = what_if_node.cal_cost('Total Cost')
    original_actual_time = original_node.cal_cost('Actual Total Time')
    ind_actual_time = ind_node.cal_cost('Actual Total Time')

    if original_total_cost is None or what_if_total_cost is None or original_actual_time is None or ind_actual_time is None:
        return 'Incomplete node data',0

    cost_comparison = (original_total_cost - what_if_total_cost) >= tolerance * original_total_cost
    time_comparison = (original_actual_time - ind_actual_time) >= tolerance * original_actual_time

    if cost_comparison == time_comparison:
        return 'Accurate',1
    else:
        multi = 1
        # Calculate the multiple for what-if node's cost estimation to make it consistent with the actual total time of original node - index node
        if cost_comparison == True: # what-if cost is underestimated
            for multiplier in [1,2,3,4,5,6,7,8,9,10]:
                k = (ind_actual_time - original_actual_time) / original_actual_time
                if (what_if_total_cost * multiplier - original_total_cost) >= k * original_total_cost:
                    break
        else: # what-if cost is over-estimated
            for multiplier in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:
                k = (original_actual_time - ind_actual_time) / ind_actual_time
                if (original_total_cost - what_if_total_cost * multiplier) >= k * original_total_cost:
                    break

        return 'Inaccurate',multiplier
    

def compare_nodes_2(original_node, what_if_node, ind_node):
    '''
    Compare if the improvement of what-if is consistent with the improvement of actual index.
    '''
    tolerance = 0.08
    if not original_node or not what_if_node or not ind_node:
        return 'Missing node data',0

    original_total_cost = original_node.cal_cost('Total Cost')
    what_if_total_cost = what_if_node.cal_cost('Total Cost')
    original_actual_time = original_node.cal_cost('Actual Total Time')
    ind_actual_time = ind_node.cal_cost('Actual Total Time')

    if original_total_cost is None or what_if_total_cost is None or original_actual_time is None or ind_actual_time is None:
        return 'Incomplete node data',0
    
    cost_improvement = (original_total_cost - what_if_total_cost) / original_total_cost
    time_improvement = (original_actual_time - ind_actual_time) / original_actual_time

    if abs(cost_improvement - time_improvement) <= tolerance:
        return 'Accurate',1
    else:
        multi = (original_total_cost - time_improvement * original_total_cost) / what_if_total_cost
        return 'Inaccurate',multi


def find_inaccurate_node(original_tree, what_if_tree, ind_tree):
    """
    Detect the node in query plan that leads to inaccurate what-if estimation.
    """
    original_nodes = original_tree.children
    what_if_nodes = what_if_tree.children
    ind_nodes = ind_tree.children
    
    for i, (orig_node, what_if_node, ind_node) in enumerate(zip(original_nodes, what_if_nodes, ind_nodes)):
        result, multi = compare_nodes(orig_node, what_if_node, ind_node)
        if result != 'Accurate':
            deeper_result = find_inaccurate_node(orig_node, what_if_node, ind_node)[0]
            return deeper_result, multi
        else:
            for j, (orig_node_2, what_if_node_2, ind_node_2) in enumerate(zip(original_nodes, what_if_nodes, ind_nodes)):
                if j != i:
                    result, multi = compare_nodes(orig_node, what_if_node_2, ind_node_2)
                    if result != 'Accurate':
                        deeper_result = find_inaccurate_node(orig_node, what_if_node_2, ind_node_2)[0]
                        return deeper_result, multi
    
    return (original_tree, what_if_tree, ind_tree), 1

def compare_trees(what_if_tree, ind_tree):
    what_if_nodes = dfs_traverse(what_if_tree)
    ind_nodes = dfs_traverse(ind_tree)
    for what_if_node, ind_node in zip(what_if_nodes, ind_nodes):
        if what_if_node.node_type != ind_node.node_type:
            return False, what_if_node, ind_node
    return True, None, None

def find_inaccurate_node_2(original_tree, what_if_tree, ind_tree, ind_table, ind_cols):
    """
    Detect the node related with candidate index in query plan that leads to inaccurate what-if estimation.
    Compute the multiplier on the node cost to refine the what-if estimation error.
    """
    # Detect the node related with index scan in what_if_tree & ind_tree
    what_if_nodes = dfs_traverse(what_if_tree)
    ind_nodes = dfs_traverse(ind_tree)
    for what_if_node in what_if_nodes:
        if 'Index Scan' in what_if_node.node_type and 'Index Name' in what_if_node.attributes:
            what_if_index_name = what_if_node.attributes['Index Name']
            if 'btree' in what_if_index_name:
                break
    for ind_node in ind_nodes:
        if 'Index Scan' in ind_node.node_type and 'Index Name' in ind_node.attributes:
            ind_index_name = ind_node.attributes['Index Name']
            if ind_table in ind_index_name:
                flg = True
                for ind_col in ind_cols:
                    if ind_col not in ind_index_name:
                        flg = False
                if flg:
                    break
    
    # Explore the multiplier that makes (what-if - origin) consistent with (ind - origin)
    result, multi = compare_nodes(original_tree, what_if_tree, ind_tree)
    candidate_multis = []

    if multi >= 1:
        candidate_multis = [1,2,3,4,5,6,7,8,9,10]
    else:
        candidate_multis = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    for multiplier in candidate_multis:
        what_if_node.attributes['Total Cost New'] = what_if_node.cal_cost() * multiplier
        if what_if_node.parent:
            what_if_node.parent.update_cost('Total Cost')
        origin_cost = what_if_tree.cal_cost()
        what_if_tree.attributes['Total Cost'] = what_if_tree.attributes['Total Cost New']
        result, multi = compare_nodes(original_tree, what_if_tree, ind_tree)
        if result == 'Accurate':
            multi = multiplier
            break
        what_if_tree.attributes['Total Cost'] = origin_cost
    
    return (None, what_if_node, ind_node), multi

    
def find_inaccurate_node_3(original_tree, what_if_tree, ind_tree, ind_table, ind_cols):
    """
    Detect the node related with candidate index in query plan that leads to inaccurate what-if estimation.
    Inaccurate estimation: improvement calculated by what-if differs from the actual improvement.
    Compute the multiplier on the node cost to refine the what-if estimation error.
    """
     # Detect the node related with index scan in what_if_tree & ind_tree
    what_if_nodes = dfs_traverse(what_if_tree)
    ind_nodes = dfs_traverse(ind_tree)
    for what_if_node in what_if_nodes:
        if what_if_node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in what_if_node.attributes:
            what_if_index_name = what_if_node.attributes['Index Name']
            if 'btree' in what_if_index_name:
                break
    for ind_node in ind_nodes:
        if ind_node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in ind_node.attributes:
            ind_index_name = ind_node.attributes['Index Name']
            if ind_table in ind_index_name:
                flg = True
                for ind_col in ind_cols:
                    if ind_col not in ind_index_name:
                        flg = False
                if flg:
                    break
    
    # Explore the multiplier that makes (what-if - origin) consistent with (ind - origin)
    result, multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
    candidate_multis = []

    if multi >= 1:
        candidate_multis = [1,2,3,4,5,6,7,8,9,10,20,30,40,50]
    else:
        candidate_multis = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.08,0.06,0.04,0.02,0.01]
    last_multiplier = 1
    for multiplier in candidate_multis:
        what_if_node.attributes['Total Cost New'] = what_if_node.cal_cost() * multiplier
        if what_if_node.parent:
            what_if_node.parent.update_cost('Total Cost')
        origin_cost = what_if_tree.cal_cost()
        what_if_tree.attributes['Total Cost'] = what_if_tree.attributes['Total Cost New']
        result, new_multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
        if result == 'Accurate':
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = multiplier
            break
        if (multi >= 1 and new_multi < 1) or (multi < 1 and new_multi >= 1):
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = last_multiplier
            break
        what_if_tree.attributes['Total Cost'] = origin_cost
        last_multiplier = multiplier
    multi = multiplier 
    what_if_node.attributes['Total Cost New'] = what_if_node.cal_cost() * multi
    if what_if_node.parent:
        what_if_node.parent.update_cost('Total Cost')
    return (None, what_if_node, ind_node), multi, what_if_tree.attributes['Total Cost New']


    
def find_inaccurate_node_4(original_tree, what_if_tree, ind_tree, ind_table, ind_cols):
     # Detect the node related with index scan in what_if_tree & ind_tree
    what_if_nodes = dfs_traverse(what_if_tree)
    ind_nodes = dfs_traverse(ind_tree)
    find_related_what_if_node = False
    for what_if_node in what_if_nodes:
        if what_if_node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in what_if_node.attributes:
            what_if_index_name = what_if_node.attributes['Index Name']
            if 'btree' in what_if_index_name:
                find_related_what_if_node = True
                break
    
    if not find_related_what_if_node:
        return None, 1, what_if_tree.attributes['Total Cost']
    
    for ind_node in ind_nodes:
        if ind_node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in ind_node.attributes:
            ind_index_name = ind_node.attributes['Index Name']
            if ind_table in ind_index_name:
                flg = True
                for ind_col in ind_cols:
                    if ind_col not in ind_index_name:
                        flg = False
                if flg:
                    break
    
    # Explore the multiplier that makes (what-if - origin) consistent with (ind - origin)
    result, multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
    candidate_multis = []

    if multi >= 1:
        candidate_multis = [1,2,3,4,5,6,7,8,9,10,20,30,40,50]
    else:
        candidate_multis = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.08,0.06,0.04,0.02,0.01]
    last_multiplier = 1
    found_flag = False
    for multiplier in candidate_multis:
        origin_cost = what_if_tree.cal_cost()
        new_cost = (what_if_node.attributes['Total Cost'] - what_if_node.attributes['Startup Cost']) * multiplier 
        what_if_node.update_cost_2(new_cost)
        what_if_tree.attributes['Total Cost'] = what_if_tree.attributes['Total Cost New']
        result, new_multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
        if result == 'Accurate':
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = multiplier
            found_flag = True
            break
        if (multi >= 1 and new_multi < 1) or (multi < 1 and new_multi >= 1):
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = last_multiplier
            found_flag = True
            break
        what_if_tree.attributes['Total Cost'] = origin_cost
        last_multiplier = multiplier
    if not found_flag:
        multi = multiplier 
    new_cost = (what_if_node.attributes['Total Cost'] - what_if_node.attributes['Startup Cost']) * multi
    what_if_node.update_cost_2(new_cost)
    return (None, what_if_node, ind_node), multi, what_if_tree.attributes['Total Cost New']   

    
def find_inaccurate_node_8(original_tree, what_if_tree, ind_tree, ind_table, ind_cols, parameters):
     # Detect the node related with index scan in what_if_tree & ind_tree
    what_if_nodes = dfs_traverse(what_if_tree)
    ind_nodes = dfs_traverse(ind_tree)
    find_related_what_if_node = False
    for what_if_node in what_if_nodes:
        if what_if_node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in what_if_node.attributes:
            what_if_index_name = what_if_node.attributes['Index Name']
            if 'btree' in what_if_index_name:
                find_related_what_if_node = True
                break
    
    if not find_related_what_if_node:
        for what_if_node in what_if_nodes:
            if what_if_node.node_type == 'Seq Scan' and 'Relation Name' in what_if_node.attributes:
                what_if_table_name = what_if_node.attributes['Relation Name']
                if what_if_table_name == ind_table:
                    find_related_what_if_node = True
                    break
                
    if not find_related_what_if_node:
        return None, 1, what_if_tree.attributes['Total Cost']
    
    for ind_node in ind_nodes:
        if ind_node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in ind_node.attributes:
            ind_index_name = ind_node.attributes['Index Name']
            if ind_table in ind_index_name:
                flg = True
                for ind_col in ind_cols:
                    if ind_col not in ind_index_name:
                        flg = False
                if flg:
                    break
    
    # Explore the multiplier that makes (what-if - origin) consistent with (ind - origin)
    result, multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
    
    # Get candidate_multis from parameters.classes
    if multi >= 1:
        candidate_multis = [c for c in parameters.classes if c >= 1]
        candidate_multis = sorted(candidate_multis)  # Sort from small to large
    else:
        candidate_multis = [c for c in parameters.classes if c < 1]
        candidate_multis = sorted(candidate_multis, reverse=True)  # Sort from large to small
    last_multiplier = 1
    found_flag = False
    for multiplier in candidate_multis:
        origin_cost = what_if_tree.cal_cost()
        new_cost = (what_if_node.attributes['Total Cost'] - what_if_node.attributes['Startup Cost']) * multiplier 
        what_if_node.update_cost_2(new_cost)
        what_if_tree.attributes['Total Cost'] = what_if_tree.attributes['Total Cost New']
        result, new_multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
        if result == 'Accurate':
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = multiplier
            found_flag = True
            break
        if (multi >= 1 and new_multi < 1) or (multi < 1 and new_multi >= 1):
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = last_multiplier
            found_flag = True
            break
        what_if_tree.attributes['Total Cost'] = origin_cost
        last_multiplier = multiplier
    if not found_flag:
        multi = multiplier 
    new_cost = (what_if_node.attributes['Total Cost'] - what_if_node.attributes['Startup Cost']) * multi
    what_if_node.update_cost_2(new_cost)
    return (None, what_if_node, ind_node), multi, what_if_tree.attributes['Total Cost New']   


def gen_CAMs():
     # Fine-grained region around 1.0 (from 10^-1.3 to 10^1.3)
    fine_exponents = np.arange(-1.5, 1.5, 0.05)  # step=0.1
    fine_multipliers = [round(10 ** exp, 6) for exp in fine_exponents]

    # Coarse-grained extremes
    # coarse_exponents = [-4, -3, -2, 2, 2.5, 3]
    coarse_exponents = np.arange(-4,-2,0.5)
    coarse_exponents = np.concatenate((coarse_exponents, np.arange(2,4,0.5)))
    coarse_multipliers = [round(10 ** exp, 6) for exp in coarse_exponents]

    combined = sorted(set(fine_multipliers + coarse_multipliers))

    print(combined)

    return list(combined)

    # print(f"fine:{fine_multipliers}")
    # print(f"coarse:{coarse_multipliers}")

def find_inaccurate_node_5(original_tree, what_if_tree, ind_tree, ind_table, ind_cols):
     # Detect the node related with index scan in what_if_tree & ind_tree
    what_if_nodes = dfs_traverse(what_if_tree)
    ind_nodes = dfs_traverse(ind_tree)
    find_related_what_if_node = False
    for what_if_node in what_if_nodes:
        if what_if_node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in what_if_node.attributes:
            what_if_index_name = what_if_node.attributes['Index Name']
            if 'btree' in what_if_index_name:
                find_related_what_if_node = True
                break
    
    if not find_related_what_if_node:
        return None, 1, what_if_tree.attributes['Total Cost']
    
    for ind_node in ind_nodes:
        if ind_node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in ind_node.attributes:
            ind_index_name = ind_node.attributes['Index Name']
            if ind_table in ind_index_name:
                flg = True
                for ind_col in ind_cols:
                    if ind_col not in ind_index_name:
                        flg = False
                if flg:
                    break
    
    # Explore the multiplier that makes (what-if - origin) consistent with (ind - origin)
    result, multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
    # candidate_multis = gen_CAMs()

    if multi >= 1:
        candidate_multis = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
    else:
        candidate_multis = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
    # if multi >= 1:
    #     candidate_multis = [candidate for candidate in candidate_multis if candidate >= 1]
    # else:
    #     candidate_multis = [candidate for candidate in candidate_multis if candidate < 1]
    #     candidate_multis = sorted(candidate_multis, reverse=True)

    last_multiplier = 1
    found_flag = False
    for multiplier in candidate_multis:
        origin_cost = what_if_tree.cal_cost()
        new_cost = (what_if_node.attributes['Total Cost'] - what_if_node.attributes['Startup Cost']) * multiplier 
        what_if_node.update_cost_2(new_cost)
        what_if_tree.attributes['Total Cost'] = what_if_tree.attributes['Total Cost New']
        result, new_multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
        if result == 'Accurate':
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = multiplier
            found_flag = True
            break
        if (multi >= 1 and new_multi < 1) or (multi < 1 and new_multi >= 1):
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = last_multiplier
            found_flag = True
            break
        what_if_tree.attributes['Total Cost'] = origin_cost
        last_multiplier = multiplier
    if not found_flag:
        multi = multiplier 
    new_cost = (what_if_node.attributes['Total Cost'] - what_if_node.attributes['Startup Cost']) * multi
    what_if_node.update_cost_2(new_cost)
    return (None, what_if_node, ind_node), multi, what_if_tree.attributes['Total Cost New']

def find_inaccurate_node_6(original_tree, what_if_tree, ind_tree, ind_table, ind_cols):
     # Detect error node
    leaf_nodes = dfs_traverse_leaf(what_if_tree)
    max_sens = 0
    for leaf_node in leaf_nodes:
        multiplier = 2
        new_cost = (leaf_node.attributes['Total Cost'] - leaf_node.attributes['Startup Cost']) * multiplier 
        leaf_node.update_cost_2(new_cost)
        sensitivity = (what_if_tree.attributes['Total Cost New'] - what_if_tree.attributes['Total Cost']) / (leaf_node.attributes['Total Cost New'] - leaf_node.attributes['Total Cost']) 
        if sensitivity >= max_sens:
            max_sens = sensitivity
            error_node = leaf_node
        # clear effect
        current_node = leaf_node
        while current_node:
            current_node.attributes['Total Cost New'] = current_node.attributes['Total Cost']
            current_node.attributes['Startup Cost New'] = current_node.attributes['Startup Cost']
            current_node = current_node.parent
    
 
    # calculate multi
    result, multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
    candidate_multis = []

    if multi >= 1:
        candidate_multis = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
    else:
        candidate_multis = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
    last_multiplier = 1
    found_flag = False
    for multiplier in candidate_multis:
        origin_cost = what_if_tree.cal_cost()
        new_cost = (error_node.attributes['Total Cost'] - error_node.attributes['Startup Cost']) * multiplier 
        error_node.update_cost_2(new_cost)
        what_if_tree.attributes['Total Cost'] = what_if_tree.attributes['Total Cost New']
        result, new_multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
        if result == 'Accurate':
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = multiplier
            found_flag = True
            break
        if (multi >= 1 and new_multi < 1) or (multi < 1 and new_multi >= 1):
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = last_multiplier
            found_flag = True
            break
        what_if_tree.attributes['Total Cost'] = origin_cost
        last_multiplier = multiplier
    if not found_flag:
        multi = multiplier 
    new_cost = (error_node.attributes['Total Cost'] - error_node.attributes['Startup Cost']) * multi
    error_node.update_cost_2(new_cost)
    return error_node, multi, what_if_tree.attributes['Total Cost New']
    


def find_inaccurate_node_7(original_tree, what_if_tree, ind_tree, ind_table, ind_cols):
     # Detect the node related with index scan in what_if_tree & ind_tree
     # sensitivity analysis
    what_if_nodes = dfs_traverse(what_if_tree)
    ind_nodes = dfs_traverse(ind_tree)
    find_related_what_if_node = False
    related_nodes = []
    for what_if_node in what_if_nodes:
        if what_if_node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in what_if_node.attributes:
            what_if_index_name = what_if_node.attributes['Index Name']
            if 'btree' in what_if_index_name:
                find_related_what_if_node = True
                related_nodes.append(what_if_node)
                # break
    
    if not find_related_what_if_node:
        return None, 1, what_if_tree.attributes['Total Cost']
    
    if len(related_nodes) > 1:
        for node in related_nodes:
            multiplier = 2
            new_cost = (node.attributes['Total Cost'] - node.attributes['Startup Cost']) * multiplier 
            node.update_cost_2(new_cost)
            sensitivity = (what_if_tree.attributes['Total Cost New'] - what_if_tree.attributes['Total Cost']) / (leaf_node.attributes['Total Cost New'] - leaf_node.attributes['Total Cost']) 
            if sensitivity >= max_sens:
                max_sens = sensitivity
                what_if_node = node
            # clear effect
            current_node = node
            while current_node:
                current_node.attributes['Total Cost New'] = current_node.attributes['Total Cost']
                current_node.attributes['Startup Cost New'] = current_node.attributes['Startup Cost']
                current_node = current_node.parent
            
    for ind_node in ind_nodes:
        if ind_node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in ind_node.attributes:
            ind_index_name = ind_node.attributes['Index Name']
            if ind_table in ind_index_name:
                flg = True
                for ind_col in ind_cols:
                    if ind_col not in ind_index_name:
                        flg = False
                if flg:
                    break
    
    # Explore the multiplier that makes (what-if - origin) consistent with (ind - origin)
    result, multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
    candidate_multis = []

    if multi >= 1:
        candidate_multis = [1,2,3,4,5,6,7,8,9,10,20,30,40,50]
    else:
        candidate_multis = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.08,0.06,0.04,0.02,0.01]
    last_multiplier = 1
    found_flag = False
    for multiplier in candidate_multis:
        origin_cost = what_if_tree.cal_cost()
        new_cost = (what_if_node.attributes['Total Cost'] - what_if_node.attributes['Startup Cost']) * multiplier 
        what_if_node.update_cost_2(new_cost)
        what_if_tree.attributes['Total Cost'] = what_if_tree.attributes['Total Cost New']
        result, new_multi = compare_nodes_2(original_tree, what_if_tree, ind_tree)
        if result == 'Accurate':
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = multiplier
            found_flag = True
            break
        if (multi >= 1 and new_multi < 1) or (multi < 1 and new_multi >= 1):
            what_if_tree.attributes['Total Cost'] = origin_cost
            multi = last_multiplier
            found_flag = True
            break
        what_if_tree.attributes['Total Cost'] = origin_cost
        last_multiplier = multiplier
    if not found_flag:
        multi = multiplier 
    new_cost = (what_if_node.attributes['Total Cost'] - what_if_node.attributes['Startup Cost']) * multi
    what_if_node.update_cost_2(new_cost)
    return (None, what_if_node, ind_node), multi, what_if_tree.attributes['Total Cost New']  



def change_alias2table(column, alias2table):
    relation_name = column.split('.')[0] if '.' in column else column
    column_name = column.split('.')[1] if '.' in column else column
    if relation_name in alias2table:
        return alias2table[relation_name] + '.' + column_name
    else:
        return column

def get_alias2table(root, alias2table):
    if 'Relation Name' in root.attributes and 'Alias' in root.attributes:
        alias2table[root.attributes['Alias']] = root.attributes['Relation Name']
    for child in root.children:
            get_alias2table(child, alias2table)

def extract_relation_from_index_name(index_name,parameters):
    current_relation = None 
    for relation_name in parameters.tables_id:
        if relation_name in index_name:
            if current_relation is None:
                current_relation = relation_name
            else:
                if len(relation_name) > len(current_relation):
                    current_relation = relation_name
    return current_relation
            

    

def extract_info_from_node(node, alias2table, parameters):
    relation_name, index_name = None, None
    if 'Relation Name' in node.attributes:
        relation_name = node.attributes['Relation Name']
    if 'Index Name' in node.attributes:
        index_name = node.attributes['Index Name']
    if relation_name is None and index_name is not None:
        relation_name = extract_relation_from_index_name(index_name, parameters)
    if node.node_type == 'Materialize':
        return Materialize(), None
    elif node.node_type == 'Hash':
        return Hash(), None
    elif node.node_type == 'Sort':
        keys = [change_alias2table(key, alias2table) for key in node.attributes['Sort Key']]
        return Sort(keys), None
    elif node.node_type == 'BitmapAnd':
        return BitmapCombine('BitmapAnd'), None
    elif node.node_type == 'BitmapOr':
        return BitmapCombine('BitmapOr'), None
    elif node.node_type == 'Result':
        return Result(), None
    elif node.node_type == 'Hash Join':
        return Join('Hash Join', pre2seq(node.attributes["Hash Cond"], alias2table, relation_name, index_name)), None
    elif node.node_type == 'Merge Join':
        return Join('Merge Join', pre2seq(node.attributes["Merge Cond"], alias2table, relation_name, index_name)), None
    elif node.node_type == 'Nested Loop':
        if 'Join Filter' in node.attributes:
            condition = pre2seq(node.attributes['Join Filter'], alias2table, relation_name, index_name)
        else:
            condition = []
        return Join('Nested Loop', condition), None
    elif node.node_type == 'Aggregate':
        if 'Group Key' in node.attributes:
            keys = [change_alias2table(key, alias2table) for key in node.attributes['Group Key']]
        else:
            keys = []
        return Aggregate(node.attributes['Strategy'], keys), None
    elif node.node_type == 'Seq Scan':
        if 'Filter' in node.attributes:
            condition_seq_filter = pre2seq(node.attributes['Filter'], alias2table, relation_name, index_name)
        else:
            condition_seq_filter = []
        condition_seq_index, relation_name, index_name, plan_rows = [], node.attributes["Relation Name"], None, node.attributes.get('Plan Rows', 1)
        return Scan('Seq Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, plan_rows), None
    elif node.node_type == 'CTE Scan':
        if 'Filter' in node.attributes:
            condition_seq_filter = pre2seq(node.attributes['Filter'], alias2table, relation_name, index_name)
        else:
            condition_seq_filter = []
        condition_seq_index, relation_name, index_name, plan_rows = [], node.attributes["CTE Name"], None, node.attributes.get('Plan Rows', 1)
        return Scan('CTE Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, plan_rows), None
    elif node.node_type == 'Bitmap Heap Scan':
        if 'Filter' in node.attributes:
            condition_seq_filter = pre2seq(node.attributes['Filter'], alias2table, relation_name, index_name)
        else:
            condition_seq_filter = []
        condition_seq_index, relation_name, index_name, plan_rows = [], node.attributes["Relation Name"], None, node.attributes.get('Plan Rows', 1)
        return Scan('Bitmap Heap Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, plan_rows), None
    elif node.node_type == 'Index Scan':
        if 'Filter' in node.attributes:
            condition_seq_filter = pre2seq(node.attributes['Filter'], alias2table, relation_name, index_name)
        else:
            condition_seq_filter = []
        if 'Index Cond' in node.attributes:
            condition_seq_index = pre2seq(node.attributes['Index Cond'], alias2table, relation_name, index_name)
        else:
            condition_seq_index = []
        relation_name, index_name, plan_rows = node.attributes["Relation Name"], node.attributes['Index Name'], node.attributes.get('Plan Rows', 1)
        if len(condition_seq_index) == 1 and re.match(r'[a-zA-Z]+', condition_seq_index[0].right_value) is not None:
            return Scan('Index Scan', condition_seq_filter, condition_seq_index, relation_name,
                        index_name, plan_rows), condition_seq_index
        else:
            return Scan('Index Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, plan_rows), None
    elif node.node_type == 'Bitmap Index Scan':
        if 'Index Cond' in node.attributes:
            condition_seq_index = pre2seq(node.attributes['Index Cond'], alias2table, relation_name, index_name)
        else:
            condition_seq_index = []
        condition_seq_filter, relation_name, index_name, plan_rows = [], None, node.attributes['Index Name'], node.attributes.get('Plan Rows', 1)
        if len(condition_seq_index) == 1 and re.match(r'[a-zA-Z]+', condition_seq_index[0].right_value) is not None:
            return Scan('Bitmap Index Scan', condition_seq_filter, condition_seq_index, relation_name,
                        index_name, plan_rows), condition_seq_index
        else:
            return Scan('Bitmap Index Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, plan_rows), None
    elif node.node_type == 'Index Only Scan':
        if 'Index Cond' in node.attributes:
            condition_seq_index = pre2seq(node.attributes['Index Cond'], alias2table, relation_name, index_name)
        else:
            condition_seq_index = []
        condition_seq_filter, relation_name, index_name, plan_rows = [], None, node.attributes['Index Name'], node.attributes.get('Plan Rows', 1)
        if len(condition_seq_index) == 1 and re.match(r'[a-zA-Z]+', condition_seq_index[0].right_value) is not None:
            return Scan('Index Only Scan', condition_seq_filter, condition_seq_index, relation_name,
                        index_name, plan_rows), condition_seq_index
        else:
            return Scan('Index Only Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, plan_rows), None
    elif node.node_type == 'Gather Merge' or node.node_type == 'Gather':
        return Gather(), None
    elif node.node_type == 'Limit':
        return Limit(), None
    else:
        return Default(node.node_type), None
        


def plan2seq(root, alias2table, parameters):
    sequence = []
    join_conditions = []
    node, join_condition = extract_info_from_node(root, alias2table, parameters)
    if join_condition is not None:
        join_conditions += join_condition
    sequence.append(node)
    for child in root.children:
            next_sequence, next_join_conditions = plan2seq(child, alias2table, parameters)
            sequence += next_sequence
            join_conditions += next_join_conditions
    sequence.append(None)
    return sequence, join_conditions


def get_leaf_nodes(root):
    leaf_nodes = []
    for child in root.children:
        if len(child.children) == 0:
            leaf_nodes.append(child)
        else:
            leaf_nodes += get_leaf_nodes(child)
    return leaf_nodes

def plan2seq_leaf(root, alias2table, parameters):
    sequence = []
    join_conditions = []
    leaf_nodes = get_leaf_nodes(root)
    for leaf in leaf_nodes:
        node, join_condition = extract_info_from_node(leaf, alias2table, parameters)
        if join_condition is not None:
            join_conditions += join_condition
        sequence.append(node)
    return sequence, join_conditions

   

def feature_extractor(root, parameters):
    # if root.node_type == 'Limit':
    #     root = root.children[0]
    alias2table = {}
    get_alias2table(root, alias2table)
    seq, _ = plan2seq(root, alias2table, parameters)
    nodes = [node for node in seq if node is not None]
    return nodes

def leaf_extractor(root, parameters):
    alias2table = {}
    get_alias2table(root, alias2table)
    seq, _ = plan2seq_leaf(root, alias2table, parameters)
    leafs = [node for node in seq if node is not None]
    return leafs

def get_representation(value, word2vec_model):
    if value in word2vec_model.wv.key_to_index:
        vector = word2vec_model.wv[value]
        return vector
    candidates = []
    prefixes = [value[:i] for i in range(len(value), 0, -1)]
    candidates += prefixes
    for candidate in candidates:
        if candidate in word2vec_model.wv.key_to_index:
            return word2vec_model.wv[candidate]
    return np.zeros(500)


def get_representation_2(value, word2vec_model):
    if value in word2vec_model.wv.key_to_index:
        vector = word2vec_model.wv[value]
        return vector
    candidates = []
    prefixes = [value[:i] for i in range(len(value), 0, -1)]
    candidates += prefixes
    for candidate in candidates:
        if candidate in word2vec_model.wv.key_to_index:
            return word2vec_model.wv[candidate]
        for key in word2vec_model.wv.key_to_index:
            if key.startswith(candidate):
                return word2vec_model.wv[key]
    return np.zeros(500)

    # vector = word2vec_model.wv[value]
    # return vector
    # string embedding
    # if value in word_vectors:
    #     embedded_result = np.array(list(word_vectors[value]))
    # else:
    #     embedded_result = np.array([0.0 for _ in range(500)])
    # hash bitmap
    # hash_result = np.array([0.0 for _ in range(500)])
    # for t in value:
    #     hash_result[hash(t) % 500] = 1.0
    # return hash_result
    # return np.concatenate((embedded_result, hash_result), 0)


def get_str_representation(value, word2vec_model):
    vec = np.array([])
    count = 0
    # prefix = determine_prefix(column)
    prefix = ''
    for v in value.split('%'):
        if len(v) > 0:
            if len(vec) == 0:
                vec = get_representation(prefix + v, word2vec_model)
                count = 1
            else:
                new_vec = get_representation(prefix + v, word2vec_model)
                vec = vec + new_vec
                count += 1
    if count > 0:
        vec = vec / float(count)
    return vec

def get_str_representation_2(value, word2vec_model, selectivity):
    vec = np.array([])
    count = 0
    # prefix = determine_prefix(column)
    prefix = ''
    for v in value.split('%'):
        if len(v) > 0:
            if len(vec) == 0:
                vec = get_representation_2(prefix + v, word2vec_model)
                count = 1
            else:
                new_vec = get_representation_2(prefix + v, word2vec_model)
                vec = vec + new_vec
                count += 1
    if count > 0:
        vec = vec / float(count)
    
    stas_vec = np.array([selectivity])
    combined_vec = np.concatenate((vec, stas_vec), axis=0)
    return combined_vec




def train_word2vec_2(record_path, model_path, embedding_dim=500, window=5, min_count=1, workers=4, iter=10):
    """
    Train word2vec model with records.
    """
    records = {}
    with open(record_path, 'r') as file:
        records = json.load(file)
        # for key, value in data.items():
        #     records[key] = value
    sequences = []
    for query_index, details in records.items():
        what_if_tree = build_query_plan_tree(details['what-if_plan'])
        nodes = dfs_traverse(what_if_tree)
        for node in nodes:
            if node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan']:
                if 'Filter' in node.attributes:
                    filter = node.attributes['Filter']
                    tokens = word_tokenize(filter)
                    filtered_tokens = [token.replace("'", "") for token in tokens]
                    sequences.append(filtered_tokens)
     # Train the Word2Vec model
    model = Word2Vec(sentences=sequences, vector_size=embedding_dim, window=window, min_count=min_count, workers=workers)
    # test
    words = list(model.wv.key_to_index)
    print(words)
    # print(model.wv["1996-01-01"])
    # Save the trained model
    model.save(model_path)
    print(f"Word2Vec model saved to {model_path}")
    
                   

def encode_condition_op(condition_op, relation_name, index_name, parameters, plan_rows=1):
    """
        Encode a single condition operation into a vector representation.
        
        Args:
            condition_op (dict): The condition operation to encode.
            relation_name (str): The name of the relation (table).
            index_name (str): The name of the index.
            parameters (Parameters): The parameters for encoding.
        
        Returns:
            np.array: The encoded vector for the condition operation.
    """
    # bool_operator + left_value + compare_operator + right_value
    if condition_op is None:
        vec = [0 for _ in range(parameters.condition_op_dim)]
    elif condition_op.op_type == 'Bool':
        idx = parameters.bool_ops_id[condition_op.operator]
        vec = [0 for _ in range(parameters.bool_ops_total_num)]
        vec[idx - 1] = 1
    elif condition_op.op_type == 'Failure':
        idx = parameters.bool_ops_total_num + 1
        vec = [0 for _ in range(parameters.condition_op_dim)]
        vec[idx - 1] = 1
    else:
        operator = condition_op.operator
        left_value = condition_op.left_value
        # look for relation

        if relation_name is None:
            if '.' in left_value and left_value.split('.')[0] in parameters.schema:
                relation_name = left_value.split('.')[0]
            else:
                relation_name = extract_relation_from_index_name(index_name, tables_id=  parameters.tables_id)
                left_value = relation_name + '.' + left_value
        
        table_row_num = parameters.table_row_info.get(relation_name, plan_rows)
        selectivity = float(plan_rows) / table_row_num
        # if relation_name is None:
        #     for table in parameters.schema:
        #         if (table in left_value) or (index_name and table in index_name):
        #             if relation_name is not None:
        #                if relation_name in table:
        #                    relation_name = table
        #             else:
        #                 relation_name = table
                    
        # for column_name in parameters.schema[relation_name]:
        #     if column_name in left_value:
        #         left_value = relation_name + '.' + column_name
        #         break
        left_value_idx = parameters.columns_id[left_value]
        left_value_vec = [0 for _ in range(parameters.column_total_num)]
        left_value_vec[left_value_idx - 1] = 1
        right_value = condition_op.right_value
        for table in parameters.schema:
            if right_value in parameters.schema[table]:
                right_value = table + '.' + right_value
        right_value_vec = [0 for _ in range(max(parameters.column_total_num, parameters.condition_word_dim))]
        column_name = left_value.split('.')[1]
        # Case: operator is 'IS' and right value is 'None'
        if operator == 'IS' and right_value == 'None':
            operator_idx = parameters.compare_ops_id['=']
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value_vec = [0]
        # Case: right value is in the format of table_name.column_name
        elif re.match(r'^[a-z][a-zA-Z0-9_]*\.[a-z][a-zA-Z0-9_]*$', right_value) is not None:
            if right_value.split('.')[0] in parameters.schema:
                # one-hot encoding for operator
                operator_idx = parameters.compare_ops_id[operator]
                operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
                operator_vec[operator_idx - 1] = 1
                # one-hot encoding for column
                right_value_idx = parameters.columns_id[right_value]
                right_value_vec = [0 for _ in range(parameters.column_total_num)]
                right_value_vec[right_value_idx - 1] = 1
            else:
                # one-hot encoding for operator
                operator_idx = parameters.compare_ops_id[operator]
                operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
                operator_vec[operator_idx - 1] = 1
                # one-hot encoding for column
                # right_value_idx = parameters.columns_id[right_value]
                right_value_vec = [0 for _ in range(parameters.column_total_num)]
                # right_value_vec[right_value_idx - 1] = 1

        # Case: right value is a number
        elif parameters.schema[relation_name][column_name] in ['integer', 'bigint', 'numeric', 'real', 'double precision'] :
            # encoding for numerical number(min-max)
            value_max = parameters.min_max_info[relation_name][column_name]['max']
            value_min = parameters.min_max_info[relation_name][column_name]['min']
            if re.match(r'^__ANY__', right_value) is not None:
                operator_idx = parameters.compare_ops_id['=']
                operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
                operator_vec[operator_idx - 1] = 1
                right_value = right_value.strip('\'')[7:].strip('{}')
                right_value_vec = []
                for v in right_value.split(','):
                    v = float(v)
                    right_value_vec.append((v - value_min) / (value_max - value_min))
            else:
                right_value = float(right_value)
                right_value_vec[0] = (right_value - value_min) / (value_max - value_min)
                # the same for operator
                operator_idx = parameters.compare_ops_id[operator]
                operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
                operator_vec[operator_idx - 1] = 1
        # Case: right value is a string
        elif re.match(r'^__LIKE__', right_value) is not None:
            operator_idx = parameters.compare_ops_id['~~']
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value = right_value.strip('\'')[8:]
            right_value_vec = get_str_representation_2(right_value, parameters.word2vec_model, selectivity).tolist()
        elif re.match(r'^__NOTLIKE__', right_value) is not None:
            operator_idx = parameters.compare_ops_id['!~~']
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value = right_value.strip('\'')[11:]
            right_value_vec = get_str_representation_2(right_value, parameters.word2vec_model, selectivity).tolist()
        elif re.match(r'^__NOTEQUAL__', right_value) is not None:
            operator_idx = parameters.compare_ops_id['!=']
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value = right_value.strip('\'')[12:]
            right_value_vec = get_str_representation_2(right_value, parameters.word2vec_model, selectivity).tolist()
        elif re.match(r'^__ANY__', right_value) is not None:
            operator_idx = parameters.compare_ops_id['=']
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value = right_value.strip('\'')[7:].strip('{}')
            right_value_vec = []
            count = 0
            for v in right_value.split(','):
                v = v.strip('"').strip('\'')
                if len(v) > 0:
                    count += 1
                    vec = get_str_representation_2(v, parameters.word2vec_model, selectivity).tolist()
                    if len(right_value_vec) == 0:
                        right_value_vec = [0 for _ in vec]
                    for idx, vv in enumerate(vec):
                        right_value_vec[idx] += vv
            for idx in range(len(right_value_vec)):
                right_value_vec[idx] /= len(right_value.split(','))
        elif right_value == 'None':
            operator_idx = parameters.compare_ops_id['!Null']
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            if operator == 'IS':
                right_value_vec = [1]
            elif operator == '!=':
                right_value_vec = [0]
            else:
                print(operator)
                raise
        else:
            #             print (left_value, operator, right_value)
            operator_idx = parameters.compare_ops_id[operator]
            operator_vec = [0 for _ in range(parameters.compare_ops_total_num)]
            operator_vec[operator_idx - 1] = 1
            right_value_vec = get_str_representation_2(right_value, parameters.word2vec_model, selectivity).tolist()
        vec = [0 for _ in range(parameters.bool_ops_total_num)]
        vec = vec + left_value_vec + operator_vec + right_value_vec
    num_pad = parameters.condition_op_dim - len(vec)
    result = np.pad(vec, (0, num_pad), 'constant')
    #     print 'condition op: ', result
    return result


def encode_condition(condition, relation_name, index_name, parameters, plan_rows=1):
    if len(condition) == 0:
        vecs = [[0 for _ in range(parameters.condition_op_dim)]]
    else:
        vecs = [encode_condition_op(condition_op, relation_name, index_name, parameters, plan_rows) for condition_op in condition if condition_op is not None]
        if len(vecs) > parameters.condition_max_num:
            vecs = vecs[:parameters.condition_max_num]
    num_pad = parameters.condition_max_num - len(vecs)

    result = np.pad(vecs, ((0, num_pad), (0, 0)), 'constant')
    return result

def get_operators(nodes):
    physic_ops_id = {}
    for node in nodes:
        if node.node_type not in physic_ops_id:
            physic_ops_id[node.node_type] = len(physic_ops_id) + 1
    return physic_ops_id

# def get_tables_and_columns(file_path):
#     with open(file_path,'r') as f:
#         schema = json.load(f)
#     columns_id = {}
#     tables_id = {}
#     for table in schema:
#         table_name = table['table']
#         tables_id[table_name] = len(tables_id) + 1
#         for column_name in table['columns']:
#             columns_id[table_name + '.' + column_name] = len(columns_id) + 1
#     return tables_id, columns_id

def get_db_data(db = 'tpch'):
    
    schema_paths = {
        'tpch': 'xx',
        'imdb': 'xx',
        'tpcds': 'xx'
    }
    db_conf = {
        'tpcds': {
            "database":'tpcds',
            "host":'localhost',
            "port":'5436',
            "user":'xx',
            "password":'xx'
        },
        'tpch':
        {
            "database":'tpch',
            "host":'localhost',
            "port":'5434',
            "user":'xx',
            "password":'xx'
        }}
    dbconnector = psycopg2.connect(database=db_conf[db]['database'], host=db_conf[db]['host'], port=db_conf[db]['port'], user=db_conf[db]['user'], password=db_conf[db]['password'])
    with open(schema_paths[db],'r') as f:
        schema = json.load(f)
    columns_id = {}
    tables_id = {}
    min_max_info = {}
    type_info = {}
    for table_info in schema:
        table_name = table_info['table']
        min_max_info[table_name] = {}
        type_info[table_name] = {}
        tables_id[table_name] = len(tables_id) + 1
        for column_info in table_info['columns']:
            column_name = column_info['name']
            column_type = column_info['type']
            type_info[table_name][column_name] = column_type
            if column_type in ['integer', 'bigint', 'numeric', 'real', 'double precision']:
                min_value, max_value = get_min_max_value(dbconnector, column_name, table_name)
                min_max_info[table_name][column_name] = {'min': float(min_value) if isinstance(min_value, Decimal) else min_value, 'max': float(max_value) if isinstance(max_value, Decimal) else max_value}
            columns_id[table_name + '.' + column_name] = len(columns_id) + 1
    return tables_id, columns_id, min_max_info, type_info



def get_min_max_value(dbconnector, column_name, table_name):
    cursor = dbconnector.cursor()
    cursor.execute(f"""
                    SELECT MIN({column_name}), MAX({column_name})
                    FROM {table_name}
                """)
    min_value, max_value = cursor.fetchone()
    return min_value, max_value



def get_conditions():
    condition_op_dim = 8
    condition_word_dim = 500
    codition_ops_id = {
        '>=': 1,
        '<=': 2,
        '>': 3,
        '<': 4,
        '=': 5,
        '!=': 6,
        '~~': 7,
        '!~~': 8
    }
    return condition_op_dim, condition_word_dim, codition_ops_id
    # for node in nodes:

def collect_parameters(nodes,record_path, model_path, db):
    operators_id = get_operators(nodes)
    physic_op_total_num = len(operators_id)
    tables_id, columns_id, min_max_info, schema = get_db_data(db)
    column_total_num = len(columns_id)
    table_total_num = len(tables_id)
    compare_ops_total_num, condition_word_dim, compare_ops_id = get_conditions()
    bool_ops_id = {
        'AND': 1,
        'OR': 2,
        'NOT': 3
    }
    bool_ops_total_num = len(bool_ops_id)
    condition_op_dim = bool_ops_total_num + compare_ops_total_num + max(condition_word_dim, column_total_num) + column_total_num
    train_word2vec_2(record_path, model_path)
    classes =  [1,2,3,4,5,6,7,8,9,10,20,30,40,50,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.08,0.06,0.04,0.02,0.01],
    parameters = Parameters(3, tables_id, columns_id, operators_id, column_total_num, table_total_num, physic_op_total_num, condition_op_dim, condition_word_dim, compare_ops_id, bool_ops_id, bool_ops_total_num, compare_ops_total_num, schema, min_max_info, classes, model_path)
    return parameters





def parse_index_name(index_name, relation, schema):
  
    columns = []
    if relation is None:
        for table_info in schema:
            if table_info in index_name:
                if relation is not None:
                    if relation in table_info:
                        relation = table_info
                else:
                    relation = table_info
    for table_info in schema:
        if table_info == relation:
            for column_info in schema[table_info]:
                if column_info in index_name:
                    columns.append(relation + '.' + column_info)
    return relation, columns


def parse_index_name_2(index_name, relation, candidate_indexes, schema):

    if 'btree_' in index_name:
        index_name = index_name.split('btree_', 1)[1]
    if index_name in candidate_indexes:
        columns = []
        table_name = candidate_indexes[index_name]['table']
        for column in candidate_indexes[index_name]['columns']:
            columns.append(table_name + '.' + column)
        return table_name, columns
    else:
        return parse_index_name(index_name, relation, schema)
    


def node_encoding(node, parameters, candidate_indexes = None):  
    """
    Encode a query plan node into a vector representation.
    
    Args:
        node (QueryPlanNode): The query plan node to encode.
        parameters (Parameters): The parameters for encoding.
    
    Returns:
        tuple: A tuple containing the encoded vectors for the node.
    """
    # Initialize vectors for different components of the node
    operator_vec = np.array([0 for _ in range(parameters.physic_op_total_num)])
    extra_info_vec = np.array([[0 for _ in range(max(parameters.column_total_num, parameters.table_total_num))] for _ in range(parameters.index_max_col_num)])
    condition1_vec = np.array([[0 for _ in range(parameters.condition_op_dim)] for _ in range(parameters.condition_max_num)])
    condition2_vec = np.array([[0 for _ in range(parameters.condition_op_dim)] for _ in range(parameters.condition_max_num)])
    # sample_vec = np.array([1 for _ in range(1000)])  # Sample vector for bitmap operations
    has_condition = 0  # Flag to indicate if the node has conditions

    if node is not None:
        # Encode the operator type as a one-hot vector
        operator = node.node_type
        operator_idx = parameters.physic_ops_id[operator]
        operator_vec[operator_idx - 1] = 1
        # Encode additional information based on the node type
        if operator == 'Materialize' or operator == 'BitmapAnd' or operator == 'Result':
            pass
        elif operator == 'Sort':
            for key in node.sort_keys:
                if key in parameters.columns_id:
                    extra_info_inx = parameters.columns_id[key]      
                    extra_info_vec[0][extra_info_inx - 1] = 1
        elif operator == 'Hash Join' or operator == 'Merge Join' or operator == 'Nested Loop':
            condition1_vec = encode_condition(node.condition, None, None, parameters)
        elif operator == 'Aggregate':
            for key in node.group_keys:
                if key in parameters.columns_id:
                    extra_info_inx = parameters.columns_id[key]      
                    extra_info_vec[0][extra_info_inx - 1] = 1
        elif operator in ['Seq Scan', 'Bitmap Heap Scan', 'Index Scan', 'Bitmap Index Scan', 'Index Only Scan']:
            relation_name = node.relation_name
            index_name = node.index_name
            plan_rows = node.plan_rows
            # Encode the column nme info for index scan nodes
            if operator in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan']:
                relation_name, columns = parse_index_name_2(index_name, relation_name, candidate_indexes, parameters.schema)
                # relation_name, columns = parse_index_name(index_name, relation_name)
                for i in range(parameters.index_max_col_num):
                    if i < len(columns):
                        extra_info_inx = parameters.columns_id[columns[i]]
                        current_info_vec = np.array([0 for _ in range(max(parameters.column_total_num, parameters.table_total_num))])
                        current_info_vec[extra_info_inx - 1] = 1
                        extra_info_vec[i] = current_info_vec
            else:
                extra_info_inx = parameters.tables_id[relation_name]
                extra_info_vec[0][extra_info_inx - 1] = 1
            # if relation_name is not None:
            #     extra_info_inx = parameters.tables_id[relation_name]
            # else:
                # extra_info_inx = parameters.indexes_id[index_name]
            # extra_info_vec[extra_info_inx - 1] = 1
            condition1_vec = encode_condition(node.condition_filter, None, index_name, parameters, plan_rows)
            condition2_vec = encode_condition(node.condition_index, None, index_name, parameters, plan_rows)
            # if 'bitmap' in node:
            #     sample_vec = encode_sample(node['bitmap'])
            #     has_condition = 1
            # if 'bitmap_filter' in node:
            #     sample_vec = bitand(encode_sample(node['bitmap_filter']), sample_vec)
            #     has_condition = 1
            # if 'bitmap_index' in node:
            #     sample_vec = bitand(encode_sample(node['bitmap_index']), sample_vec)
            #     has_condition = 1

    return operator_vec, extra_info_vec, condition1_vec, condition2_vec, has_condition

def cal_distance(node1, node2, parameters):
    """
    Calculate the distance between two nodes based on the Cosine Similarity of their node encodings.
    """
    # Encode the nodes
    operator_vec1, extra_info_vec1, condition1_vec1, condition2_vec1, _ = node_encoding(node1, parameters)
    operator_vec2, extra_info_vec2, condition1_vec2, condition2_vec2, _ = node_encoding(node2, parameters)

    # Convert to PyTorch tensors
    operator_vec1 = torch.tensor(operator_vec1, dtype=torch.float32)
    extra_info_vec1 = torch.tensor(extra_info_vec1, dtype=torch.float32).view(-1)
    condition1_vec1 = torch.tensor(condition1_vec1, dtype=torch.float32).view(-1)
    condition2_vec1 = torch.tensor(condition2_vec1, dtype=torch.float32).view(-1)

    operator_vec2 = torch.tensor(operator_vec2, dtype=torch.float32)
    extra_info_vec2 = torch.tensor(extra_info_vec2, dtype=torch.float32).view(-1)
    condition1_vec2 = torch.tensor(condition1_vec2, dtype=torch.float32).view(-1)
    condition2_vec2 = torch.tensor(condition2_vec2, dtype=torch.float32).view(-1)

    # Concatenate the vectors
    encoding1 = torch.cat((operator_vec1, extra_info_vec1, condition1_vec1, condition2_vec1), dim=0)
    encoding2 = torch.cat((operator_vec2, extra_info_vec2, condition1_vec2, condition2_vec2), dim=0)

    # Calculate cosine similarity
    cosine_sim = F.cosine_similarity(encoding1.unsqueeze(0), encoding2.unsqueeze(0))

    # Convert cosine similarity to distance
    distance = 1 - cosine_sim.item()

    return distance

def cal_uncertainty_based_on_node_distance(node, known_nodes, parameters):
    """
    Calculate the uncertainty of a node based on the distances to known nodes.
    Return the most similar known node and corresponding node distance.
    """
    
    distances = [cal_distance(node, known_node, parameters) for known_node in known_nodes]
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    return min_distance, known_nodes[min_index]

def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
def train_estimation_model(parameters, operator_vec, extra_info_vec, condition1_vec, condition2_vec, target, num_samples = 5):
    '''
    Train the node estimation model.
    '''
    cuda_use = torch.cuda.is_available()
    model = node_estimation.SRU(cuda_use, parameters)
    dataset = TensorDataset(torch.tensor(operator_vec, dtype=torch.float32), torch.tensor(extra_info_vec, dtype=torch.float32), torch.tensor(condition1_vec, dtype=torch.float32), torch.tensor(condition2_vec, dtype=torch.float32), torch.tensor(target, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    if cuda_use:
        model.cuda()
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            operator_feat, extra_info_feat, condition1_feat, condition2_feat, target = batch
            if cuda_use:
                operator_feat = operator_feat.cuda()
                extra_info_feat = extra_info_feat.cuda()
                condition1_feat = condition1_feat.cuda()
                condition2_feat = condition2_feat.cuda()
                target = target.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # outputs = model(operator_feat, extra_info_feat, condition1_feat, condition2_feat)
            # Forward wiz dropout
            outputs,_ = model.mc_dropout_forward(operator_feat, extra_info_feat, condition1_feat, condition2_feat, num_samples)



            # Compute the loss
            loss = criterion(outputs, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss/len(dataloader)}")
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

    print("Training finished.")

    # Save the trained model
    model_path = 'sru_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
def train_estimation_model_2(parameters, feat_vec, target_vec, save_path = 'sru_model.pth', num_samples = 10):
    '''
    Train the node estimation model.
    '''
    set_random_seed(42)
    cuda_use = torch.cuda.is_available()
    model = node_estimation.SRU(cuda_use, parameters)

    operator_tensor = torch.stack([feat[0] for feat in feat_vec])
    extra_info_tensor = torch.stack([feat[1] for feat in feat_vec])
    condition1_tensor = torch.stack([feat[2] for feat in feat_vec])
    condition2_tensor = torch.stack([feat[3] for feat in feat_vec])
    target_tensor = torch.tensor([target for target in target_vec])
    dataset = TensorDataset(operator_tensor, extra_info_tensor, condition1_tensor, condition2_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    if cuda_use:
        model.cuda()
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            operator_feat, extra_info_feat, condition1_feat, condition2_feat, target = batch
            if cuda_use:
                operator_feat = operator_feat.cuda()
                extra_info_feat = extra_info_feat.cuda()
                condition1_feat = condition1_feat.cuda()
                condition2_feat = condition2_feat.cuda()
                target = target.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # outputs = model(operator_feat, extra_info_feat, condition1_feat, condition2_feat)
            # Forward wiz dropout
            outputs, all_outputs = model.mc_dropout_forward(operator_feat, extra_info_feat, condition1_feat, condition2_feat)
            # Compute the loss
            loss = criterion(outputs, target)

            # if epoch in [0,50,100,150,200,250,300]:
            mc_uncertainty = compute_mc_uncertainty(all_outputs)
            print(f"MCD uncertainty: {mc_uncertainty.mean().item()}")
    

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss/len(dataloader)}")
            
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

    print("Training finished.")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def train_estimation_model_3(parameters, feat_vec, target_vec, save_path = 'sru_model.pth', num_samples = 10):
    '''
    Train the node estimation model.
    '''
    set_random_seed(42)
    cuda_use = torch.cuda.is_available()
    model = node_estimation.OperatorModel(cuda_use, parameters)

    extra_info_tensor = torch.stack([feat[1] for feat in feat_vec])
    condition1_tensor = torch.stack([feat[2] for feat in feat_vec])
    condition2_tensor = torch.stack([feat[3] for feat in feat_vec])
    target_tensor = torch.tensor([target for target in target_vec])
    dataset = TensorDataset(extra_info_tensor, condition1_tensor, condition2_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    if cuda_use:
        model.cuda()
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            extra_info_feat, condition1_feat, condition2_feat, target = batch
            if cuda_use:
              
                extra_info_feat = extra_info_feat.cuda()
                condition1_feat = condition1_feat.cuda()
                condition2_feat = condition2_feat.cuda()
                target = target.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # outputs = model(operator_feat, extra_info_feat, condition1_feat, condition2_feat)
            # Forward wiz dropout
            outputs, all_outputs = model.mc_dropout_forward(extra_info_feat, condition1_feat, condition2_feat)
            # Compute the loss
            loss = criterion(outputs, target)

            # if epoch in [0,50,100,150,200,250,300]:
            mc_uncertainty = compute_mc_uncertainty(all_outputs)
    

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if epoch % 100 == 0:
                print(f"MCD uncertainty: {mc_uncertainty.mean().item()}")
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss/len(dataloader)}")
            
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

    print("Training finished.")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def compute_mc_uncertainty(outputs_tensor):
    """
    Compute the uncertainty based on the Monte Carlo dropout outputs.
    outputs_tensor [10,110,28]: num of samples, batch size, num of classes
    """
    probs = F.softmax(outputs_tensor, dim=-1)
    # Mean prediction over MC samples (shape: [110, 28])
    mean_probs = probs.mean(dim=0)
    variance = probs.var(dim=0)

    # 2. Predictive Entropy H(mean) (shape: [110])
    predictive_entropy = -torch.sum(mean_probs * mean_probs.log(), dim=-1)

    # 3. Expected Entropy E[H(p_t)] (shape: [110])
    # Compute entropy of each MC sample, then average
    entropy_each = -torch.sum(probs * probs.log(), dim=-1)  # shape: [10, 110]
    expected_entropy = entropy_each.mean(dim=0)  # shape: [110]

    # 4. Mutual Information (Epistemic Uncertainty) = H(mean) - E[H]
    mutual_information = predictive_entropy - expected_entropy  # shape: [110]

    return variance.max(dim=1)[0]


def update_model(model_path, parameters, feat_vec, target_vec,  save_path, num_epochs=100, batch_size=32, learning_rate=0.001):
    """
    Update the node estimation model with input feat_vec and target_vec.
    
    Args:
        model_path (str): Path to the existing model.
        feat_vec (list): List of feature vectors.
        target_vec (list): List of target values.
        parameters (Parameters): Model parameters.
        save_path (str): Path to save the updated model.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
    """
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Check if CUDA is available
    cuda_use = torch.cuda.is_available()
    
    # Load the existing model
    model = node_estimation.SRU(cuda_use, parameters)
    if model_path and os.path.exists(model_path) and os.path.getsize(model_path) != 0:
        model.load_state_dict(torch.load(model_path))
    
    # Prepare the data
    operator_tensor = torch.stack([feat[0] for feat in feat_vec])
    extra_info_tensor = torch.stack([feat[1] for feat in feat_vec])
    condition1_tensor = torch.stack([feat[2] for feat in feat_vec])
    condition2_tensor = torch.stack([feat[3] for feat in feat_vec])
    target_tensor = torch.tensor([target for target in target_vec])
    
    dataset = TensorDataset(operator_tensor, extra_info_tensor, condition1_tensor, condition2_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if cuda_use:
        model.cuda()
        operator_tensor = operator_tensor.cuda()
        extra_info_tensor = extra_info_tensor.cuda()
        condition1_tensor = condition1_tensor.cuda()
        condition2_tensor = condition2_tensor.cuda()
        target_tensor = target_tensor.cuda()
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            operator_feat, extra_info_feat, condition1_feat, condition2_feat, target = batch
            if cuda_use:
                operator_feat = operator_feat.cuda()
                extra_info_feat = extra_info_feat.cuda()
                condition1_feat = condition1_feat.cuda()
                condition2_feat = condition2_feat.cuda()
                target = target.cuda()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, all_outputs = model.mc_dropout_forward(operator_feat, extra_info_feat, condition1_feat, condition2_feat)
            
            # Compute the loss
            loss = criterion(outputs, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")
        # logging.info(f"MCD uncertainty: {vars.mean()}")
    
    logging.info("Training finished.")
    
    # Save the updated model
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")


def update_model_with_replaybuffer(model_path, parameters, feat_vec, target_vec,  replay_buffer, save_path, num_epochs=100, batch_size=32, learning_rate=0.001, replay_buffer_capacity = 100):
    """
    Update the node estimation model with input feat_vec and target_vec.
    
    Args:
        model_path (str): Path to the existing model.
        feat_vec (list): List of feature vectors.
        target_vec (list): List of target values.
        parameters (Parameters): Model parameters.
        save_path (str): Path to save the updated model.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
    """
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Check if CUDA is available
    cuda_use = torch.cuda.is_available()
    
    # Load the existing model
    model = node_estimation.SRU(cuda_use, parameters)
    # if model_path and os.path.exists(model_path) and os.path.getsize(model_path) != 0:
    #     model.load_state_dict(torch.load(model_path))
    
    # Prepare the data
    operator_tensor = torch.stack([feat[0] for feat in feat_vec])
    extra_info_tensor = torch.stack([feat[1] for feat in feat_vec])
    condition1_tensor = torch.stack([feat[2] for feat in feat_vec])
    condition2_tensor = torch.stack([feat[3] for feat in feat_vec])
    target_tensor = torch.tensor([target for target in target_vec])
    
    dataset = TensorDataset(operator_tensor, extra_info_tensor, condition1_tensor, condition2_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if cuda_use:
        model.cuda()
        operator_tensor = operator_tensor.cuda()
        extra_info_tensor = extra_info_tensor.cuda()
        condition1_tensor = condition1_tensor.cuda()
        condition2_tensor = condition2_tensor.cuda()
        target_tensor = target_tensor.cuda()
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the replay buffer
    # replay_buffer = initial_replay_buffer if initial_replay_buffer else ReplayBuffer(replay_buffer_capacity)
    
    # Add initial data to the replay buffer if it's not pre-filled
   
    for data in dataloader:
        operator_feat, extra_info_feat, condition1_feat, condition2_feat, target = data
        for i in range(len(target)):
            replay_buffer.add((operator_feat[i], extra_info_feat[i], condition1_feat[i], condition2_feat[i], target[i]))
        # replay_buffer.add(data)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # for batch in dataloader:
        if len(replay_buffer) < batch_size:
            batch_size = len(replay_buffer)
        replay_batch = replay_buffer.sample(batch_size)
        operator_feat = torch.stack(([data[0] for data in replay_batch]), dim=0)
        extra_info_feat = torch.stack(([data[1] for data in replay_batch]), dim=0)
        condition1_feat = torch.stack(([data[2] for data in replay_batch]), dim=0)
        condition2_feat = torch.stack(([data[3] for data in replay_batch]), dim=0)
        target = torch.stack(([data[4] for data in replay_batch]), dim=0)
        if cuda_use:
            operator_feat = operator_feat.cuda()
            extra_info_feat = extra_info_feat.cuda()
            condition1_feat = condition1_feat.cuda()
            condition2_feat = condition2_feat.cuda()
            target = target.cuda()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, all_outputs = model.mc_dropout_forward(operator_feat, extra_info_feat, condition1_feat, condition2_feat)
        
        # Compute the loss
        loss = criterion(outputs, target)

        mc_uncertainty = compute_mc_uncertainty(all_outputs)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
            # operator_feat, extra_info_feat, condition1_feat, condition2_feat, target = batch
            # if cuda_use:
            #     operator_feat = operator_feat.cuda()
            #     extra_info_feat = extra_info_feat.cuda()
            #     condition1_feat = condition1_feat.cuda()
            #     condition2_feat = condition2_feat.cuda()
            #     target = target.cuda()
            
            # # Zero the parameter gradients
            # optimizer.zero_grad()
            
            # # Forward pass
            # outputs, vars = model.mc_dropout_forward(operator_feat, extra_info_feat, condition1_feat, condition2_feat)
            
            # # Compute the loss
            # loss = criterion(outputs, target)
            
            # # Backward pass and optimize
            # loss.backward()
            # optimizer.step()
            
            # Print statistics
            # running_loss += loss.item()
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")
        logging.info(f"MCD uncertainty: {mc_uncertainty.mean().item()}")
    
    logging.info("Training finished.")
    
    # Save the updated model
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")


def update_model_with_replaybuffer_2(model_path, parameters, feat_vec, target_vec,  replay_buffer, save_path, num_epochs=100, batch_size=32, learning_rate=0.001, replay_buffer_capacity = 100):
    """
    Update the node estimation model with input feat_vec and target_vec.
    
    Args:
        model_path (str): Path to the existing model.
        feat_vec (list): List of feature vectors.
        target_vec (list): List of target values.
        parameters (Parameters): Model parameters.
        save_path (str): Path to save the updated model.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
    """
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Check if CUDA is available
    cuda_use = torch.cuda.is_available()
    
    # Load the existing model
    model = node_estimation.OperatorModel(cuda_use, parameters)

    # if model_path and os.path.exists(model_path) and os.path.getsize(model_path) != 0:
    #     model.load_state_dict(torch.load(model_path))
    
    # Prepare the data
    extra_info_tensor = torch.stack([feat[1] for feat in feat_vec])
    condition1_tensor = torch.stack([feat[2] for feat in feat_vec])
    condition2_tensor = torch.stack([feat[3] for feat in feat_vec])
    target_tensor = torch.tensor([target for target in target_vec])
    
    dataset = TensorDataset(extra_info_tensor, condition1_tensor, condition2_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if cuda_use:
        model.cuda()
        extra_info_tensor = extra_info_tensor.cuda()
        condition1_tensor = condition1_tensor.cuda()
        condition2_tensor = condition2_tensor.cuda()
        target_tensor = target_tensor.cuda()
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the replay buffer
    # replay_buffer = initial_replay_buffer if initial_replay_buffer else ReplayBuffer(replay_buffer_capacity)
    
    # Add initial data to the replay buffer if it's not pre-filled
   
    for data in dataloader:
        extra_info_feat, condition1_feat, condition2_feat, target = data
        for i in range(len(target)):
            replay_buffer.add((extra_info_feat[i], condition1_feat[i], condition2_feat[i], target[i]))
        # replay_buffer.add(data)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # for batch in dataloader:
        if len(replay_buffer) < batch_size:
            batch_size = len(replay_buffer)
        replay_batch = replay_buffer.sample(batch_size)
        extra_info_feat = torch.stack(([data[0] for data in replay_batch]), dim=0)
        condition1_feat = torch.stack(([data[1] for data in replay_batch]), dim=0)
        condition2_feat = torch.stack(([data[2] for data in replay_batch]), dim=0)
        target = torch.stack(([data[3] for data in replay_batch]), dim=0)
        if cuda_use:
            extra_info_feat = extra_info_feat.cuda()
            condition1_feat = condition1_feat.cuda()
            condition2_feat = condition2_feat.cuda()
            target = target.cuda()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, all_outputs = model.mc_dropout_forward(extra_info_feat, condition1_feat, condition2_feat)
        
        # Compute the loss
        loss = criterion(outputs, target)

        mc_uncertainty = compute_mc_uncertainty(all_outputs)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
            # operator_feat, extra_info_feat, condition1_feat, condition2_feat, target = batch
            # if cuda_use:
            #     operator_feat = operator_feat.cuda()
            #     extra_info_feat = extra_info_feat.cuda()
            #     condition1_feat = condition1_feat.cuda()
            #     condition2_feat = condition2_feat.cuda()
            #     target = target.cuda()
            
            # # Zero the parameter gradients
            # optimizer.zero_grad()
            
            # # Forward pass
            # outputs, vars = model.mc_dropout_forward(operator_feat, extra_info_feat, condition1_feat, condition2_feat)
            
            # # Compute the loss
            # loss = criterion(outputs, target)
            
            # # Backward pass and optimize
            # loss.backward()
            # optimizer.step()
            
            # Print statistics
            # running_loss += loss.item()
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")
        logging.info(f"MCD uncertainty: {mc_uncertainty.mean().item()}")
    
    logging.info("Training finished.")
    
    # Save the updated model
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")


def update_model_with_replaybuffer_3(model_path, parameters, error_nodes, feat_vec, target_vec,  replay_buffer, save_path, num_epochs=100, batch_size=32, learning_rate=0.001, replay_buffer_capacity = 100):
    """
    Update the node estimation model with input feat_vec and target_vec.
    
    Args:
        model_path (str): Path to the existing model.
        feat_vec (list): List of feature vectors.
        target_vec (list): List of target values.
        parameters (Parameters): Model parameters.
        save_path (str): Path to save the updated model.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
    """
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Check if CUDA is available
    cuda_use = torch.cuda.is_available()
    
    # Load the existing model
    model = node_estimation.OperatorModel(cuda_use, parameters)

    # if model_path and os.path.exists(model_path) and os.path.getsize(model_path) != 0:
    #     model.load_state_dict(torch.load(model_path))
    
    # Prepare the data
    extra_info_tensor = torch.stack([feat[1] for feat in feat_vec])
    condition1_tensor = torch.stack([feat[2] for feat in feat_vec])
    condition2_tensor = torch.stack([feat[3] for feat in feat_vec])
    target_tensor = torch.tensor([target for target in target_vec])
    index_tensor = torch.arange(len(target_vec))
    
    dataset = TensorDataset(extra_info_tensor, condition1_tensor, condition2_tensor, target_tensor, index_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if cuda_use:
        model.cuda()
        extra_info_tensor = extra_info_tensor.cuda()
        condition1_tensor = condition1_tensor.cuda()
        condition2_tensor = condition2_tensor.cuda()
        target_tensor = target_tensor.cuda()
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the replay buffer
    # replay_buffer = initial_replay_buffer if initial_replay_buffer else ReplayBuffer(replay_buffer_capacity)
    
    # Add initial data to the replay buffer if it's not pre-filled
   
    for data in dataloader:
        extra_info_feat, condition1_feat, condition2_feat, target, origin_index = data
        for i in range(len(target)):
            replay_buffer.add_2((extra_info_feat[i], condition1_feat[i], condition2_feat[i], target[i]), error_nodes[origin_index[i].item()])
        # replay_buffer.add(data)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # for batch in dataloader:
        if len(replay_buffer) < batch_size:
            batch_size = len(replay_buffer)
        replay_batch = replay_buffer.sample(batch_size)
        extra_info_feat = torch.stack(([data[0] for data in replay_batch]), dim=0)
        condition1_feat = torch.stack(([data[1] for data in replay_batch]), dim=0)
        condition2_feat = torch.stack(([data[2] for data in replay_batch]), dim=0)
        target = torch.stack(([data[3] for data in replay_batch]), dim=0)
        if cuda_use:
            extra_info_feat = extra_info_feat.cuda()
            condition1_feat = condition1_feat.cuda()
            condition2_feat = condition2_feat.cuda()
            target = target.cuda()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, all_outputs = model.mc_dropout_forward(extra_info_feat, condition1_feat, condition2_feat)
        
        # Compute the loss
        loss = criterion(outputs, target)

        mc_uncertainty = compute_mc_uncertainty(all_outputs)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")
        logging.info(f"MCD uncertainty: {mc_uncertainty.mean().item()}")
    
    logging.info("Training finished.")
    
    # Save the updated model
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")


def evaluate_estimation_model(model_path, feat_vec, target_vec, parameters):
    """
    Evaluate the performance of the trained model.
    """
    set_random_seed(42)
    cuda_use = torch.cuda.is_available()
    model = node_estimation.SRU(cuda_use, parameters)
    model.load_state_dict(torch.load(model_path))
    operator_tensor = torch.stack([feat[0] for feat in feat_vec])
    extra_info_tensor = torch.stack([feat[1] for feat in feat_vec])
    condition1_tensor = torch.stack([feat[2] for feat in feat_vec])
    condition2_tensor = torch.stack([feat[3] for feat in feat_vec])
    target_tensor = torch.tensor([target for target in target_vec])
    
    if cuda_use:
        model.cuda()
        operator_tensor = operator_tensor.cuda()
        extra_info_tensor = extra_info_tensor.cuda()
        condition1_tensor = condition1_tensor.cuda()
        condition2_tensor = condition2_tensor.cuda()
        target_tensor = target_tensor.cuda()
        
        
    model.eval()
    # model.train()

    
    # Print condition1_tensor
#     condition_list = condition1_tensor.detach().cpu().numpy().tolist()
#     outputs = []
#     for condition in condition_list:
#         for index, element in enumerate(condition):
#             if element != 0:
#                 outputs.append(index)
#     numpy_array = np.array(outputs)


    
# # Save the NumPy array to a file
#     np.savetxt('condition_infer.txt', numpy_array, delimiter=',', fmt='%f')
    
    outputs = model.forward(operator_tensor, extra_info_tensor, condition1_tensor, condition2_tensor)
    # calculate softmax and entrophy
    probas = F.softmax(outputs, dim=1)
    entropy = -torch.sum(probas * torch.log(probas + 1e-8), dim=1)



    _, predicted = torch.max(outputs, 1)
    total_correct = (predicted == target_tensor).sum().item()
    total_samples = target_tensor.size(0)
    accuracy = total_correct / total_samples

    # Detect mismatches
    mismatches = (predicted != target_tensor).nonzero(as_tuple=True)[0]
    mismatch_info = []
    for idx in mismatches:
        mismatch_info.append({
            'index': idx.item(),
            'predicted': predicted[idx].item(),
            'target': target_tensor[idx].item()
        })

    # Output mismatches
    for info in mismatch_info:
        print(f"Mismatch at index {info['index']}: predicted={info['predicted']}, target={info['target']}")


    return accuracy, entropy, mismatch_info

    # dataset = TensorDataset(operator_tensor, extra_info_tensor, condition1_tensor, condition2_tensor, target_tensor)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # total_correct = 0
    # total_samples = 0
    # with torch.no_grad():
    #     for batch in dataloader:
    #         operator_feat, extra_info_feat, condition1_feat, condition2_feat, target = batch
    #         if cuda_use:
    #             operator_feat = operator_feat.cuda()
    #             extra_info_feat = extra_info_feat.cuda()
    #             condition1_feat = condition1_feat.cuda()
    #             condition2_feat = condition2_feat.cuda()
    #             target = target.cuda()

    #         # Forward pass
    #         outputs = model(operator_feat, extra_info_feat, condition1_feat, condition2_feat)
    #         _, predicted = torch.max(outputs, 1)
    #         total_correct += (predicted == target).sum().item()
    #         total_samples += target.size(0)

    # accuracy = total_correct / total_samples
    # return accuracy


def evaluate_estimation_model_2(model_path, feat_vec, target_vec, parameters):
    """
    Evaluate the performance of the trained model. one model for an operator
    """
    set_random_seed(42)
    cuda_use = torch.cuda.is_available()
    model = node_estimation.OperatorModel(cuda_use, parameters)
    model.load_state_dict(torch.load(model_path))
    extra_info_tensor = torch.stack([feat[1] for feat in feat_vec])
    condition1_tensor = torch.stack([feat[2] for feat in feat_vec])
    condition2_tensor = torch.stack([feat[3] for feat in feat_vec])
    target_tensor = torch.tensor([target for target in target_vec])
    
    if cuda_use:
        model.cuda()
        extra_info_tensor = extra_info_tensor.cuda()
        condition1_tensor = condition1_tensor.cuda()
        condition2_tensor = condition2_tensor.cuda()
        target_tensor = target_tensor.cuda()
        
        
    model.eval()
    # model.train()

    
    # Print condition1_tensor
#     condition_list = condition1_tensor.detach().cpu().numpy().tolist()
#     outputs = []
#     for condition in condition_list:
#         for index, element in enumerate(condition):
#             if element != 0:
#                 outputs.append(index)
#     numpy_array = np.array(outputs)


    
# # Save the NumPy array to a file
#     np.savetxt('condition_infer.txt', numpy_array, delimiter=',', fmt='%f')
    
    outputs = model.forward(extra_info_tensor, condition1_tensor, condition2_tensor)
    # calculate softmax and entrophy
    probas = F.softmax(outputs, dim=1)
    entropy = -torch.sum(probas * torch.log(probas + 1e-8), dim=1)



    _, predicted = torch.max(outputs, 1)
    total_correct = (predicted == target_tensor).sum().item()
    total_samples = target_tensor.size(0)
    accuracy = total_correct / total_samples

    # Detect mismatches
    mismatches = (predicted != target_tensor).nonzero(as_tuple=True)[0]
    mismatch_info = []
    for idx in mismatches:
        mismatch_info.append({
            'index': idx.item(),
            'predicted': predicted[idx].item(),
            'target': target_tensor[idx].item()
        })

    # Output mismatches
    for info in mismatch_info:
        print(f"Mismatch at index {info['index']}: predicted={info['predicted']}, target={info['target']}")


    return accuracy, entropy, mismatch_info


def evaluate_mc_dropout(model_path, feat_vec, target_vec, parameters):
    """
    Evaluate the performance of the trained model.
    """
    cuda_use = torch.cuda.is_available()
    model = node_estimation.SRU(cuda_use, parameters)
    model.load_state_dict(torch.load(model_path))
    if cuda_use:
        model.cuda()
    model.eval()

    operator_tensor = torch.stack([feat[0] for feat in feat_vec])
    extra_info_tensor = torch.stack([feat[1] for feat in feat_vec])
    condition1_tensor = torch.stack([feat[2] for feat in feat_vec])
    condition2_tensor = torch.stack([feat[3] for feat in feat_vec])
    target_tensor = torch.tensor([target for target in target_vec])
    dataset = TensorDataset(operator_tensor, extra_info_tensor, condition1_tensor, condition2_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    total_correct = 0
    total_samples = 0
    total_uncertainty = 0
    with torch.no_grad():
        for batch in dataloader:
            operator_feat, extra_info_feat, condition1_feat, condition2_feat, target = batch
            if cuda_use:
                operator_feat = operator_feat.cuda()
                extra_info_feat = extra_info_feat.cuda()
                condition1_feat = condition1_feat.cuda()
                condition2_feat = condition2_feat.cuda()
                target = target.cuda()

            # Forward pass with MC Dropout
            mean_output, var_output = model.mc_dropout_forward(operator_feat, extra_info_feat, condition1_feat, condition2_feat)
            _, predicted = torch.max(mean_output, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
            total_uncertainty += var_output.mean().item()


    accuracy = total_correct / total_samples
    average_uncertainty = total_uncertainty / len(dataloader)
    return accuracy, average_uncertainty
    

def plan_estimation(query_plans, model, parameter_path = "parameters.json"):
    """
    Refine the what-if estimation of query_plan with model.
    """
    # Load parameters
    with open(parameter_path, 'r') as f:
        parameters = Parameters.from_dict(json.load(f))

    # Feature extraction
    vectors = []
    estimation = []
    # actual_benefits = []
    for query_plan in query_plans:
        vectors = []
        root = build_query_plan_tree(query_plan)
        dfs_nodes = dfs_traverse(root)
        nodes = feature_extractor(root)
        # Feature extraction
        for node in nodes:
    
            operator_vec, extrainfo_vec, condition1, condition2, hash_condition = node_encoding(node, parameters)
            operator_vec = torch.tensor(operator_vec, dtype=torch.float32)
            extrainfo_vec = torch.tensor(extrainfo_vec, dtype=torch.float32).view(-1)
            condition1 = torch.tensor(condition1, dtype=torch.float32).view(-1)
            condition2 = torch.tensor(condition2, dtype=torch.float32).view(-1)
            vectors.append([
                operator_vec,
                extrainfo_vec,
                condition1,
                condition2,
            ])
        operator_vec = np.array([vec[0] for vec in vectors])
        extrainfo_vec = np.array([np.array(vec[1]).reshape(1,-1) for vec in vectors])
        extrainfo_vec = extrainfo_vec.reshape(extrainfo_vec.shape[0], -1)
        condition1 = np.array([np.array(vec[2]).reshape(1,-1) for vec in vectors])
        condition1 = condition1.reshape(condition1.shape[0], -1)
        condition2 = np.array([np.array(vec[3]).reshape(1,-1) for vec in vectors])
        condition2 = condition2.reshape(condition2.shape[0], -1)

         # Convert to PyTorch tensors
        operator_vec = torch.tensor(operator_vec, dtype=torch.float32)
        extrainfo_vec = torch.tensor(extrainfo_vec, dtype=torch.float32)
        condition1 = torch.tensor(condition1, dtype=torch.float32)
        condition2 = torch.tensor(condition2, dtype=torch.float32)

        if model.cuda_use:
            operator_vec = operator_vec.cuda()
            extrainfo_vec = extrainfo_vec.cuda()
            condition1 = condition1.cuda()
            condition2 = condition2.cuda()

        # Forward pass with MC Dropout
        mean_outputs, uncertainties = model.mc_dropout_forward(operator_vec, extrainfo_vec, condition1, condition2, num_samples=5)
        estimation.append(uncertainties)
        # for id in range(len(dfs_nodes)):
        #     if uncertainties[id] < 0.1: # Node Refinement
        #         dfs_nodes[id].attributes['Total Cost New'] = mean_outputs[id] * dfs_nodes[id].attributes['Total Cost']
        #         if dfs_nodes[id].parent:
        #             dfs_nodes[id].parent.update_cost()
        #         # dfs_nodes[id].update_cost()
        #         estimation.append(root.attributes['Total Cost New'])
        #         break
    return estimation


def plan_estimation_test_7(query_plans, model_paths, parameters, candidate_indexes = None, rho = 0.1, alpha = 0.5, query_index_mapping = None):
    """
    Estimate the end-to-end performance of model-based plan estimation.
    Error detection: leaf level.
    Refinement method: uncertainty estimation (MC Dropout + cross entropy).
    Model: Operator model
    
    Args:
        query_plans: List of query plans to estimate
        model_paths: Dictionary of model paths for each operator type
        parameters: Parameters object
        candidate_indexes: Dictionary mapping index names to index info
        rho: Uncertainty threshold for correction
        query_index_mapping: Optional list of (query_id, index_id) tuples corresponding to query_plans
    
    Returns:
        results: List of [cost, uncertainty, CAM, dfs_node] for each plan
        statistics: Dictionary with 'corrected' and 'uncorrected' lists if query_index_mapping is provided
    """
    logging.info("Plan estimation started")
    # Load parameters
    
    multi_2_label = {label: index for index, label in enumerate(parameters.classes)}
    label_2_multi = {index: label for index, label in enumerate(parameters.classes)}

    operator_types = parameters.operator_model_types
    
    # Load the trained model
    results = []
    # check is model exist
    models = {}
    cuda_use = torch.cuda.is_available()
    for key, model_path in model_paths.items():
        if not model_path or not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            logging.info(f"Model {key} not found. Estimate with initial plan.")
            models[key] = None
        else:
            model = node_estimation.OperatorModel(cuda_use, parameters)
            model.load_state_dict(torch.load(model_path))
            if cuda_use:
                 model.cuda()
            model.eval()
            models[key] = model

    # Get known nodes
    known_nodes = []
    multis = []
    
    # Statistics tracking
    statistics = {'corrected': [], 'uncorrected': []} if query_index_mapping else None
    
    for plan_idx, query_plan in enumerate(query_plans):
        root = build_query_plan_tree(query_plan)
        dfs_nodes = dfs_traverse(root)
        nodes = feature_extractor(root, parameters)
        index_related_nodes = []
        similar_nodes = []
        vectors = {operator_type:[] for operator_type in operator_types}
        nodes_to_check = {operator_type:[] for operator_type in operator_types}
        labels = {operator_type:[] for operator_type in operator_types}
        uncertainties = {operator_type:[] for operator_type in operator_types}
        mc_dropout_uncertainties = {operator_type:[] for operator_type in operator_types}
        cross_entropy_uncertainties = {operator_type:[] for operator_type in operator_types}
   
        is_nodes_to_check = False
        # Node encoding
        for index, node in enumerate(nodes):
            # Only check leaf & index related nodes
            if node and isinstance(node, Scan) and node.node_type in nodes_to_check.keys():
                is_nodes_to_check = True
                nodes_to_check[node.node_type].append(dfs_nodes[index])
                # Get node encoding
                operator_vec, extrainfo_vec, condition1, condition2, _ = node_encoding(node, parameters, candidate_indexes)
                operator_vec = torch.tensor(operator_vec, dtype=torch.float32)
                extrainfo_vec = torch.tensor(extrainfo_vec, dtype=torch.float32).view(-1)
                condition1 = torch.tensor(condition1, dtype=torch.float32).view(-1)
                condition2 = torch.tensor(condition2, dtype=torch.float32).view(-1)
                vectors[node.node_type].append([
                    operator_vec,
                    extrainfo_vec,
                    condition1,
                    condition2,
                ])
        if not is_nodes_to_check:
            results.append([root.attributes['Total Cost'], 1, 1, None])
            # Record statistics if mapping is provided
            if statistics is not None and query_index_mapping is not None and plan_idx < len(query_index_mapping):
                query_id, index_id = query_index_mapping[plan_idx]
                statistics['uncorrected'].append({
                    'plan_idx': plan_idx,
                    'query_id': query_id,
                    'index_id': index_id,
                    'uncertainty': 1.0,
                    'MCD': 1.0,
                    'CE': 1.0,
                    'original_cost': float(root.attributes['Total Cost']),
                    'CAM': 1.0,
                    'reason': 'no_nodes_to_check'
                })
            continue
        # CAM prediction
        for key, nodes in nodes_to_check.items():
            model = models[key]
            if len(nodes) != 0:
                # If the model is not None, use the model to predict ; else use the original cost
                if model is None:
                    labels[key] = [1 for node in nodes] 
                    uncertainties[key] = [1 for node in nodes]
                    mc_dropout_uncertainties[key] = [1.0 for node in nodes]
                    cross_entropy_uncertainties[key] = [1.0 for node in nodes]
                else:
                    extra_info_tensor = torch.stack([feat[1] for feat in vectors[key]])
                    condition1_tensor = torch.stack([feat[2] for feat in vectors[key]])
                    condition2_tensor = torch.stack([feat[3] for feat in vectors[key]])
                    mean_outputs, all_ouputs = model.mc_dropout_forward(extra_info_tensor, condition1_tensor, condition2_tensor)
                    mc_dropout_uncertainty = compute_mc_uncertainty(all_ouputs)
                    probs = F.softmax(mean_outputs, dim=1)
                    cross_entropy_uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    _, type_labels = torch.max(mean_outputs, 1)
                    labels[key] = type_labels.detach().cpu().numpy().tolist()
                    combined_uncertainty = alpha * mc_dropout_uncertainty + (1 - alpha) * cross_entropy_uncertainty
                    uncertainties[key] = combined_uncertainty.detach().cpu().numpy().tolist()
                    mc_dropout_uncertainties[key] = mc_dropout_uncertainty.detach().cpu().numpy().tolist()
                    cross_entropy_uncertainties[key] = cross_entropy_uncertainty.detach().cpu().numpy().tolist()
        # Uncertainty-aware
        for key,nodes in nodes_to_check.items():
            for index, node in enumerate(nodes):
                if 'btree' in node.attributes.get('Index Name',''):
                    min_index = index
                    dfs_node = nodes[index]
                    min_uncertainty = uncertainties[key][min_index]
                    min_mc_dropout_uncertainty = mc_dropout_uncertainties[key][min_index]
                    min_cross_entropy_uncertainty = cross_entropy_uncertainties[key][min_index]
                    flag = True
                    break
        min_uncertainty = 1
        min_type = ''
        min_index = -1
        min_mc_dropout_uncertainty = 1.0
        min_cross_entropy_uncertainty = 1.0
        dfs_node = None
        for key, nodes in nodes_to_check.items():
            if len(nodes) != 0:
                for index, node in enumerate(nodes):
                    if uncertainties[key][index] < min_uncertainty:
                        min_uncertainty = uncertainties[key][index]
                        min_mc_dropout_uncertainty = mc_dropout_uncertainties[key][index]
                        min_cross_entropy_uncertainty = cross_entropy_uncertainties[key][index]
                        min_index = index
                        min_type = key
                        dfs_node = nodes[min_index]
        if min_uncertainty < rho and min_index >= 0 and dfs_node is not None:
            CAM = label_2_multi[labels[min_type][min_index]]
            new_cost = CAM * (dfs_node.attributes['Total Cost'] - dfs_node.attributes['Startup Cost'])
            dfs_node.update_cost_2(new_cost)
            if root.attributes['Total Cost New'] < 0:
                logging.info(f"Node {dfs_node.to_json()} is refined with origin cost: {root.attributes['Total Cost']} \n new cost: {new_cost} \n CAM: {CAM}, uncertainty: {min_uncertainty}")
            results.append([root.attributes['Total Cost New'], min_uncertainty, CAM, dfs_node])
            # logging.info(f"Node {dfs_node.to_json()} is refined with origin cost: {root.attributes['Total Cost']} \n new cost: {new_cost} \n CAM: {CAM}, uncertainty: {min_uncertainty}")
            # Record corrected statistics
            if statistics is not None and query_index_mapping is not None and plan_idx < len(query_index_mapping):
                query_id, index_id = query_index_mapping[plan_idx]
                statistics['corrected'].append({
                    'plan_idx': plan_idx,
                    'query_id': query_id,
                    'index_id': index_id,
                    'uncertainty': float(min_uncertainty),
                    'CAM': float(CAM),
                    'original_cost': float(root.attributes['Total Cost']),
                    'new_cost': float(new_cost),
                    'MCD': float(min_mc_dropout_uncertainty),
                    'CE': float(min_cross_entropy_uncertainty)
                })
        else:
            results.append([root.attributes['Total Cost'], min_uncertainty, 1, None])
            # Record uncorrected statistics
            if statistics is not None and query_index_mapping is not None and plan_idx < len(query_index_mapping):
                query_id, index_id = query_index_mapping[plan_idx]
                statistics['uncorrected'].append({
                    'plan_idx': plan_idx,
                    'query_id': query_id,
                    'index_id': index_id,
                    'uncertainty': float(min_uncertainty),
                    'MCD': float(min_mc_dropout_uncertainty),
                    'CE': float(min_cross_entropy_uncertainty),
                    'original_cost': float(root.attributes['Total Cost']),
                    'CAM': 1.0
                })

    if statistics is not None:
        return results, statistics
    else:
        return results


def plan_estimation_regres(query_plans, model_paths, parameters, candidate_indexes=None, rho=0.1, alpha=0.5, query_index_mapping=None):
    """
    Estimate the end-to-end performance of regression model-based plan estimation.
    Uses OperatorModelRegres to predict continuous multiplier values.
    
    Args:
        query_plans: List of query plans to estimate
        model_paths: Dictionary of model paths for each operator type
        parameters: Parameters object
        candidate_indexes: Dictionary mapping index names to index info
        rho: Uncertainty threshold for correction
        alpha: Weight for combining uncertainties (not used in regression, kept for compatibility)
        query_index_mapping: Optional list of (query_id, index_id) tuples corresponding to query_plans
    
    Returns:
        results: List of [cost, uncertainty, CAM, dfs_node] for each plan
        statistics: Dictionary with 'corrected' and 'uncorrected' lists if query_index_mapping is provided
    """
    logging.info("Plan estimation (regression) started")
    
    operator_types = parameters.operator_model_types
    
    # Load the trained models
    results = []
    models = {}
    cuda_use = torch.cuda.is_available()
    for key, model_path in model_paths.items():
        if not model_path or not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            logging.info(f"Model {key} not found. Estimate with initial plan.")
            models[key] = None
        else:
            model = node_estimation.OperatorModelRegres(cuda_use, parameters)
            model.load_state_dict(torch.load(model_path))
            if cuda_use:
                model.cuda()
            model.eval()
            models[key] = model
    
    # Statistics tracking
    statistics = {'corrected': [], 'uncorrected': []} if query_index_mapping else None
    
    for plan_idx, query_plan in enumerate(query_plans):
        root = build_query_plan_tree(query_plan)
        dfs_nodes = dfs_traverse(root)
        nodes = feature_extractor(root, parameters)
        vectors = {operator_type:[] for operator_type in operator_types}
        nodes_to_check = {operator_type:[] for operator_type in operator_types}
        multipliers = {operator_type:[] for operator_type in operator_types}
        uncertainties = {operator_type:[] for operator_type in operator_types}
        mc_dropout_uncertainties = {operator_type:[] for operator_type in operator_types}
   
        is_nodes_to_check = False
        # Node encoding
        for index, node in enumerate(nodes):
            # Only check leaf & index related nodes
            if node and isinstance(node, Scan) and node.node_type in nodes_to_check.keys():
                is_nodes_to_check = True
                nodes_to_check[node.node_type].append(dfs_nodes[index])
                # Get node encoding
                operator_vec, extrainfo_vec, condition1, condition2, _ = node_encoding(node, parameters, candidate_indexes)
                operator_vec = torch.tensor(operator_vec, dtype=torch.float32)
                extrainfo_vec = torch.tensor(extrainfo_vec, dtype=torch.float32).view(-1)
                condition1 = torch.tensor(condition1, dtype=torch.float32).view(-1)
                condition2 = torch.tensor(condition2, dtype=torch.float32).view(-1)
                vectors[node.node_type].append([
                    operator_vec,
                    extrainfo_vec,
                    condition1,
                    condition2,
                ])
        
        if not is_nodes_to_check:
            results.append([root.attributes['Total Cost'], 1, 1, None])
            # Record statistics if mapping is provided
            if statistics is not None and query_index_mapping is not None and plan_idx < len(query_index_mapping):
                query_id, index_id = query_index_mapping[plan_idx]
                statistics['uncorrected'].append({
                    'plan_idx': plan_idx,
                    'query_id': query_id,
                    'index_id': index_id,
                    'uncertainty': 1.0,
                    'MCD': 1.0,
                    'original_cost': float(root.attributes['Total Cost']),
                    'CAM': 1.0,
                    'reason': 'no_nodes_to_check'
                })
            continue
        
        # CAM prediction using regression model
        for key, nodes in nodes_to_check.items():
            model = models[key]
            if len(nodes) != 0:
                # If the model is not None, use the model to predict; else use multiplier 1
                if model is None:
                    multipliers[key] = [1.0 for node in nodes]
                    uncertainties[key] = [1.0 for node in nodes]
                    mc_dropout_uncertainties[key] = [1.0 for node in nodes]
                else:
                    extra_info_tensor = torch.stack([feat[1] for feat in vectors[key]])
                    condition1_tensor = torch.stack([feat[2] for feat in vectors[key]])
                    condition2_tensor = torch.stack([feat[3] for feat in vectors[key]])
                    
                    # Use MC dropout for uncertainty estimation
                    mean_outputs, all_outputs = model.mc_dropout_forward(extra_info_tensor, condition1_tensor, condition2_tensor, num_samples=10)
                    mc_dropout_uncertainty = torch.std(all_outputs, dim=0).mean(dim=1)  # Standard deviation as uncertainty
                    
                    # Get multiplier values (mean of MC dropout samples)
                    multipliers[key] = mean_outputs.detach().cpu().numpy().flatten().tolist()
                    uncertainties[key] = mc_dropout_uncertainty.detach().cpu().numpy().tolist()
                    mc_dropout_uncertainties[key] = mc_dropout_uncertainty.detach().cpu().numpy().tolist()
        
        # Uncertainty-aware refinement
        min_uncertainty = 1.0
        min_type = ''
        min_index = -1
        min_mc_dropout_uncertainty = 1.0
        dfs_node = None
        for key, nodes in nodes_to_check.items():
            if len(nodes) != 0:
                for index, node in enumerate(nodes):
                    if uncertainties[key][index] < min_uncertainty:
                        min_uncertainty = uncertainties[key][index]
                        min_mc_dropout_uncertainty = mc_dropout_uncertainties[key][index]
                        min_index = index
                        min_type = key
                        dfs_node = nodes[min_index]
        
        if min_uncertainty < rho and min_index >= 0 and dfs_node is not None:
            CAM = multipliers[min_type][min_index]
            new_cost = CAM * (dfs_node.attributes['Total Cost'] - dfs_node.attributes['Startup Cost'])
            dfs_node.update_cost_2(new_cost)
            if root.attributes['Total Cost New'] < 0:
                logging.info(f"Node {dfs_node.to_json()} is refined with origin cost: {root.attributes['Total Cost']} \n new cost: {new_cost} \n CAM: {CAM}, uncertainty: {min_uncertainty}")
            results.append([root.attributes['Total Cost New'], min_uncertainty, CAM, dfs_node])
            # Record corrected statistics
            if statistics is not None and query_index_mapping is not None and plan_idx < len(query_index_mapping):
                query_id, index_id = query_index_mapping[plan_idx]
                statistics['corrected'].append({
                    'plan_idx': plan_idx,
                    'query_id': query_id,
                    'index_id': index_id,
                    'uncertainty': float(min_uncertainty),
                    'CAM': float(CAM),
                    'original_cost': float(root.attributes['Total Cost']),
                    'new_cost': float(new_cost),
                    'MCD': float(min_mc_dropout_uncertainty)
                })
        else:
            results.append([root.attributes['Total Cost'], min_uncertainty, 1, None])
            # Record uncorrected statistics
            if statistics is not None and query_index_mapping is not None and plan_idx < len(query_index_mapping):
                query_id, index_id = query_index_mapping[plan_idx]
                statistics['uncorrected'].append({
                    'plan_idx': plan_idx,
                    'query_id': query_id,
                    'index_id': index_id,
                    'uncertainty': float(min_uncertainty),
                    'MCD': float(min_mc_dropout_uncertainty),
                    'original_cost': float(root.attributes['Total Cost']),
                    'CAM': 1.0
                })

    if statistics is not None:
        return results, statistics
    else:
        return results



def dfs_traverse(node):
    """
    Traverse the query plan tree in a depth-first manner.
    
    Args:
        node (QueryPlanNode): The root node of the query plan tree.
    
    Returns:
        list: A list of nodes in the query plan tree.
    """
    nodes = [node]
    for child in node.children:
        nodes += dfs_traverse(child)
    return nodes    

def dfs_traverse_leaf(node):
    """
    Traverse the query plan tree in a depth-first manner.
    
    Args:
        node (QueryPlanNode): The root node of the query plan tree.
    
    Returns:
        list: A list of nodes in the query plan tree.
    """
    nodes = []
    if not node.children:
        nodes.append(node)
    for child in node.children:
        nodes += dfs_traverse_leaf(child)
    return nodes



if __name__ == "__main__":
    gen_CAMs()
    # train_word2vec_2()
    # train_embedding()
    # test_feature_extraction()
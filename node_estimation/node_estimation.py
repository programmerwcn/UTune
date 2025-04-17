import torch.nn.functional as F
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import query_plan_node
import json


class OperatorModel(torch.nn.Module):
    def __init__(self, cuda_use, parameters, droupout_rate=0.2):
        super(OperatorModel, self).__init__()

        self.use_time = 0.0
        self.cuda_use = cuda_use
        # self.input_operator_dim = 4
        self.input_extra_info_dim = 16
        self.input_condition1_dim = 64
        self.input_condition2_dim = 64
        self.dropout_rate = droupout_rate

        # node encoding info
        # self.operat_dim = parameters.physic_op_total_num
        self.extra_info = max(parameters.column_total_num, parameters.table_total_num) * parameters.index_max_col_num
        self.condition1_vec_dim = parameters.condition_op_dim * parameters.condition_max_num
        self.condition2_vec_dim = parameters.condition_op_dim * parameters.condition_max_num
        self.input_dim = 64
        # self.input_dim = self.operat_dim + self.extra_info + self.condition1_vec_dim + self.condition2_vec_dim
        self.output_dim = 64
        self.num_classes = parameters.num_classes


        # node embedding
        # self.feature_mpl_operation = torch.nn.Linear(self.operat_dim, self.input_operator_dim)
        self.feature_mpl_extra_info = torch.nn.Linear(self.extra_info, self.input_extra_info_dim)
        self.feature_mpl_condition1 = torch.nn.Linear(self.condition1_vec_dim, self.input_condition1_dim)
        self.feature_mpl_condition2 = torch.nn.Linear(self.condition2_vec_dim, self.input_condition2_dim)

        # Dropout layers
        self.dropout = nn.Dropout(droupout_rate)


        # self.W_xou = torch.nn.Linear(self.input_dim * 4, 3 * self.mem_dim)
        #output module
        self.out_mlp1 = nn.Linear(self.input_extra_info_dim + self.input_condition1_dim + self.input_condition2_dim , self.output_dim)
        self.out_mlp2 = nn.Linear(self.output_dim,self.num_classes)


        torch.nn.init.xavier_uniform_(self.feature_mpl_extra_info.weight)
        torch.nn.init.constant_(self.feature_mpl_extra_info.bias, 0)
        torch.nn.init.xavier_uniform_(self.feature_mpl_condition1.weight)
        torch.nn.init.constant_(self.feature_mpl_condition1.bias, 0)
        torch.nn.init.xavier_uniform_(self.feature_mpl_condition2.weight)
        torch.nn.init.constant_(self.feature_mpl_condition2.bias, 0)

        torch.nn.init.xavier_uniform_(self.out_mlp1.weight)
        torch.nn.init.constant_(self.out_mlp1.bias, 0)
        torch.nn.init.xavier_uniform_(self.out_mlp2.weight)
        torch.nn.init.constant_(self.out_mlp2.bias, 0)

        if self.cuda_use:

            self.feature_mpl_extra_info.cuda()
            self.feature_mpl_condition1.cuda()
            self.feature_mpl_condition2.cuda()
            self.out_mlp1.cuda()
            self.out_mlp2.cuda()


    
    def forward(self, extra_info_feat, condition1_feat, condition2_feat):

        # batch_size = node_order.shape[0]
        # Ensure all tensors are on the same device
        device = next(self.parameters()).device

        extra_info_feat = extra_info_feat.to(device)
        condition1_feat = condition1_feat.to(device)
        condition2_feat = condition2_feat.to(device)


        # Embedding layer   
        extra_info_feat = F.relu(self.feature_mpl_extra_info(extra_info_feat))
        condition1_feat = F.relu(self.feature_mpl_condition1(condition1_feat))
        condition2_feat = F.relu(self.feature_mpl_condition2(condition2_feat))
        x = torch.cat((extra_info_feat, condition1_feat, condition2_feat), 1)
        
    # Estimation Layer
        out = torch.relu(self.out_mlp1(x))
        out = self.dropout(out)
        out = self.out_mlp2(out)
        return out


    def mc_dropout_forward(self, extra_info_feat, condition1_feat, condition2_feat, num_samples=10):
        self.train()  # Ensure dropout is enabled
        outputs = []
        for _ in range(num_samples):
            outputs.append(self.forward(extra_info_feat, condition1_feat, condition2_feat))
        outputs_tensor = torch.stack(outputs)
        mean_output = torch.mean(outputs_tensor, dim=0)
        var_output = torch.var(outputs_tensor, dim=0)
        return mean_output, outputs_tensor
       
class OperatorModelRegres(torch.nn.Module):
    def __init__(self, cuda_use, parameters, droupout_rate=0.2):
        super(OperatorModelRegres, self).__init__()

        self.use_time = 0.0
        self.cuda_use = cuda_use
        # self.input_operator_dim = 4
        self.input_extra_info_dim = 16
        self.input_condition1_dim = 64
        self.input_condition2_dim = 64
        self.dropout_rate = droupout_rate

        # node encoding info
        # self.operat_dim = parameters.physic_op_total_num
        self.extra_info = max(parameters.column_total_num, parameters.table_total_num) * parameters.index_max_col_num
        self.condition1_vec_dim = parameters.condition_op_dim * parameters.condition_max_num
        self.condition2_vec_dim = parameters.condition_op_dim * parameters.condition_max_num
        self.input_dim = 64
        # self.input_dim = self.operat_dim + self.extra_info + self.condition1_vec_dim + self.condition2_vec_dim
        self.output_dim = 64

        # node embedding
        # self.feature_mpl_operation = torch.nn.Linear(self.operat_dim, self.input_operator_dim)
        self.feature_mpl_extra_info = torch.nn.Linear(self.extra_info, self.input_extra_info_dim)
        self.feature_mpl_condition1 = torch.nn.Linear(self.condition1_vec_dim, self.input_condition1_dim)
        self.feature_mpl_condition2 = torch.nn.Linear(self.condition2_vec_dim, self.input_condition2_dim)

        # Dropout layers
        self.dropout = nn.Dropout(droupout_rate)

        # self.W_xou = torch.nn.Linear(self.input_dim * 4, 3 * self.mem_dim)
        #output module - regression output (single value)
        self.out_mlp1 = nn.Linear(self.input_extra_info_dim + self.input_condition1_dim + self.input_condition2_dim , self.output_dim)
        self.out_mlp2 = nn.Linear(self.output_dim, 1)

        torch.nn.init.xavier_uniform_(self.feature_mpl_extra_info.weight)
        torch.nn.init.constant_(self.feature_mpl_extra_info.bias, 0)
        torch.nn.init.xavier_uniform_(self.feature_mpl_condition1.weight)
        torch.nn.init.constant_(self.feature_mpl_condition1.bias, 0)
        torch.nn.init.xavier_uniform_(self.feature_mpl_condition2.weight)
        torch.nn.init.constant_(self.feature_mpl_condition2.bias, 0)

        torch.nn.init.xavier_uniform_(self.out_mlp1.weight)
        torch.nn.init.constant_(self.out_mlp1.bias, 0)
        torch.nn.init.xavier_uniform_(self.out_mlp2.weight)
        torch.nn.init.constant_(self.out_mlp2.bias, 0)

        if self.cuda_use:

            self.feature_mpl_extra_info.cuda()
            self.feature_mpl_condition1.cuda()
            self.feature_mpl_condition2.cuda()
            self.out_mlp1.cuda()
            self.out_mlp2.cuda()

    
    def forward(self, extra_info_feat, condition1_feat, condition2_feat):

        # batch_size = node_order.shape[0]
        # Ensure all tensors are on the same device
        device = next(self.parameters()).device

        extra_info_feat = extra_info_feat.to(device)
        condition1_feat = condition1_feat.to(device)
        condition2_feat = condition2_feat.to(device)

        # Embedding layer   
        extra_info_feat = F.relu(self.feature_mpl_extra_info(extra_info_feat))
        condition1_feat = F.relu(self.feature_mpl_condition1(condition1_feat))
        condition2_feat = F.relu(self.feature_mpl_condition2(condition2_feat))
        x = torch.cat((extra_info_feat, condition1_feat, condition2_feat), 1)
        
        # Estimation Layer
        out = torch.relu(self.out_mlp1(x))
        out = self.dropout(out)
        out = self.out_mlp2(out)
        return out

    def mc_dropout_forward(self, extra_info_feat, condition1_feat, condition2_feat, num_samples=10):
        self.train()  # Ensure dropout is enabled
        outputs = []
        for _ in range(num_samples):
            outputs.append(self.forward(extra_info_feat, condition1_feat, condition2_feat))
        outputs_tensor = torch.stack(outputs)
        mean_output = torch.mean(outputs_tensor, dim=0)
        var_output = torch.var(outputs_tensor, dim=0)
        return mean_output, outputs_tensor



class SRU(torch.nn.Module):
    def __init__(self, cuda_use, parameters, droupout_rate=0.2):
        super(SRU, self).__init__()

        self.use_time = 0.0
        self.cuda_use = cuda_use
        self.input_operator_dim = 4
        self.input_extra_info_dim = 16
        self.input_condition1_dim = 64
        self.input_condition2_dim = 64
        self.dropout_rate = droupout_rate
        # self.feature_dim = feature_dim
        # self.input_dim = embed_dim
        # self.mem_dim = mem_dim
        # self.outputdim = outputdim

        # node encoding info
        self.operat_dim = parameters.physic_op_total_num
        self.extra_info = max(parameters.column_total_num, parameters.table_total_num) * parameters.index_max_col_num
        self.condition1_vec_dim = parameters.condition_op_dim * parameters.condition_max_num
        self.condition2_vec_dim = parameters.condition_op_dim * parameters.condition_max_num
        self.input_dim = 64
        # self.input_dim = self.operat_dim + self.extra_info + self.condition1_vec_dim + self.condition2_vec_dim
        self.output_dim = 64
        self.num_classes = parameters.num_classes


        # node embedding
        self.feature_mpl_operation = torch.nn.Linear(self.operat_dim, self.input_operator_dim)
        self.feature_mpl_extra_info = torch.nn.Linear(self.extra_info, self.input_extra_info_dim)
        self.feature_mpl_condition1 = torch.nn.Linear(self.condition1_vec_dim, self.input_condition1_dim)
        self.feature_mpl_condition2 = torch.nn.Linear(self.condition2_vec_dim, self.input_condition2_dim)

        # Dropout layers
        self.dropout = nn.Dropout(droupout_rate)

        # self.feature_mpl_operation = torch.nn.Linear(self.operat_dim, self.input_dim)
        # self.feature_mpl_table = torch.nn.Linear(self.table_dim, self.input_dim)
        # self.feature_mpl_filter = torch.nn.Linear(self.filter_dim, self.input_dim)
        # self.feature_mpl_join = torch.nn.Linear(self.join_dim , self.input_dim)

        # self.feature_mpl_operation_2 = torch.nn.Linear(self.input_dim, self.input_dim)
        # self.feature_mpl_table_2 =  torch.nn.Linear(self.input_dim, self.input_dim)
        # self.feature_mpl_filter_2  = torch.nn.Linear(self.input_dim, self.input_dim)
        # self.feature_mpl_join_2  = torch.nn.Linear(self.input_dim, self.input_dim)


        # self.W_xou = torch.nn.Linear(self.input_dim * 4, 3 * self.mem_dim)
        #output module
        self.out_mlp1 = nn.Linear(self.input_operator_dim + self.input_extra_info_dim + self.input_condition1_dim + self.input_condition2_dim , self.output_dim)
        self.out_mlp2 = nn.Linear(self.output_dim,self.num_classes)

        torch.nn.init.xavier_uniform_(self.feature_mpl_operation.weight)
        torch.nn.init.constant_(self.feature_mpl_operation.bias, 0)
        torch.nn.init.xavier_uniform_(self.feature_mpl_extra_info.weight)
        torch.nn.init.constant_(self.feature_mpl_extra_info.bias, 0)
        torch.nn.init.xavier_uniform_(self.feature_mpl_condition1.weight)
        torch.nn.init.constant_(self.feature_mpl_condition1.bias, 0)
        torch.nn.init.xavier_uniform_(self.feature_mpl_condition2.weight)
        torch.nn.init.constant_(self.feature_mpl_condition2.bias, 0)

        torch.nn.init.xavier_uniform_(self.out_mlp1.weight)
        torch.nn.init.constant_(self.out_mlp1.bias, 0)
        torch.nn.init.xavier_uniform_(self.out_mlp2.weight)
        torch.nn.init.constant_(self.out_mlp2.bias, 0)

        # torch.nn.init.xavier_uniform_(self.feature_mpl_operation.weight)
        # torch.nn.init.constant_(self.feature_mpl_operation.bias, 0)
        # torch.nn.init.xavier_uniform_(self.feature_mpl_filter.weight)
        # torch.nn.init.constant_(self.feature_mpl_filter.bias, 0)
        # torch.nn.init.xavier_uniform_(self.feature_mpl_join.weight)
        # torch.nn.init.constant_(self.feature_mpl_join.bias, 0)
        # torch.nn.init.xavier_uniform_(self.feature_mpl_operation_2.weight)
        # torch.nn.init.constant_(self.feature_mpl_operation_2.bias, 0)
        # torch.nn.init.xavier_uniform_(self.feature_mpl_filter_2.weight)
        # torch.nn.init.constant_(self.feature_mpl_filter_2.bias, 0)
        # torch.nn.init.xavier_uniform_(self.feature_mpl_join_2.weight)
        # torch.nn.init.constant_(self.feature_mpl_join_2.bias, 0)

        # torch.nn.init.xavier_uniform_(self.W_xou.weight)
        # torch.nn.init.constant_(self.W_xou.bias, 0)
        # torch.nn.init.xavier_uniform_(self.out_mlp1.weight)
        # torch.nn.init.constant_(self.out_mlp1.bias, 0)
        # torch.nn.init.xavier_uniform_(self.out_mlp2.weight)
        # torch.nn.init.constant_(self.out_mlp2.bias, 0)

        if self.cuda_use:
            self.feature_mpl_operation.cuda()
            self.feature_mpl_extra_info.cuda()
            self.feature_mpl_condition1.cuda()
            self.feature_mpl_condition2.cuda()
            self.out_mlp1.cuda()
            self.out_mlp2.cuda()



    def forward(self, oprator_feat, extra_info_feat, condition1_feat, condition2_feat):

        # batch_size = node_order.shape[0]
        # Ensure all tensors are on the same device
        device = next(self.parameters()).device
        oprator_feat = oprator_feat.to(device)
        extra_info_feat = extra_info_feat.to(device)
        condition1_feat = condition1_feat.to(device)
        condition2_feat = condition2_feat.to(device)
        # h = torch.zeros(batch_size, self.mem_dim, device=device)
        # c = torch.zeros(batch_size, self.mem_dim, device=device)
        #h = torch.zeros(batch_size, self.mem_dim)
        #c = torch.zeros(batch_size, self.mem_dim)

        # Embedding layer   
        op_feat = F.relu(self.feature_mpl_operation(oprator_feat))
        extra_info_feat = F.relu(self.feature_mpl_extra_info(extra_info_feat))
        condition1_feat = F.relu(self.feature_mpl_condition1(condition1_feat))
        condition2_feat = F.relu(self.feature_mpl_condition2(condition2_feat))
        x = torch.cat((op_feat, extra_info_feat, condition1_feat, condition2_feat), 1)


        
        # Estimation Layer
        out = torch.relu(self.out_mlp1(x))
        out = self.dropout(out)
        out = self.out_mlp2(out)
        return out

        # op_feat = F.relu(self.feature_mpl_operation(op_feat))
        # op_feat = F.relu(self.feature_mpl_operation_2(op_feat))
        # tb_feat = F.relu(self.feature_mpl_table(tb_feat))
        # tb_feat = F.relu(self.feature_mpl_table_2(tb_feat))
        # ft_feat = F.relu(self.feature_mpl_filter(ft_feat))
        # ft_feat = F.relu(self.feature_mpl_filter_2(ft_feat))
        # join_feat = F.relu(self.feature_mpl_join(join_feat))
        # join_feat = F.relu(self.feature_mpl_join_2(join_feat))
        # x = torch.cat((op_feat, tb_feat, ft_feat, join_feat), 1)

        # xou = self.W_xou(x)
        # xx, ff, rr = torch.split(xou, xou.size(1) // 3, dim=1)
        # ff = torch.sigmoid(ff)
        # rr = torch.sigmoid(rr)

        # self._run_init(h, c, xx, ff, rr, x, node_order)
        # for n in range(1, node_order.max() + 1):
        #     self._run_SRU(n, h, c, xx, ff, rr, x, node_order, adjacency_list, edge_order)


        # hid_output = F.relu(self.out_mlp1(h))
        # out = torch.sigmoid(self.out_mlp2(hid_output))
        # return out

    def mc_dropout_forward(self, operator_feat, extra_info_feat, condition1_feat, condition2_feat, num_samples=10):
        self.train()  # Ensure dropout is enabled
        outputs = []
        for _ in range(num_samples):
            outputs.append(self.forward(operator_feat, extra_info_feat, condition1_feat, condition2_feat))
        outputs_tensor = torch.stack(outputs)
        mean_output = torch.mean(outputs_tensor, dim=0)
        var_output = torch.var(outputs_tensor, dim=0)
        return mean_output, outputs_tensor
       
    


    def _run_init (self, h, c, xx, ff, rr, features, node_order):
        node_mask = node_order == 0
        c[node_mask, :] = (1 - ff[node_mask, :]) * xx[node_mask, :]
        h[node_mask, :] = rr[node_mask, :] * torch.tanh(c[node_mask, :]) + (1 - rr[node_mask, :]) * features[node_mask, :]




    def _run_SRU(self, iteration, h, c, xx, ff, rr, features, node_order, adjacency_list, edge_order):
        node_mask = node_order == iteration
        edge_mask = edge_order == iteration

        adjacency_list = adjacency_list[edge_mask, :]
        parent_indexes = adjacency_list[:, 0]
        child_indexes = adjacency_list[:, 1]

        child_c = c[child_indexes, :]
        _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
        child_counts = tuple(child_counts)

        parent_children = torch.split(child_c, child_counts)
        parent_list = [item.sum(0) for item in parent_children]
        c_sum = torch.stack(parent_list)

        f = ff[node_mask, :]
        r = rr[node_mask, :]
        c[node_mask, :] = f * c_sum + (1 - f) * xx[node_mask, :]
        h[node_mask, :] = r * torch.tanh(c[node_mask, :]) + (1 - r) * features[node_mask, :]






    def _base_call_postgre(self, op_feat, tb_feat, ft_feat, join_feat):
        device = next(self.parameters()).device
        op_feat = F.relu(self.feature_mpl_operation(op_feat))
        op_feat = F.relu(self.feature_mpl_operation_2(op_feat))
        tb_feat = F.relu(self.feature_mpl_table(tb_feat))
        tb_feat = F.relu(self.feature_mpl_table_2(tb_feat))
        ft_feat = F.relu(self.feature_mpl_filter(ft_feat))
        ft_feat = F.relu(self.feature_mpl_filter_2(ft_feat))
        join_feat = F.relu(self.feature_mpl_join(join_feat))
        join_feat = F.relu(self.feature_mpl_join_2(join_feat))
        x = torch.cat((op_feat, tb_feat, ft_feat, join_feat), 1)

        xou = self.W_xou(x)
        xx, ff, rr = torch.split(xou, xou.size(1) // 3, dim=1)
        ff = torch.sigmoid(ff)
        rr = torch.sigmoid(rr)

        c = (1 - ff) * xx
        h = rr * torch.tanh(c) + (1 - rr) * x

        hid_output = F.relu(self.out_mlp1(h))
        out = torch.sigmoid(self.out_mlp2(hid_output))
        return out, c






    def _join_call_postgre(self, op_feat, tb_feat, ft_feat, join_feat, left_child, right_child):
        op_feat = F.relu(self.feature_mpl_operation(op_feat))
        op_feat = F.relu(self.feature_mpl_operation_2(op_feat))
        tb_feat = F.relu(self.feature_mpl_table(tb_feat))
        tb_feat = F.relu(self.feature_mpl_table_2(tb_feat))
        ft_feat = F.relu(self.feature_mpl_filter(ft_feat))
        ft_feat = F.relu(self.feature_mpl_filter_2(ft_feat))
        join_feat = F.relu(self.feature_mpl_join(join_feat))
        join_feat = F.relu(self.feature_mpl_join_2(join_feat))
        x = torch.cat((op_feat, tb_feat, ft_feat, join_feat), 1)

        xou = self.W_xou(x)
        xx, ff, rr = torch.split(xou, xou.size(1) // 3, dim=1)

        ff = torch.sigmoid(ff)
        rr = torch.sigmoid(rr)
        c = ff * (left_child + right_child) + (1 - ff) * xx
        h = rr * torch.tanh(c) + (1 - rr) * x

        hid_output = F.relu(self.out_mlp1(h))
        out = torch.sigmoid(self.out_mlp2(hid_output))

        return out, c

# def evaluate_model_with_uncertainty(model, operator_feat, extra_info_feat, condition1_feat, condition2_feat, threshold, num_samples=10):
#         model.eval()  # Set the model to evaluation mode
#         total_loss = 0.0
#         criterion = nn.MSELoss()
    
#         # Forward pass with MC Dropout
#         mean_outputs, std_outputs = model.mc_dropout_forward(operator_feat, extra_info_feat, condition1_feat, condition2_feat, num_samples=num_samples)

#         # Compute the loss
#         loss = criterion(mean_output, target)
#         total_loss += loss.item()

#         # Evaluate uncertainty
#         if std_output.mean().item() > threshold:
#             print("Uncertainty too high, giving up the result.")
#             continue

#         average_loss = total_loss / len(dataloader)
#         print(f"Validation Loss: {average_loss}")

def test():
    # Example data (replace with your actual data)
    operator_feat = np.random.rand(100, 12)  # Example feature tensor
    extra_info_feat = np.random.rand(100, 122)
    condition1_feat = np.random.rand(100, 1716)
    condition2_feat = np.random.rand(100, 1716)
    node_order = np.random.randint(0, 5, (100, 1))
    adjacency_list = np.random.randint(0, 100, (100, 2))
    edge_order = np.random.randint(0, 5, (100, 1))
    target = np.random.rand(100, 1)  # Example target tensor

    # Create a dataset and DataLoader
    dataset = TensorDataset(torch.tensor(operator_feat, dtype=torch.float32), torch.tensor(extra_info_feat, dtype=torch.float32), torch.tensor(condition1_feat, dtype=torch.float32), torch.tensor(condition2_feat, dtype=torch.float32), torch.tensor(node_order, dtype=torch.int64), torch.tensor(adjacency_list, dtype=torch.int64), torch.tensor(edge_order, dtype=torch.int64), torch.tensor(target, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate the model
    cuda_use = torch.cuda.is_available()
    # Load parameters
    with open('/home/wcn/indexAdvisor/ACCUCB-PostgreSQL/parameters.json', 'r') as file:
        data = json.load(file)
    parameters = query_plan_node.Parameters.from_dict(data)
    model = SRU(cuda_use, parameters)

    if cuda_use:
        model.cuda()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for batch in dataloader:
            operator_feat, extra_info_feat, condition1_feat, condition2_feat, node_order, adjacency_list, edge_order, target = batch
            if cuda_use:
                operator_feat = operator_feat.cuda()
                extra_info_feat = extra_info_feat.cuda()
                condition1_feat = condition1_feat.cuda()
                condition2_feat = condition2_feat.cuda()
                node_order = node_order.cuda()
                adjacency_list = adjacency_list.cuda()
                edge_order = edge_order.cuda()
                target = target.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(operator_feat, extra_info_feat, condition1_feat, condition2_feat)

            # Compute the loss
            loss = criterion(outputs, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

    print("Training finished.")

    # Evaluation loop
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            operator_feat, extra_info_feat, condition1_feat, condition2_feat, node_order, adjacency_list, edge_order, target = batch
            if cuda_use:
                operator_feat = operator_feat.cuda()
                extra_info_feat = extra_info_feat.cuda()
                condition1_feat = condition1_feat.cuda()
                condition2_feat = condition2_feat.cuda()
                node_order = node_order.cuda()
                adjacency_list = adjacency_list.cuda()
                edge_order = edge_order.cuda()
                target = target.cuda()

            # Forward pass
            outputs = model(operator_feat, extra_info_feat, condition1_feat, condition2_feat)

            # Compute the loss
            loss = criterion(outputs, target)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss}")

def find_inaccurate_node_regres(original_tree, what_if_tree, ind_tree, ind_table, ind_cols, min_multi=0.01, max_multi=100.0, tolerance=0.1):
    """
    Detect the node related with candidate index in query plan that leads to inaccurate what-if estimation.
    Use binary search to find the multiplier that makes cost_improvement consistent with time_improvement.
    
    Args:
        original_tree: Original query plan tree
        what_if_tree: What-if query plan tree
        ind_tree: Actual index query plan tree
        ind_table: Index table name
        ind_cols: Index column names
        min_multi: Minimum multiplier for binary search (default: 0.01)
        max_multi: Maximum multiplier for binary search (default: 100.0)
        tolerance: Tolerance for binary search convergence (default: 0.1)
    
    Returns:
        tuple: (inaccurate_nodes, final_multi, refined_cost)
            - inaccurate_nodes: (None, what_if_node, ind_node) or None
            - final_multi: Final multiplier found
            - refined_cost: Refined total cost
    """
    # Detect the node related with index scan in what_if_tree & ind_tree
    what_if_nodes = query_plan_node.dfs_traverse(what_if_tree)
    ind_nodes = query_plan_node.dfs_traverse(ind_tree)
    
    # Find related what_if_node
    find_related_what_if_node = False
    what_if_node = None
    for node in what_if_nodes:
        if node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in node.attributes:
            what_if_index_name = node.attributes['Index Name']
            if 'btree' in what_if_index_name:
                what_if_node = node
                find_related_what_if_node = True
                break
    
    if not find_related_what_if_node:
        for node in what_if_nodes:
            if node.node_type == 'Seq Scan' and 'Relation Name' in node.attributes:
                what_if_table_name = node.attributes['Relation Name']
                if what_if_table_name == ind_table:
                    what_if_node = node
                    find_related_what_if_node = True
                    break
    
    if not find_related_what_if_node or what_if_node is None:
        return None, 1, what_if_tree.attributes.get('Total Cost', 0)
    
    # Find related ind_node
    ind_node = None
    for node in ind_nodes:
        if node.node_type in ['Index Scan', 'Bitmap Index Scan', 'Index Only Scan'] and 'Index Name' in node.attributes:
            ind_index_name = node.attributes['Index Name']
            if ind_table in ind_index_name:
                flg = True
                for ind_col in ind_cols:
                    if ind_col not in ind_index_name:
                        flg = False
                        break
                if flg:
                    ind_node = node
                    break
    
    if ind_node is None:
        return None, 1, what_if_tree.attributes.get('Total Cost', 0)
    
    # Binary search for multiplier
    current_min = min_multi
    current_max = max_multi
    improvement_tolerance = 0.08  # Same as compare_nodes_2
    
    # Save original cost for restoration
    original_what_if_node_cost = what_if_node.attributes.get('Total Cost', 0)
    original_what_if_node_startup = what_if_node.attributes.get('Startup Cost', 0)
    original_what_if_tree_cost = what_if_tree.attributes.get('Total Cost', 0)
    
    final_multi = 1.0
    found_accurate = False
    
    # Binary search loop
    while current_max - current_min >= tolerance:
        mid_multi = (current_min + current_max) / 2.0
        
        # Update what_if_node cost with mid_multi
        node_cost = original_what_if_node_cost - original_what_if_node_startup
        new_cost = node_cost * mid_multi
        what_if_node.update_cost_2(new_cost)
        what_if_tree.attributes['Total Cost'] = what_if_tree.attributes.get('Total Cost New', original_what_if_tree_cost)
        
        # Calculate improvements
        original_total_cost = original_tree.cal_cost('Total Cost')
        what_if_total_cost = what_if_tree.cal_cost('Total Cost')
        original_actual_time = original_tree.cal_cost('Actual Total Time')
        ind_actual_time = ind_tree.cal_cost('Actual Total Time')
        
        # Check if we have valid data
        if (original_total_cost is None or what_if_total_cost is None or 
            original_actual_time is None or ind_actual_time is None or
            original_total_cost == 0 or original_actual_time == 0):
            # Restore node attributes and tree cost
            what_if_node.attributes['Total Cost'] = original_what_if_node_cost
            what_if_node.attributes['Startup Cost'] = original_what_if_node_startup
            what_if_tree.attributes['Total Cost'] = original_what_if_tree_cost
            break
        
        cost_improvement = (original_total_cost - what_if_total_cost) / original_total_cost
        time_improvement = (original_actual_time - ind_actual_time) / original_actual_time
        
        # Check if accurate (within tolerance)
        if abs(cost_improvement - time_improvement) <= improvement_tolerance:
            final_multi = mid_multi
            found_accurate = True
            # Don't restore here, we'll use this multiplier
            break
        
        # Adjust search range based on improvement comparison
        if cost_improvement > time_improvement:
            # Overestimated: cost improvement is larger than time improvement
            # Need to increase multiplier (which will increase what_if_total_cost, reducing cost_improvement)
            current_min = mid_multi
        else:
            # Underestimated: cost improvement is smaller than time improvement
            # Need to decrease multiplier (which will decrease what_if_total_cost, increasing cost_improvement)
            current_max = mid_multi
        
        # Restore original cost for next iteration
        what_if_node.attributes['Total Cost'] = original_what_if_node_cost
        what_if_node.attributes['Startup Cost'] = original_what_if_node_startup
        what_if_tree.attributes['Total Cost'] = original_what_if_tree_cost
    
    # If we didn't find an accurate value, use average of final range
    if not found_accurate:
        final_multi = (current_min + current_max) / 2.0
        # Apply final multiplier and calculate refined cost
        node_cost = original_what_if_node_cost - original_what_if_node_startup
        new_cost = node_cost * final_multi
        what_if_node.update_cost_2(new_cost)
    
    refined_cost = what_if_tree.attributes.get('Total Cost New', original_what_if_tree_cost)
    
    return (None, what_if_node, ind_node), final_multi, refined_cost

if __name__ == '__main__':
    test()
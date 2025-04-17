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

if __name__ == '__main__':
    test()
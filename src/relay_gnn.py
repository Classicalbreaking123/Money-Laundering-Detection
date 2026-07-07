import torch
import torch.nn as nn
import torch.nn.functional as F


class RelayGNNLayer(nn.Module):
    def __init__(self, hidden_dim, relay_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.relay_dim = relay_dim

        self.in_msg_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_msg_linear = nn.Linear(hidden_dim, hidden_dim)

        self.gate_linear = nn.Linear(3 * hidden_dim + relay_dim, 1)

        self.relay_linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.normal_linear = nn.Linear(hidden_dim, hidden_dim)

        self.self_linear = nn.Linear(hidden_dim, hidden_dim)

    def aggregate_incoming(self, h, edge_index):
        num_nodes = h.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        transformed = self.in_msg_linear(h[src])

        incoming_sum = torch.zeros(
            num_nodes,
            self.hidden_dim,
            device=h.device
        )
        incoming_sum.index_add_(0, dst, transformed)

        incoming_count = torch.zeros(num_nodes, device=h.device)
        ones = torch.ones(dst.size(0), device=h.device)
        incoming_count.index_add_(0, dst, ones)

        incoming_count = incoming_count.unsqueeze(1).clamp(min=1.0)
        incoming_mean = incoming_sum / incoming_count
        return incoming_mean

    def aggregate_outgoing(self, h, edge_index):
        num_nodes = h.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        transformed = self.out_msg_linear(h[dst])

        outgoing_sum = torch.zeros(
            num_nodes,
            self.hidden_dim,
            device=h.device
        )
        outgoing_sum.index_add_(0, src, transformed)

        outgoing_count = torch.zeros(num_nodes, device=h.device)
        ones = torch.ones(src.size(0), device=h.device)
        outgoing_count.index_add_(0, src, ones)

        outgoing_count = outgoing_count.unsqueeze(1).clamp(min=1.0)
        outgoing_mean = outgoing_sum / outgoing_count
        return outgoing_mean

    def forward(self, h, edge_index, relay_x):
        m_in = self.aggregate_incoming(h, edge_index)
        m_out = self.aggregate_outgoing(h, edge_index)

        gate_input = torch.cat([h, m_in, m_out, relay_x], dim=1)
        gate = torch.sigmoid(self.gate_linear(gate_input))

        relay_input = torch.cat([m_in, m_out], dim=1)
        u_relay = self.relay_linear(relay_input)

        normal_input = m_in + m_out
        u_normal = self.normal_linear(normal_input)

        u = gate * u_relay + (1.0 - gate) * u_normal

        candidate = F.relu(self.self_linear(h) + u)
        h_next = h + candidate

        return h_next


class RelayGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        relay_dim,
        hidden_dim=64,
        num_layers=3
    ):
        super().__init__()

        self.in_dim = in_dim
        self.relay_dim = relay_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_linear = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                RelayGNNLayer(
                    hidden_dim=hidden_dim,
                    relay_dim=relay_dim
                )
            )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        relay_x = data.relay_x

        h = self.input_linear(x)

        for layer in self.layers:
            h = layer(h, edge_index, relay_x)

        logits = self.classifier(h).squeeze(-1)
        return logits

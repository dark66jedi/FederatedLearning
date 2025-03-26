import flwr as fl
from flwr.server.strategy import FedAvg

# Define the strategy
strategy = FedAvg()

# Start the Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)

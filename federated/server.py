import flwr as fl
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

class CustomFedAvg(FedAvg):
    def __init__(self, server_instance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server = server_instance

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.server.on_fit_result(server_round, results)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes]]
    ) -> Optional[float]:
        self.server.on_evaluate_result(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class FederatedLearningServer:
    def __init__(self, 
                 num_rounds: int = 5, 
                 fraction_fit: float = 1.0, 
                 min_fit_clients: int = 2, 
                 min_available_clients: int = 2,
                 output_dir: str = "./server_results"):
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.output_dir = output_dir
        self.global_history = {
            'round': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'loss': [],
            'client_metrics': {}
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
    
    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate metrics from multiple clients weighted by number of examples."""
        # Calculate weighted averages for each metric
        if not metrics:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "loss": 0.0}
        
        total_examples = sum([num_examples for num_examples, _ in metrics])
        
        if total_examples == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "loss": 0.0}
        
        # Handle metrics safely - clients might not all report the same metrics
        aggregated_metrics = {}
        metric_keys = ["accuracy", "precision", "recall", "f1", "test_loss"]
        
        for key in metric_keys:
            values = [num_examples * m.get(key, 0.0) for num_examples, m in metrics if key in m]
            if values:
                aggregated_metrics[key] = sum(values) / total_examples
            else:
                aggregated_metrics[key] = 0.0
                
        # Rename test_loss to loss for consistency
        if "test_loss" in aggregated_metrics:
            aggregated_metrics["loss"] = aggregated_metrics.pop("test_loss")
            
        return aggregated_metrics
    
    def fit_config(self, server_round: int) -> Dict:
        """Return training configuration dict for each round."""
        return {
            "round": server_round,
            "batch_size": 32,
            "epochs": 1,
            "learning_rate": 0.001 * (0.9 ** server_round)  # Decreasing learning rate
        }
    
    def evaluate_config(self, server_round: int) -> Dict:
        """Return evaluation configuration dict for each round."""
        return {
            "round": server_round,
            "batch_size": 32
        }
    
    def on_fit_result(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]]):
        """Process results after each training round."""
        print(f"\n--- Round {server_round} Training Results ---")
        
        # Store client-specific metrics
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            metrics = fit_res.metrics
            
            if client_id not in self.global_history['client_metrics']:
                self.global_history['client_metrics'][client_id] = {
                    'round': [],
                    'train_loss': [],
                    'test_loss': [],
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': []
                }
            
            client_history = self.global_history['client_metrics'][client_id]
            client_history['round'].append(server_round)
            
            # Safely add metrics (not all clients might report all metrics)
            metric_keys = ['train_loss', 'test_loss', 'accuracy', 'precision', 'recall', 'f1']
            for key in metric_keys:
                if key in metrics:
                    client_history[key].append(metrics[key])
                    print(f"Client {client_id} - {key.replace('_', ' ').title()}: {metrics[key]:.4f}")
                else:
                    # Append None or the last value to keep arrays aligned
                    if client_history[key]:
                        client_history[key].append(client_history[key][-1])
                    else:
                        client_history[key].append(None)
    
    def on_evaluate_result(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]]):
        """Process results after each evaluation round."""
        print(f"\n--- Round {server_round} Evaluation Results ---")
        
        if not results:
            print("No evaluation results received.")
            return
            
        # Extract metrics
        metrics_list = [(eval_res.num_examples, eval_res.metrics) for _, eval_res in results]
        aggregated_metrics = self.weighted_average(metrics_list)
        
        # Store global metrics
        self.global_history['round'].append(server_round)
        self.global_history['accuracy'].append(aggregated_metrics.get('accuracy', 0.0))
        self.global_history['precision'].append(aggregated_metrics.get('precision', 0.0))
        self.global_history['recall'].append(aggregated_metrics.get('recall', 0.0))
        self.global_history['f1'].append(aggregated_metrics.get('f1', 0.0))
        self.global_history['loss'].append(aggregated_metrics.get('loss', 0.0))
        
        print(f"Global Metrics - Round {server_round}:")
        print(f"  Accuracy: {aggregated_metrics.get('accuracy', 0.0):.4f}")
        print(f"  Precision: {aggregated_metrics.get('precision', 0.0):.4f}")
        print(f"  Recall: {aggregated_metrics.get('recall', 0.0):.4f}")
        print(f"  F1 Score: {aggregated_metrics.get('f1', 0.0):.4f}")
        print(f"  Loss: {aggregated_metrics.get('loss', 0.0):.4f}")
    
    def save_results(self):
        """Save evaluation history to files and generate plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"global_results_{timestamp}.json")
        
        # Save global results as JSON
        with open(results_file, 'w') as f:
            json.dump(self.global_history, f, indent=2)
        
        # Save global results as CSV
        csv_path = os.path.join(self.output_dir, f"global_results_{timestamp}.csv")
        df = pd.DataFrame({
            'round': self.global_history['round'],
            'accuracy': self.global_history['accuracy'],
            'precision': self.global_history['precision'],
            'recall': self.global_history['recall'],
            'f1': self.global_history['f1'],
            'loss': self.global_history['loss']
        })
        df.to_csv(csv_path, index=False)
        
        # Generate global plots
        plots_dir = os.path.join(self.output_dir, "plots")
        
        # Global accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.global_history['round'], self.global_history['accuracy'], 'g-', label='Accuracy')
        plt.title('Global Accuracy over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"global_accuracy_{timestamp}.png"))
        plt.close()
        
        # Global precision and recall plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.global_history['round'], self.global_history['precision'], 'm-', label='Precision')
        plt.plot(self.global_history['round'], self.global_history['recall'], 'c-', label='Recall')
        plt.title('Global Precision and Recall over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"global_precision_recall_{timestamp}.png"))
        plt.close()
        
        # Global F1 score plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.global_history['round'], self.global_history['f1'], 'orange', label='F1 Score')
        plt.title('Global F1 Score over Rounds')
        plt.xlabel('Round')
        plt.ylabel('F1 Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"global_f1_{timestamp}.png"))
        plt.close()
        
        # Global loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.global_history['round'], self.global_history['loss'], 'r-', label='Loss')
        plt.title('Global Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"global_loss_{timestamp}.png"))
        plt.close()
        
        # Generate per-client plots
        client_dir = os.path.join(self.output_dir, "client_metrics")
        os.makedirs(client_dir, exist_ok=True)
        
        for client_id, metrics in self.global_history['client_metrics'].items():
            # Save client metrics to CSV
            client_file = os.path.join(client_dir, f"client_{client_id}_metrics_{timestamp}.csv")
            client_df = pd.DataFrame(metrics)
            client_df.to_csv(client_file, index=False)
            
            # Plot client metrics
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            if metrics['train_loss'] and any(x is not None for x in metrics['train_loss']):
                plt.plot(metrics['round'], metrics['train_loss'], 'b-', label='Train Loss')
            if metrics['test_loss'] and any(x is not None for x in metrics['test_loss']):
                plt.plot(metrics['round'], metrics['test_loss'], 'r-', label='Test Loss')
            plt.title(f'Client {client_id} - Loss over Rounds')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            if metrics['accuracy'] and any(x is not None for x in metrics['accuracy']):
                plt.plot(metrics['round'], metrics['accuracy'], 'g-', label='Accuracy')
            if metrics['precision'] and any(x is not None for x in metrics['precision']):
                plt.plot(metrics['round'], metrics['precision'], 'm-', label='Precision')
            if metrics['recall'] and any(x is not None for x in metrics['recall']):
                plt.plot(metrics['round'], metrics['recall'], 'c-', label='Recall')
            if metrics['f1'] and any(x is not None for x in metrics['f1']):
                plt.plot(metrics['round'], metrics['f1'], 'orange', label='F1 Score')
            plt.title(f'Client {client_id} - Performance Metrics over Rounds')
            plt.xlabel('Round')
            plt.ylabel('Score')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(client_dir, f"client_{client_id}_metrics_{timestamp}.png"))
            plt.close()
        
        print(f"All results saved to {self.output_dir}")
    
    def start_server(self):
        """Initialize and start the FL server."""
        strategy = CustomFedAvg(
            server_instance=self,
            fraction_fit=self.fraction_fit,
            min_fit_clients=self.min_fit_clients,
            min_available_clients=self.min_available_clients,
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
            evaluate_metrics_aggregation_fn=self.weighted_average,
        )

        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy
        )

        # After server ends, save results
        self.save_results()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated learning rounds")
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fraction of clients to sample")
    parser.add_argument("--min_fit_clients", type=int, default=2, help="Minimum number of clients to train in each round")
    parser.add_argument("--min_available_clients", type=int, default=2, help="Minimum number of available clients required")
    parser.add_argument("--output_dir", type=str, default="./server_results", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    print(f"Starting Federated Learning Server with:")
    print(f"- Number of rounds: {args.rounds}")
    print(f"- Minimum fit clients: {args.min_fit_clients}")
    print(f"- Minimum available clients: {args.min_available_clients}")
    print(f"- Output directory: {args.output_dir}")
    print(f"Waiting for clients to connect...")
    
    server = FederatedLearningServer(
        num_rounds=args.rounds,
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        output_dir=args.output_dir
    )
    
    server.start_server()
"""
Command-Line Interface (CLI) for the Bioprocess PINN Digital Twin.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pinn_bioprocess.core.kinetics import BioprocessParameters
from pinn_bioprocess.core.simulator import BioreactorSimulator
from pinn_bioprocess.models.neural_pinn import BioprocessPINN
from pinn_bioprocess.training.metrics import compute_regression_metrics
from pinn_bioprocess.training.trainer import BioprocessPINNTrainer


def run_train(args: argparse.Namespace) -> None:
    """Executes PINN training and reports benchmark metrics."""
    print("=" * 80)
    print("ELEVATE BIOTECH | PINN BIOPROCESS DIGITAL TWIN ENGINE")
    print("=" * 80)

    params = BioprocessParameters(t_final_h=args.duration)
    sim = BioreactorSimulator(params)

    if args.input_telemetry:
        print(f"\n[1/3] Loading batch telemetry from {args.input_telemetry}...")
        gt, obs = sim.load_telemetry_file(args.input_telemetry)
        params.t_final_h = max(1e-3, float(gt["t"][-1]))
        params.X0 = float(gt["X"][0])
        params.S0 = float(gt["S"][0])
        params.CL0 = float(gt["CL"][0])
        params.P0 = float(gt["P"][0])
        print(f"Operational Horizon: {params.t_final_h:.2f} h | Epochs: {args.epochs} | Learning Rate: {args.lr}")
        print(f"  - Telemetry samples loaded: {len(gt['t'])}")
    else:
        print(f"Operational Horizon: {args.duration} h | Epochs: {args.epochs} | Learning Rate: {args.lr}")
        print("\n[1/3] Generating synthetic bioprocess telemetry & laboratory sampling...")
        gt, obs = sim.generate_multirate_sensors(random_seed=args.seed)
        print(f"  - Ground truth points: {len(gt['t'])}")
        print(f"  - Offline analytical assays (HPLC/Biomass): {len(obs['t_offline'])} points")
        print(f"  - Online optical DO sensor: {len(obs['t_online'])} points")

    print("\n[2/3] Initializing PyTorch BioprocessPINN and Physics Residual Collocation Engine...")
    model = BioprocessPINN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        t_max=params.t_final_h,
        scale_X=max(12.0, float(np.max(gt["X"])) * 1.2),
        scale_S=max(40.0, float(np.max(gt["S"])) * 1.2),
        scale_CL=max(0.010, float(np.max(gt["CL"])) * 1.2),
        scale_P=max(5.0, float(np.max(gt["P"])) * 1.2),
    )
    trainer = BioprocessPINNTrainer(
        model=model,
        params=params,
        n_collocation=args.collocation,
        lambda_phys=args.lambda_phys,
        lambda_ic=args.lambda_ic,
        lambda_balance=args.lambda_balance,
    )

    print(f"\n[3/3] Training PINN over {args.epochs} epochs with autograd physics feedback...")
    fit_res = trainer.fit(obs, epochs=args.epochs, lr=args.lr, verbose=True)
    print(f"Optimization converged in {fit_res['training_time_seconds']:.2f} seconds.")

    # Validation against continuous ground truth
    pred = trainer.predict(gt["t"])
    metrics = compute_regression_metrics(gt, pred)

    print("\n" + "=" * 80)
    print("GROUND TRUTH TRAJECTORY VALIDATION METRICS (DENSE EVALUATION N=401)")
    print("=" * 80)
    for state_name, m in metrics.items():
        unit = "mg/L" if "Oxygen" in state_name else "g/L"
        print(f"{state_name:<16}: R² = {m['r2']:.4f} | MAE = {m['mae']:.4f} {unit} | RMSE = {m['rmse']:.4f} {unit}")
    print("=" * 80)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_data = {
            "training_time_seconds": fit_res["training_time_seconds"],
            "final_loss": fit_res["final_loss"],
            "metrics": metrics,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"\nResults successfully written to: {out_path}")


def run_simulate(args: argparse.Namespace) -> None:
    """Executes pure mechanistic ODE simulation."""
    params = BioprocessParameters(t_final_h=args.duration)
    sim = BioreactorSimulator(params)
    gt = sim.simulate_dense(n_points=args.points)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = "time_h,biomass_g_L,substrate_g_L,dissolved_oxygen_g_L,product_g_L\n"
    lines = [header]
    for i in range(len(gt["t"])):
        lines.append(f"{gt['t'][i]:.4f},{gt['X'][i]:.4f},{gt['S'][i]:.4f},{gt['CL'][i]:.6f},{gt['P'][i]:.4f}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Dense ODE simulation exported to: {out_path} ({len(gt['t'])} points)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pinn-bioprocess",
        description="Physics-Informed Neural Network (PINN) Bioprocess Digital Twin",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the PINN digital twin on bioreactor telemetry")
    train_parser.add_argument("--duration", type=float, default=40.0, help="Cultivation horizon [h]")
    train_parser.add_argument("--epochs", type=int, default=4000, help="Training epochs")
    train_parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    train_parser.add_argument("--collocation", type=int, default=150, help="Collocation points")
    train_parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    train_parser.add_argument("--num-layers", type=int, default=3, help="Number of hidden layers")
    train_parser.add_argument("--lambda-phys", type=float, default=0.05, help="Weight for physics loss")
    train_parser.add_argument("--lambda-ic", type=float, default=0.15, help="Weight for initial condition loss")
    train_parser.add_argument("--lambda-balance", type=float, default=0.01, help="Weight for balance loss")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument(
        "--input-telemetry",
        type=str,
        default=None,
        help="Optional CSV or bioprocess-visual-twin JSON telemetry file",
    )
    train_parser.add_argument("--output", type=str, default="results/metrics.json", help="Path to save metrics JSON")

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run numerical ODE integration")
    sim_parser.add_argument("--duration", type=float, default=40.0, help="Cultivation horizon [h]")
    sim_parser.add_argument("--points", type=int, default=401, help="Number of evaluation points")
    sim_parser.add_argument("--output", type=str, default="results/ground_truth.csv", help="Path to save CSV")

    args = parser.parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "simulate":
        run_simulate(args)


if __name__ == "__main__":
    main()

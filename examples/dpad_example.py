"""
Example script demonstrating DPAD training with your data.

This script shows how to:
1. Load configuration for DPAD
2. Train a DPAD model
3. Make predictions
4. Compare with PSID
"""

import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).parent
sys.path.insert(0, str(root))

from utils.config import get_config
from utils.logger import setup_logger
from training.components.trainer import Trainer


def train_dpad_example():
    """Train DPAD model on behavioral data."""

    print("=" * 60)
    print("DPAD Training Example")
    print("=" * 60)

    # 1. Load DPAD configuration
    config_path = root / "training/setups/dpad_behavioral.yaml"
    config = get_config(config_path)

    # Setup logger
    logger = setup_logger(config.results.log_dir, name="dpad_example")
    logger.info(f"Configuration loaded from: {config_path}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"nx={config.model.nx}, n1={config.model.n1}")

    # 2. Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(config)

    # 3. Split data
    logger.info("Splitting data into train/val/test...")
    trainer.split_data()
    logger.info(f"Train: {len(trainer.train_loader.dataset)} trials")
    logger.info(f"Val: {len(trainer.val_loader.dataset)} trials")
    logger.info(f"Test: {len(trainer.test_loader.dataset)} trials")

    # 4. Train model
    logger.info("Starting DPAD training...")
    logger.info(
        "Note: DPAD training takes longer than PSID (gradient-based optimization)"
    )
    logger.info("Expected time: 30-60 minutes depending on hardware")

    val_results = trainer.train()

    # 5. Display results
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Validation Pearson correlation: {val_results['pearson_r_mean']:.4f}")

    # 6. Examine latent states
    logger.info("\nLatent State Analysis:")
    Xp = val_results["Xp"][0]  # First trial
    logger.info(f"Latent state shape: {Xp.shape}")
    logger.info(f"  - Behavior-prioritized (x^1): dimensions 0-{config.model.n1-1}")
    logger.info(
        f"  - Non-prioritized (x^2): dimensions {config.model.n1}-{config.model.nx-1}"
    )

    logger.info("\nModel saved to: " + str(config.results.save_dir))
    logger.close()

    return trainer, val_results


def compare_psid_dpad():
    """Compare PSID and DPAD on the same data."""

    print("\n" + "=" * 60)
    print("Comparing PSID and DPAD")
    print("=" * 60)

    from utils.config import get_config

    # Load both configs
    psid_config = get_config(root / "training/setups/psid_behavioral.yaml")
    dpad_config = get_config(root / "training/setups/dpad_behavioral.yaml")

    # Train both models
    print("\n1. Training PSID (fast, linear)...")
    psid_trainer = Trainer(psid_config)
    psid_trainer.split_data()
    psid_results = psid_trainer.train()

    print("\n2. Training DPAD (slower, nonlinear)...")
    dpad_trainer = Trainer(dpad_config)
    dpad_trainer.split_data()
    dpad_results = dpad_trainer.train()

    # Compare results
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print(f"PSID Pearson R: {psid_results['pearson_r_mean']:.4f}")
    print(f"DPAD Pearson R: {dpad_results['pearson_r_mean']:.4f}")

    improvement = (
        (dpad_results["pearson_r_mean"] - psid_results["pearson_r_mean"])
        / psid_results["pearson_r_mean"]
        * 100
    )
    print(f"\nImprovement: {improvement:+.1f}%")

    if improvement > 5:
        print("✓ DPAD shows significant improvement - nonlinear dynamics present!")
    elif improvement < -5:
        print("! PSID performs better - data may be more linear or DPAD overfitting")
    else:
        print("≈ Similar performance - dynamics may be approximately linear")

    return psid_results, dpad_results


def demonstrate_method_codes():
    """Demonstrate different DPAD architectures via method codes."""

    print("\n" + "=" * 60)
    print("DPAD Method Code Examples")
    print("=" * 60)

    method_codes = {
        "Linear (like PSID)": "DPAD",
        "Nonlinear Cz only": "DPAD_Cz2HL128U",
        "Fully nonlinear": "DPAD_uAKCzCy2HL128U",
        "With L2 regularization": "DPAD_uAKCzCy2HL128U_RGL2L1e-2",
        "With early stopping": "DPAD_uAKCzCy2HL128U_ErSV16",
        "Smaller network": "DPAD_uAKCzCy2HL64U",
    }

    for name, code in method_codes.items():
        print(f"\n{name}:")
        print(f"  method_code: '{code}'")

        # Parse to show what it means
        from DPAD import DPADModel

        try:
            args = DPADModel.DPADModel.prepare_args(code)

            if "A1_args" in args and args["A1_args"].get("units"):
                print(
                    f"  - A (state transition): {len(args['A1_args']['units'])} layers, {args['A1_args']['units'][0]} units"
                )
            else:
                print(f"  - A (state transition): Linear")

            if "Cz1_args" in args and args["Cz1_args"].get("units"):
                print(
                    f"  - Cz (behavior decoder): {len(args['Cz1_args']['units'])} layers, {args['Cz1_args']['units'][0]} units"
                )
            else:
                print(f"  - Cz (behavior decoder): Linear")

        except Exception as e:
            print(f"  (Could not parse: {e})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DPAD Example Script")
    parser.add_argument(
        "--mode",
        choices=["train", "compare", "method_codes"],
        default="train",
        help="What to demonstrate",
    )

    args = parser.parse_args()

    if args.mode == "train":
        # Just train DPAD
        trainer, results = train_dpad_example()

    elif args.mode == "compare":
        # Compare PSID and DPAD
        psid_results, dpad_results = compare_psid_dpad()

    elif args.mode == "method_codes":
        # Show different method code options
        demonstrate_method_codes()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

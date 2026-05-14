"""
Forecast pipeline entry point.

Run from the project root:

  # Train all demand types (Optuna + CV selection) and save backtest
  python src/main.py train

  # Train specific types with more Optuna trials
  python src/main.py train --types smooth --trials 50

  # Generate final H-week forecast using saved artifacts (run after train)
  python src/main.py predict

  # Train then immediately predict in one shot
  python src/main.py train predict
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.train import run_training, DEMAND_TYPES


def _add_common_args(p: argparse.ArgumentParser):
    p.add_argument('--types', nargs='+', choices=DEMAND_TYPES, default=DEMAND_TYPES,
                   metavar='TYPE', help='Demand types (default: all)')
    p.add_argument('--artifacts', default='artifacts/models',
                   help='Model artifact directory (default: artifacts/models)')
    p.add_argument('--predictions', default='data/gold/forecasting/predictions',
                   help='Predictions root directory (default: data/gold/forecasting/predictions)')


def _cmd_train(args):
    run_training(
        demand_types=args.types,
        artifact_base=Path(args.artifacts),
        predictions_base=Path(args.predictions),
        n_trials=args.trials,
        horizon=args.horizon,
    )


def _cmd_predict(args):
    from pipeline.predict import run_final_forecast
    run_final_forecast(
        demand_types=args.types,
        artifact_base=Path(args.artifacts),
        predictions_base=Path(args.predictions),
        horizon=args.horizon,
    )


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description='Weekly demand forecast pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = root.add_subparsers(dest='command', required=True)

    # ── train ──────────────────────────────────────────────────────────────────
    p_train = sub.add_parser('train', help='Train models and save backtest predictions')
    _add_common_args(p_train)
    p_train.add_argument('--trials', type=int, default=50,
                         help='Optuna trials for LightGBM (default: 50)')
    p_train.add_argument('--horizon', type=int, default=4,
                         help='Forecast horizon in weeks (default: 4)')

    # ── predict ────────────────────────────────────────────────────────────────
    p_pred = sub.add_parser('predict', help='Generate final forecast using saved artifacts')
    _add_common_args(p_pred)
    p_pred.add_argument('--horizon', type=int, default=4,
                        help='Forecast horizon in weeks (default: 4)')

    return root


if __name__ == '__main__':
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == 'train':
        _cmd_train(args)
    elif args.command == 'predict':
        _cmd_predict(args)

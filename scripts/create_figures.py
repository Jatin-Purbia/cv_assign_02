from src import runtime
runtime.configure()
import argparse
from src.visualization import plot_history, overlays, plot_comparison


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='kind', required=True)
    history = sub.add_parser('history')
    history.add_argument('--csv', required=True)
    history.add_argument('--out', required=True)
    compare = sub.add_parser('comparison')
    compare.add_argument('--csv', default='results/model_comparison.csv')
    compare.add_argument('--out', default='results/figures')
    overlay = sub.add_parser('overlays')
    overlay.add_argument('--weights', default='weights/final_model.pth')
    overlay.add_argument('--images', required=True)
    overlay.add_argument('--masks', required=True)
    overlay.add_argument('--manifest')
    overlay.add_argument('--out', required=True)
    args = p.parse_args()
    if args.kind == 'history':
        plot_history(args.csv, args.out)
    elif args.kind == 'comparison':
        plot_comparison(args.csv, args.out)
    else:
        overlays(args.weights, args.images, args.masks, args.out, args.manifest)


if __name__ == '__main__':
    main()

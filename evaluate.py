"""Evaluate validation, frozen DUTS-TE, or captured photographs."""
from src import runtime
runtime.configure()
import argparse
import json
from src.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default='weights/final_model.pth')
    parser.add_argument('--images', required=True)
    parser.add_argument('--masks', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--split', required=True, choices=['validation', 'duts-test', 'captured', 'synthetic'])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = evaluate(args.weights, args.images, args.masks, args.out, args.split, args.manifest)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_image'}, indent=2))


if __name__ == '__main__':
    main()

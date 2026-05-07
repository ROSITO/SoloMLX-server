import argparse
import uvicorn

from mlxserve.api.app import app
from mlxserve.config import settings
from mlxserve.models.manager import ModelManager
from mlxserve.runtime.autotune import run_autotune_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlxserve")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve")
    sub.add_parser("models-list")
    pull = sub.add_parser("models-pull")
    pull.add_argument("--model", required=True)
    rm = sub.add_parser("models-rm")
    rm.add_argument("--model", required=True)

    tune = sub.add_parser("autotune")
    tune.add_argument("--model", default="mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    tune.add_argument("--max-tokens", type=int, default=96)
    tune.add_argument(
        "--prompt",
        default="Donne 8 recommandations concretes pour eviter le swap lors d'un serving LLM local sur Apple Silicon.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in (None, "serve"):
        uvicorn.run(app, host=settings.host, port=settings.port)
        return

    if args.command == "autotune":
        print(run_autotune_json(model_id=args.model, prompt=args.prompt, max_tokens=args.max_tokens))
        return

    manager = ModelManager()
    if args.command == "models-list":
        for m in manager.list_local():
            print(f"{m.id}\t{m.source}\t{m.local_path}")
        return
    if args.command == "models-pull":
        m = manager.pull(args.model)
        print(f"pulled {m.source} as {m.id}")
        return
    if args.command == "models-rm":
        deleted = manager.remove(args.model)
        print("deleted" if deleted else "not-found")
        return

    parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    # Exposed for tests.
    return _build_parser()


if __name__ == "__main__":
    main()

from mlxserve.cli import build_parser


def test_cli_default_is_serve_when_no_command():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.command is None


def test_cli_autotune_parsing():
    parser = build_parser()
    args = parser.parse_args(["autotune", "--model", "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "--max-tokens", "128"])
    assert args.command == "autotune"
    assert args.model.endswith("Mistral-7B-Instruct-v0.3-4bit")
    assert args.max_tokens == 128

#!/usr/bin/env python3
"""
Minesweeper Eval Demo
=====================
This script shows how your agent will be called during the final evaluation.

During the competition, the controller will:
  1. Write a game state JSON to  inputs/game_state.json
  2. Your agent reads it, runs inference, and writes  outputs/action.json
  3. The controller reads your action and scores it

WHAT YOU CAN CHANGE:
  - Model path:  Update agents/minesweeper_model.py to point to YOUR model
  - Game state:  Provide your own via --game_state_file, or use the built-in sample

Usage:
    python demo_eval.py                                  # Use built-in sample game state
    python demo_eval.py --game_state_file my_state.json  # Use your own game state
"""

import json
import time
from pathlib import Path

from agents.minesweeper_agent import MinesweeperPlayer


# ─── Sample game state (used when no --game_state_file is provided) ───────────
SAMPLE_GAME_STATE = {
    "board": [
        ["1", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
    ],
    "rows": 8,
    "cols": 8,
    "mines": 10,
    "flags_placed": 0,
    "cells_revealed": 1,
}


def pretty_board(game_state: dict) -> str:
    """Return a formatted board string."""
    board = game_state["board"]
    cols = len(board[0])
    header = "     " + "  ".join(f"{i:2d}" for i in range(cols))
    sep = "    " + "─" * (cols * 4 + 1)
    lines = [header, sep]
    for r, row in enumerate(board):
        cells = "  ".join(f" {c}" for c in row)
        lines.append(f" {r:2d} │ {cells}")
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo: shows how your agent is called during evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--game_state_file", type=str, default=None,
        help="Path to a game state JSON file. If not provided, uses a built-in sample.",
    )
    args = parser.parse_args()

    # ── Setup directories ─────────────────────────────────────────────────
    inputs_dir = Path("inputs")
    outputs_dir = Path("outputs")
    inputs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    # ── Print banner ──────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║             MINESWEEPER AI — EVALUATION DEMO                       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # ── Load game state ───────────────────────────────────────────────────
    if args.game_state_file:
        print(f"Loading game state from: {args.game_state_file}")
        with open(args.game_state_file, "r") as f:
            game_state = json.load(f)
    else:
        print("Using built-in sample game state (8x8, 10 mines, 1 cell revealed)")
        game_state = SAMPLE_GAME_STATE

    # Save to inputs/ (this is what the controller does)
    state_file = inputs_dir / "game_state.json"
    with open(state_file, "w") as f:
        json.dump(game_state, f, indent=2)
    print(f"  Saved to: {state_file}")

    print(f"\nBoard ({game_state['rows']}x{game_state['cols']}, "
          f"{game_state['mines']} mines):\n")
    print(pretty_board(game_state))
    print()

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model...")
    player = MinesweeperPlayer()

    # Load generation config (max_new_tokens=128 is fixed)
    gen_kwargs = {}
    config_path = Path("minesweeper_config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            gen_kwargs.update(yaml.safe_load(f))
    gen_kwargs["tgps_show"] = True

    # Warmup inference (compile GPU kernels so timing is accurate)
    print("Warming up model...", end=" ", flush=True)
    warmup_state = {
        "board": [[".", "."], ["1", "."]],
        "rows": 2, "cols": 2, "mines": 1,
    }
    player.play_action(warmup_state, **gen_kwargs)
    print("done.")
    print("Model ready!\n")

    # ── Run inference ─────────────────────────────────────────────────────
    print("Running inference...")
    start = time.time()
    action, tokens, gen_time = player.play_action(game_state, **gen_kwargs)
    elapsed = time.time() - start

    # ── Write output ──────────────────────────────────────────────────────
    action_file = outputs_dir / "action.json"

    if action is None:
        print("\nFailed to parse a valid action from model output.")
        result = {"error": "parse_failed"}
        with open(action_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved error to: {action_file}")
    else:
        with open(action_file, "w") as f:
            json.dump(action, f, indent=2)

        print(f"\nAction: {json.dumps(action)}")
        print(f"  Saved to: {action_file}")

    # ── Stats ─────────────────────────────────────────────────────────────
    print(f"\nInference time: {elapsed:.2f}s")
    if tokens:
        print(f"Tokens generated: {tokens}")
        if gen_time and gen_time > 0:
            print(f"Throughput: {tokens / gen_time:.1f} tok/s")

    print()


if __name__ == "__main__":
    main()

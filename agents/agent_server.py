#!/usr/bin/env python3
"""
Persistent Minesweeper Agent Server
Keeps model loaded in memory and watches for new game states
"""
import json
import time
import yaml
from pathlib import Path
from typing import Optional
import signal
import sys

from .minesweeper_agent import MinesweeperPlayer


class AgentServer:
    """Persistent agent server that keeps model loaded"""

    def __init__(self, config_file: str = "minesweeper_config.yaml"):
        print("Starting Minesweeper Agent Server...")
        print("Loading model (this may take 30-60 seconds)...")

        # Load model ONCE at startup
        self.player = MinesweeperPlayer()

        # Load generation config
        self.gen_kwargs = {}
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, "r") as f:
                self.gen_kwargs.update(yaml.safe_load(f))

        print("Model loaded! Server ready to process actions.")
        print(f"Config: {self.gen_kwargs}")

        self.running = True
        self.inputs_dir = Path("inputs")
        self.outputs_dir = Path("outputs")

        # Ensure directories exist
        self.inputs_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)

        # Track last processed sequence number to avoid reprocessing
        self.last_mtime = 0
        self.last_sequence = -1
        self.current_round = -1  # Track current round to reset sequence

    def process_game_state(self, game_state_file: Path) -> tuple[Optional[dict], int]:
        """Process a game state and generate action"""
        try:
            # Read game state
            with open(game_state_file, "r") as f:
                game_state = json.load(f)

            # Get round and sequence number (controller includes these)
            round_num = game_state.get("_round", -1)
            sequence = game_state.get("_sequence", -1)

            # Check if we're in a new round - reset sequence counter
            if round_num != self.current_round:
                print(f"\nNEW ROUND {round_num} detected! Resetting sequence counter.")
                self.current_round = round_num
                self.last_sequence = -1  # Reset for new round

            # Skip if already processed (in current round)
            if sequence <= self.last_sequence:
                return None, sequence

            print(f"\nProcessing Round {round_num}, Sequence {sequence}: {game_state_file.name}")

            # Generate action (model already loaded!)
            action, tl, gt = self.player.play_action(game_state, **self.gen_kwargs)

            if action:
                print(f"Action: {action}")
                # Add metadata for controller (sequence, round, timing)
                action["_sequence"] = sequence
                action["_round"] = round_num
                if tl and gt:
                    action["_inference_time"] = gt
                    action["_tokens_generated"] = tl
                    print(f"⚡ Stats: {tl} tokens in {gt:.2f}s ({tl/gt:.1f} tok/s)")
                return action, sequence
            else:
                print("Failed to generate valid action")
                return {"error": "parse_failed", "_sequence": sequence, "_round": round_num}, sequence

        except Exception as e:
            print(f"Error processing game state: {e}")
            return {"error": "processing_failed", "message": str(e)}, -1

    def watch_for_game_states(self):
        """Watch inputs directory for new game states"""
        game_state_file = self.inputs_dir / "game_state.json"
        action_file = self.outputs_dir / "action.json"

        print(f"\n👀 Watching {game_state_file} for changes...")
        print("Press Ctrl+C to stop\n")

        while self.running:
            try:
                if game_state_file.exists():
                    current_mtime = game_state_file.stat().st_mtime

                    # New file or file was updated
                    if current_mtime > self.last_mtime:
                        self.last_mtime = current_mtime

                        # Process the game state
                        action, sequence = self.process_game_state(game_state_file)

                        # Write action (only if it's new)
                        if action and sequence > self.last_sequence:
                            self.last_sequence = sequence

                            # Atomic write: write to temp file first, then rename
                            # This prevents controller from reading partial JSON
                            temp_file = action_file.with_suffix('.tmp')
                            with open(temp_file, "w") as f:
                                json.dump(action, f, indent=2)

                            # Atomic rename (replaces existing file)
                            temp_file.replace(action_file)
                            print(f"💾 Saved to: {action_file}")

                # Check every 100ms
                time.sleep(0.1)

            except KeyboardInterrupt:
                print("\nShutting down server...")
                self.running = False
                break
            except Exception as e:
                print(f"Error in watch loop: {e}")
                time.sleep(1)

    def stop(self):
        """Stop the server"""
        self.running = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived shutdown signal...")
    sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Persistent Minesweeper Agent Server")
    parser.add_argument(
        "--config",
        type=str,
        default="minesweeper_config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server
    server = AgentServer(config_file=args.config)
    server.watch_for_game_states()

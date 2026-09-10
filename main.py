"""Hybrid-Brain â€” entry point.

Week 1: only load config and show system health.
Later weeks: start the brain-environment loop.
"""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "brain_config.yaml"


def load_config() -> dict:
    """Load configuration â€” no parameter is defined outside this file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    config = load_config()
    neuron = config["neuron"]
    print("í·  Hybrid-Brain v0.0.0-arch")
    print(f"   Neuron model: {neuron['model']}")
    print(f"   Threshold: {neuron['v_threshold_mV']} mV | Rest: {neuron['v_rest_mV']} mV")
    print(f"   Ï„_m: {neuron['tau_m_ms']} ms | dt: {config['simulation']['dt_ms']} ms")
    print("   âœ… Architecture contract loaded â€” ready for Week 2 (neurons)")


if __name__ == "__main__":
    main()

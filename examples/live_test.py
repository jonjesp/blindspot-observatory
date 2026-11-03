"""One paid decision, enabled explicitly with ``python examples/live_test.py --live``."""

import argparse

from blindspot.agent.client import make_live_client
from blindspot.agent.muse import MuseAgent
from blindspot.agent.schemas import TOOLS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True, help="authorize one paid request")
    parser.parse_args(argv)
    obs = {
        "pilot_measurements": {
            "sources_per_batch": 25,
            "recovery_counts": {"nominal": {"A": [18, 17, 19, 16], "B": [6, 7, 5, 8]}},
        },
        "indicators": {
            "background_mad_change": 0.27,
            "integrated_amplitude_change": 0.05,
            "rms_radius_change": 0.04,
        },
        "available_probes": list(TOOLS),
    }
    agent = MuseAgent(ledger_path="evidence/runs/live_ledger.json", max_calls=1, budget_usd=0.05)
    r = agent.decide("TEST01", obs, client=make_live_client(), allow_paid=True)
    print(r["state"], r.get("decision"), r["finish_reason"], r["cost_usd"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

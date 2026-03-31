import json
from pathlib import Path

new_thresholds = {
    "node_thresholds": {
        "population": {"mode": "relative_pct", "alert_pct_increase": 0.10},
        "rainfall": {"mode": "absolute", "threshold": 150},
        "electricity_demand": {"mode": "relative_pct", "alert_pct_increase": 0.12},
        "water_demand": {"mode": "relative_pct", "alert_pct_increase": 0.18},
        "congestion_index": {"mode": "absolute", "threshold": 0.50},
        "pm25": {"mode": "absolute", "threshold": 80}
    }
}

out = Path("cascade_model/artifacts/thresholds/thresholds.json")
out.write_text(json.dumps(new_thresholds, indent=2))
print("[UPDATED] thresholds.json written.")

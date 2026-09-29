"""
Enterprise Prometheus & APM Metrics Instrumentation Module.
Exposes standard Prometheus metrics format for Grafana dashboards and alerting.
Tracks throughput, prediction tiers, request latency histograms, and CTI feedback.
"""

import time
from typing import Dict


class MetricsRegistry:
    def __init__(self):
        self.request_count: int = 0
        self.predictions_by_tier: Dict[str, int] = {
            "tier_1_hot": 0,
            "tier_2_warm": 0,
            "tier_3_neutral": 0,
            "tier_4_cold": 0,
        }
        self.predictions_by_mode: Dict[str, int] = {
            "pre_call": 0,
            "post_call": 0,
        }
        self.feedback_count: Dict[str, int] = {
            "agreed": 0,
            "declined": 0,
            "callback": 0,
            "unreachable": 0,
        }
        self.latency_buckets = {
            0.01: 0,
            0.05: 0,
            0.1: 0,
            0.25: 0,
            0.5: 0,
            1.0: 0,
            2.5: 0,
        }
        self.total_latency_sec: float = 0.0
        self.drift_events_total: int = 0

    def record_request(self):
        self.request_count += 1

    def record_prediction(self, tier: str, mode: str, duration_sec: float):
        self.request_count += 1
        self.total_latency_sec += duration_sec

        # Record tier
        tier_key = tier.lower().replace(" ", "_")
        for k in self.predictions_by_tier:
            if k.split("_")[1] in tier_key:
                self.predictions_by_tier[k] += 1
                break

        # Record mode
        if mode in self.predictions_by_mode:
            self.predictions_by_mode[mode] += 1

        # Record latency histogram
        for b in sorted(self.latency_buckets.keys()):
            if duration_sec <= b:
                self.latency_buckets[b] += 1

    def record_feedback(self, outcome: str):
        key = outcome.lower()
        if key in self.feedback_count:
            self.feedback_count[key] += 1
        else:
            self.feedback_count["declined"] += 1

    def record_drift_event(self):
        self.drift_events_total += 1

    def to_prometheus_format(self) -> str:
        """Render metrics in standard Prometheus exposition text format."""
        lines = [
            "# HELP bank_marketing_http_requests_total Total number of HTTP requests",
            "# TYPE bank_marketing_http_requests_total counter",
            f"bank_marketing_http_requests_total {self.request_count}",
            "",
            "# HELP bank_marketing_predictions_total Total predictions by tier and mode",
            "# TYPE bank_marketing_predictions_total counter",
        ]
        for tier, count in self.predictions_by_tier.items():
            lines.append(f'bank_marketing_predictions_total{{tier="{tier}"}} {count}')
        for mode, count in self.predictions_by_mode.items():
            lines.append(f'bank_marketing_predictions_total{{mode="{mode}"}} {count}')

        lines.extend([
            "",
            "# HELP bank_marketing_feedback_events_total CTI Feedback from call center by outcome",
            "# TYPE bank_marketing_feedback_events_total counter",
        ])
        for outcome, count in self.feedback_count.items():
            lines.append(f'bank_marketing_feedback_events_total{{outcome="{outcome}"}} {count}')

        lines.extend([
            "",
            "# HELP bank_marketing_latency_seconds Latency histogram in seconds",
            "# TYPE bank_marketing_latency_seconds histogram",
        ])
        cumulative = 0
        for b, count in sorted(self.latency_buckets.items()):
            cumulative += count
            lines.append(f'bank_marketing_latency_seconds_bucket{{le="{b}"}} {cumulative}')
        lines.append(f'bank_marketing_latency_seconds_bucket{{le="+Inf"}} {self.request_count}')
        lines.append(f"bank_marketing_latency_seconds_sum {round(self.total_latency_sec, 4)}")
        lines.append(f"bank_marketing_latency_seconds_count {self.request_count}")

        lines.extend([
            "",
            "# HELP bank_marketing_drift_events_total Total detected distribution drift events",
            "# TYPE bank_marketing_drift_events_total counter",
            f"bank_marketing_drift_events_total {self.drift_events_total}",
        ])

        return "\n".join(lines) + "\n"


# Global singleton metrics instance
metrics = MetricsRegistry()

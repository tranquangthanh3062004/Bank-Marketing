"""
Locust Performance & Stress Load Testing Benchmark.
Simulates 500 - 2,000 concurrent Telesales agents and Call Center CTI integrations.
Measures latency SLA (< 50ms) under heavy traffic.
Usage:
    locust -f benchmarks/locustfile.py --headless -u 100 -r 10 --run-time 1m --host http://localhost:8000
"""

import random
from locust import HttpUser, between, task


class TelesalesAgentUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.headers = {
            "X-API-Key": "bm-enterprise-secret-key-2026",
            "Content-Type": "application/json",
        }

    def _sample_payload(self, is_post_call=False):
        duration = random.randint(30, 800) if is_post_call else 0
        return {
            "customer_id": f"LEAD_{random.randint(1, 10000):05d}",
            "age": random.randint(18, 75),
            "job": random.choice(["technician", "management", "blue-collar", "admin.", "retired"]),
            "marital": random.choice(["married", "single", "divorced"]),
            "education": random.choice(["secondary", "tertiary", "primary"]),
            "default": "no",
            "balance": float(random.randint(-200, 15000)),
            "housing": random.choice(["yes", "no"]),
            "loan": random.choice(["yes", "no"]),
            "contact": "cellular",
            "day": random.randint(1, 28),
            "month": random.choice(["may", "aug", "jul", "nov", "jun"]),
            "duration": duration,
            "campaign": random.randint(1, 4),
            "pdays": -1 if random.random() < 0.8 else random.randint(1, 300),
            "previous": 0 if random.random() < 0.8 else random.randint(1, 5),
            "poutcome": "unknown" if random.random() < 0.8 else random.choice(["success", "failure"]),
        }

    @task(7)
    def test_pre_call_prediction(self):
        payload = self._sample_payload(is_post_call=False)
        self.client.post("/api/v1/predict/pre-call", json=payload, headers=self.headers)

    @task(2)
    def test_post_call_prediction(self):
        payload = self._sample_payload(is_post_call=True)
        self.client.post("/api/v1/predict/post-call", json=payload, headers=self.headers)

    @task(1)
    def test_submit_feedback(self):
        feedback = {
            "customer_id": f"LEAD_{random.randint(1, 10000):05d}",
            "agent_id": f"AGENT_{random.randint(1, 50):03d}",
            "call_duration": random.randint(60, 450),
            "outcome": random.choice(["agreed", "declined", "callback", "unreachable"]),
            "product_type": "term_deposit",
            "notes": "Automated CTI agent benchmark log",
        }
        self.client.post("/api/v1/telemarketing/feedback", json=feedback, headers=self.headers)

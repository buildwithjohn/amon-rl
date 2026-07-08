"""
clouds.py — Calibrated three-provider cloud model for the AMON environment.

AMON-RL thesis. Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26).

Models AWS, Azure, and GCP as three regions with:
  - inter-cloud propagation latency (fixed, from published measurements)
  - per-vCPU-hour compute price (from public on-demand pricing)
  - per-GB egress price (from public data-transfer pricing)
  - a compute capacity budget per cloud

Calibration sources (values are representative, not exact, and are stated
here so the thesis can cite provenance and acknowledge them as modelling
assumptions rather than live measurements):

  Inter-cloud propagation latency (ms), same-geography cross-provider:
    Kentik Cloud Latency Map (Oct 2024) reports Azure->AWS ~131ms and
    GCP->AWS ~134ms for same-metro cross-provider routes; intra-cloud
    and same-metro same-provider RTTs are single- to low-double-digit ms
    (Azure Network Latency docs; GCP Performance Dashboard). We model the
    three regions as co-located in Western Europe (eu-west-1 / westeurope /
    europe-west1), so intra-cloud latency is low and cross-cloud is the
    ~5-15ms range reported for same-metro hyperscaler interconnect
    (Aviatrix region-affinity tables report as low as ~2ms; we use
    conservative low-double-digit values to leave headroom for the
    time-varying queue and processing terms of the latency model).

  Compute price (USD per vCPU-hour, on-demand, general purpose, EU-West):
    Derived from published on-demand rates for m5.large-class instances;
    AWS and Azure price closely, GCP marginally lower. These are order-of-
    -magnitude correct for the relative comparison the agent must learn,
    which is what matters for the reward signal, not absolute billing.

  Egress price (USD per GB, cross-cloud / internet egress, first tier):
    All three price internet egress in the ~0.08-0.12 USD/GB band for the
    first tier in Europe. Intra-cloud same-region egress is ~0.

These are deliberately simple, transparent constants. Chapter 4 records
them as calibration assumptions and Chapter 6 (future work) notes that
live-measured, time-varying pricing and latency are a substitution point.
"""

import numpy as np

# Canonical cloud ordering used everywhere in the codebase.
CLOUDS = ("aws", "azure", "gcp")
N_CLOUDS = len(CLOUDS)
AWS, AZURE, GCP = 0, 1, 2

# Modelled region per cloud (Western Europe, co-located metros).
REGIONS = {"aws": "eu-west-1", "azure": "westeurope", "gcp": "europe-west1"}

# ---------------------------------------------------------------------------
# Inter-cloud propagation latency matrix L_prop[i][j] in milliseconds.
# Diagonal = intra-cloud (same region). Off-diagonal = same-metro cross-cloud.
# Symmetric by construction here; real routes are mildly asymmetric, noted
# as a simplification.
# ---------------------------------------------------------------------------
L_PROP = np.array([
    #   aws   azure  gcp
    [   2.0,   9.0,  11.0],   # from aws
    [   9.0,   2.0,  10.0],   # from azure
    [  11.0,  10.0,   2.0],   # from gcp
], dtype=np.float32)

# ---------------------------------------------------------------------------
# Compute price, USD per vCPU-hour, on-demand general purpose, EU-West.
# ---------------------------------------------------------------------------
COMPUTE_PRICE = np.array([0.096, 0.096, 0.089], dtype=np.float32)  # aws, azure, gcp

# ---------------------------------------------------------------------------
# Egress price, USD per GB. Cross-cloud egress leaves the source provider,
# so it is charged at the SOURCE cloud's internet-egress rate. Intra-cloud
# same-region traffic is modelled as free.
# ---------------------------------------------------------------------------
EGRESS_PRICE = np.array([0.090, 0.087, 0.120], dtype=np.float32)  # aws, azure, gcp

# ---------------------------------------------------------------------------
# Compute capacity per cloud, in vCPUs available to the workload. Calibrated
# (Chapter 4) so that concentrating all services on one cloud OVERLOADS it at
# peak traffic (single-cloud peak utilisation ~1.8), while a good spread
# across all three stays feasible (total capacity ~74 vs peak demand ~46).
# This is what makes placement a non-trivial control problem: the naive
# all-on-one-cloud policy breaches SLA under load, so the agent must learn to
# distribute. Values are deliberately asymmetric so the clouds are not
# interchangeable.
# ---------------------------------------------------------------------------
CAPACITY = np.array([26.0, 22.0, 26.0], dtype=np.float32)  # aws, azure, gcp


def egress_cost(src_cloud, dst_cloud, gb):
    """USD to move `gb` gigabytes from src_cloud to dst_cloud."""
    if src_cloud == dst_cloud:
        return 0.0
    return float(EGRESS_PRICE[src_cloud] * gb)


def prop_latency(src_cloud, dst_cloud):
    """Fixed propagation component L_prop_ij in ms."""
    return float(L_PROP[src_cloud, dst_cloud])


def compute_cost(cloud, vcpu_hours):
    """USD for `vcpu_hours` vCPU-hours on `cloud`."""
    return float(COMPUTE_PRICE[cloud] * vcpu_hours)


if __name__ == "__main__":
    print("AMON cloud model")
    print("clouds:", CLOUDS)
    print("regions:", REGIONS)
    print("\nL_prop (ms):\n", L_PROP)
    print("\ncompute (USD/vCPU-hr):", COMPUTE_PRICE)
    print("egress  (USD/GB):      ", EGRESS_PRICE)
    print("capacity (vCPU):       ", CAPACITY)
    print("\nsanity: aws->gcp 5GB egress =",
          f"${egress_cost(AWS, GCP, 5):.3f}",
          "| aws->gcp prop =", f"{prop_latency(AWS, GCP):.0f}ms",
          "| intra-aws egress =", f"${egress_cost(AWS, AWS, 5):.3f}")

"""
topology.py — Online-Boutique-style microservice dependency graph.

AMON-RL thesis. Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26).

The proposal's evaluation topology is Google's Online Boutique, a canonical
cloud-native microservices demo of ~10 interdependent services. This module
encodes that graph, each service's baseline resource demand, the call graph
that generates inter-service (and therefore potentially inter-cloud) traffic,
and the NDPR data-residency flag per service.

Services (index : name):
   0 frontend            entry point, fans out to most services
   1 cartservice         holds cart state (personal data)
   2 checkoutservice     orchestrates the purchase
   3 currencyservice     stateless currency conversion
   4 emailservice        sends order confirmation (personal data)
   5 paymentservice      charges the card (personal data)
   6 productcatalog      product listings
   7 recommendation      suggested products
   8 shippingservice     shipping quote + order
   9 redis               cart backing store (personal data)

Directed edges are caller -> callee, weighted by the average request rate
along that dependency (calls per user request). Traffic on an edge that
crosses a cloud boundary incurs egress cost and inter-cloud latency.

NDPR flag: services that process or store personal data of end users. Under
Nigeria's NDPR / EU GDPR framing in the proposal, these must be placed only
in approved regions. In the modelled multi-cloud, the agent must keep these
services on approved clouds; the environment enforces this via action
masking (see amon_env.py).
"""

import numpy as np

N_SERVICES = 10

SERVICE_NAMES = (
    "frontend", "cartservice", "checkoutservice", "currencyservice",
    "emailservice", "paymentservice", "productcatalog", "recommendation",
    "shippingservice", "redis",
)

# Directed dependency edges (caller, callee, calls_per_request).
# Calls-per-request approximates the Online Boutique call graph: the
# frontend hits catalog/recommendation/currency on every page load, and
# the checkout path fans out to cart, shipping, payment, email, currency.
DEPENDENCIES = [
    (0, 6, 2.0),   # frontend -> productcatalog (browse, heavy)
    (0, 7, 1.0),   # frontend -> recommendation
    (0, 3, 1.5),   # frontend -> currency (price display)
    (0, 1, 0.8),   # frontend -> cart (view cart)
    (0, 2, 0.3),   # frontend -> checkout (a fraction of sessions buy)
    (7, 6, 1.0),   # recommendation -> productcatalog
    (2, 1, 1.0),   # checkout -> cart (read items)
    (2, 3, 1.0),   # checkout -> currency
    (2, 5, 1.0),   # checkout -> payment
    (2, 8, 1.0),   # checkout -> shipping
    (2, 4, 1.0),   # checkout -> email
    (1, 9, 2.0),   # cart -> redis (read+write, heavy)
]

# Baseline resource demand per service at unit load (1.0 traffic scale):
# vCPUs required, and average payload size per call in MB (drives egress GB).
# redis and catalog are memory/IO heavy; payment/email are light.
BASE_VCPU = np.array(
    [3.0, 2.0, 2.5, 1.0, 0.8, 1.2, 3.0, 2.0, 1.5, 2.5], dtype=np.float32)

PAYLOAD_MB = np.array(
    [0.05, 0.08, 0.10, 0.02, 0.03, 0.04, 0.20, 0.15, 0.06, 0.12],
    dtype=np.float32)

# Intrinsic per-call processing time (ms) at the destination service, the
# L_proc term's baseline before load-dependent inflation.
BASE_PROC_MS = np.array(
    [3.0, 4.0, 6.0, 2.0, 5.0, 8.0, 4.0, 5.0, 4.0, 1.5], dtype=np.float32)

# NDPR: services touching end-user personal data.
NDPR_PERSONAL_DATA = np.array(
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
#    fe cart co cur em  pay ca re sh redis


def adjacency():
    """Unweighted directed adjacency (N, N), a[caller, callee] = 1."""
    a = np.zeros((N_SERVICES, N_SERVICES), dtype=np.float32)
    for u, v, _ in DEPENDENCIES:
        a[u, v] = 1.0
    return a


def edge_index():
    """(2, E) directed edge index for the GAT, caller -> callee.

    Message passing in the GAT flows callee -> caller and caller -> callee
    both, so we return edges in both directions (a service's state depends
    on both what it calls and what calls it).
    """
    fwd = [(u, v) for u, v, _ in DEPENDENCIES]
    both = fwd + [(v, u) for u, v in fwd]
    return np.array(both, dtype=np.int64).T


def call_rates():
    """(N, N) matrix of calls-per-request, rate[caller, callee]."""
    r = np.zeros((N_SERVICES, N_SERVICES), dtype=np.float32)
    for u, v, w in DEPENDENCIES:
        r[u, v] = w
    return r


if __name__ == "__main__":
    print("AMON service topology")
    print(f"{N_SERVICES} services, {len(DEPENDENCIES)} dependencies")
    print("NDPR-restricted:",
          [SERVICE_NAMES[i] for i in range(N_SERVICES)
           if NDPR_PERSONAL_DATA[i]])
    print("total base vCPU demand:", BASE_VCPU.sum(),
          "(vs per-cloud capacity 40-48)")
    r = call_rates()
    print("busiest dependency edges (calls/request):")
    idx = np.dstack(np.unravel_index(np.argsort(r.ravel())[::-1],
                                     r.shape))[0]
    for u, v in idx[:4]:
        print(f"  {SERVICE_NAMES[u]} -> {SERVICE_NAMES[v]}: {r[u, v]:.1f}")

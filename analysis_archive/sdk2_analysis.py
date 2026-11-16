"""Cumulative-reward analysis expressed as one matrix exponential for SDK2.

Representation
--------------
For a revision with ``n`` states and ``k`` metrics, the joint process is the
state X(t) plus the cumulative reward vector Z(t).  We propagate the tuple of
conditional moments

    y[i, alpha](t) = E[ 1{X(t)=i} * Z(t)**alpha ]

for every multi-index ``alpha`` of total degree at most two:
the constant 1, the ``k`` first powers Z_m, and the ``k(k+1)/2`` products
Z_m*Z_p (m <= p).  A vector of dimension ``n * (1 + k + k(k+1)/2)`` therefore
carries state probabilities, all means and all raw second moments at once.

The generator acts on a monomial as

* drift:     r_i . grad Z**alpha  = sum_m alpha_m * r[i][m] * Z**(alpha-e_m)
* each event e from i to j at rate lambda with payment g:
             lambda * (Z+g)**alpha = lambda * sum_{beta<=alpha}
             prod_m C(alpha_m,beta_m) g_m**(alpha_m-beta_m) * Z**beta

Both are linear in the moments, so assembling one matrix ``A`` gives
``y(t) = exp(t*A) @ y(0)`` exactly through the sole SDK2 port, with column
vector convention.  Payments of one event share one jump, so their covariance
is produced by the same ``(Z+g)**alpha`` expansion; parallel and self events
are separate jump operators and keep their own identities.
"""
import math
from fractions import Fraction

from vendor.sdk2 import propagate


def _monomials(k):
    result = [(0,) * k]
    for m in range(k):
        power = [0] * k
        power[m] = 1
        result.append(tuple(power))
    for i in range(k):
        for j in range(i, k):
            power = [0] * k
            power[i] += 1
            power[j] += 1
            result.append(tuple(power))
    return result


def _coefficient(alpha, beta, payment):
    value = Fraction(1)
    for m in range(len(alpha)):
        if beta[m]:
            value *= math.comb(alpha[m], beta[m])
        difference = alpha[m] - beta[m]
        if difference:
            value *= payment[m] ** difference
    return value


def _prepare(definition):
    states = definition["states"]
    metrics = definition["metrics"]
    n, k = len(states), len(metrics)
    index = {s["id"]: i for i, s in enumerate(states)}
    rewards = [[Fraction(x) for x in s["reward"]] for s in states]
    totals = [Fraction(0)] * n
    events = []
    for event in definition["events"]:
        source, target = index[event["source"]], index[event["target"]]
        rate = Fraction(event["rate"])
        payment = [Fraction(x) for x in event["reward"]]
        totals[source] += rate
        events.append((source, target, rate, payment))
    return n, k, rewards, totals, events


def _generator(definition):
    n, k, rewards, totals, events = _prepare(definition)
    monomials = _monomials(k)
    position = {power: i for i, power in enumerate(monomials)}
    size = len(monomials)
    dimension = n * size
    matrix = [[Fraction(0)] * dimension for _ in range(dimension)]

    def row(state, alpha):
        return state * size + position[alpha]

    for state in range(n):
        for alpha in monomials:
            current = row(state, alpha)
            matrix[current][current] -= totals[state]
            for m in range(k):
                if alpha[m]:
                    lower = list(alpha)
                    lower[m] -= 1
                    matrix[current][row(state, tuple(lower))] += alpha[m] * rewards[state][m]
    for source, target, rate, payment in events:
        for alpha in monomials:
            for beta in monomials:
                if any(beta[m] > alpha[m] for m in range(k)):
                    continue
                coefficient = _coefficient(alpha, beta, payment)
                if coefficient:
                    matrix[row(target, alpha)][row(source, beta)] += rate * coefficient
    return [[float(value) for value in line] for line in matrix], monomials


def from_representation(representation):
    """Rebuild the canonical computation definition from a target revision record.

    Target rows persist this form next to the canonical exported definition; the
    analysis consumes the persisted form directly so the audit tables cannot
    influence a computation.
    """
    return {"metrics": list(representation["metrics"]),
            "states": [{"id": state, "reward": [str(x) for x in row]}
                       for state, row in zip(representation["states"], representation["residence"])],
            "events": [{"id": event["id"], "source": event["source"], "target": event["target"],
                        "rate": str(event["rate"]),
                        "reward": [str(x) for x in event["payment"]]}
                       for event in representation["events"]]}


def analyze(definition, initial, duration):
    """Return the canonical statistics document using SDK2 only."""
    metrics = list(definition["metrics"])
    states = list(definition["states"])
    k = len(metrics)
    matrix, monomials = _generator(definition)
    size = len(monomials)
    start = []
    for state_index, state in enumerate(states):
        probability = float(Fraction(initial.get(state["id"], 0)))
        for power, alpha in enumerate(monomials):
            start.append(probability if not any(alpha) else 0.0)
    moments = propagate(matrix, start, float(duration))
    count = len(states)
    probabilities = [moments[i * size] for i in range(count)]
    columns = [i for i, alpha in enumerate(monomials) if sum(alpha) == 1]
    means = [math.fsum(moments[i * size + column] for i in range(count))
             for column in columns]
    pairs = [(i, j) for i in range(k) for j in range(i, k)]
    raw = [[0.0] * k for _ in range(k)]
    for (i, j) in pairs:
        power = [0] * k
        power[i] += 1
        power[j] += 1
        column = monomials.index(tuple(power))
        value = math.fsum(moments[s * size + column] for s in range(count))
        raw[i][j] = raw[j][i] = value
    covariance = [[raw[i][j] - means[i] * means[j] for j in range(k)] for i in range(k)]
    return {"metrics": metrics, "states": [s["id"] for s in states],
            "probabilities": probabilities, "means": means, "covariance": covariance}

"""Non-mutating weight-gradient instrumentation and a bounded allocation proxy.

V_l = sum_i (d loss / d w_i)^2 span_i^2 p_i(1-p_i).
For independent module-use draws, the first-order loss-variance proxy is
sum_l V_l/T_l. These are declared STE gradients of the training surrogate;
this proxy omits curvature/quantization discontinuities and must be validated.
"""
import math
import types
import torch
from .recomputed_probability import programmed_probability


def instrument(encoders, aliases, lookup_modules, callback):
    originals = {}
    for canonical, encoder in encoders.items():
        uses = {name for name, target in aliases.items() if target == canonical}
        base = encoder.weights
        originals[canonical] = base
        def wrapped(self, samples=0, generator=None, rows=None, counts=None,
                    base=base, canonical=canonical, uses=uses):
            assert samples == 0 and counts is None
            output = base(samples, generator, rows=rows, counts=counts)
            owners = uses & lookup_modules if rows is not None else uses - lookup_modules
            assert len(owners) == 1
            owner = next(iter(owners))
            if output.requires_grad:
                output.register_hook(lambda gradient: callback(owner, canonical, self, rows, gradient))
            return output
        encoder.weights = types.MethodType(wrapped, encoder)
    return originals


@torch.no_grad()
def variance_coefficient(encoder, rows, gradient):
    assert gradient.dtype == torch.float64
    theta = encoder.theta if rows is None else encoder.theta.index_select(0, rows)
    span = encoder.span if rows is None else encoder.span.index_select(0, rows)
    vector = theta.ndim == 1
    count = 1 if vector else theta.shape[0]
    width = theta.numel() if vector else math.prod(theta.shape[1:])
    size = min(128, width)
    variance = squared_gradient = 0.
    for start in range(0, count, 128):
        end = min(start + 128, count)
        h = theta if vector else theta[start:end]
        g = gradient if vector else gradient[start:end]
        assert torch.isfinite(g).all()
        shape = (end - start, width // size, size)
        p = programmed_probability(h, shape)
        norm = g.reshape(shape).square()
        variance += float((norm * span[start:end].double().square() * p * (1 - p)).sum())
        squared_gradient += float(norm.sum())
    return variance, squared_gradient


def allocate(scores, costs, fixed_module, reference=1024, minimum=16, maximum=4096):
    assert set(scores) == set(costs) and fixed_module in scores
    assert minimum <= reference <= maximum
    assert all(v >= 0 and math.isfinite(v) for v in scores.values())
    assert all(isinstance(v, int) and v > 0 for v in costs.values())
    names = sorted(scores)
    cap = reference * sum(costs.values())
    counts = {name: reference if name == fixed_module else minimum for name in names}
    used = sum(costs[name] * counts[name] for name in names)
    while True:
        feasible = []
        for index, name in enumerate(names):
            old = counts[name]
            if name == fixed_module or old >= maximum or scores[name] <= 0:
                continue
            extra = costs[name] * old
            if used + extra <= cap:
                benefit = scores[name] / (2 * old)
                feasible.append((benefit / extra, -index, name, extra))
        if not feasible:
            break
        _, _, name, extra = max(feasible)
        counts[name] *= 2
        used += extra
    uniform_proxy = sum(scores[name] / reference for name in names)
    greedy_proxy = sum(scores[name] / counts[name] for name in names)
    selected = counts if greedy_proxy <= uniform_proxy else {name: reference for name in names}
    return {'greedy_samples': counts, 'greedy_cost': used, 'cost_cap': cap,
            'unused_cap': cap - used, 'unused_fraction': (cap - used) / cap,
            'uniform_proxy': uniform_proxy, 'greedy_proxy': greedy_proxy,
            'selected_samples': selected, 'selected_policy': 'greedy' if greedy_proxy <= uniform_proxy else 'uniform',
            'selected_cost': sum(costs[name] * selected[name] for name in names),
            'selected_proxy': min(greedy_proxy, uniform_proxy),
            'scope': 'Greedy doubling over16..4096samples, embedding fixed1024, same upper node-work cap. Not exact equality or globally optimal knapsack allocation. Everyunusednodeevaluationreported. First-order STE loss-variance proxy, not expected taskloss or a validated quality gain.'}

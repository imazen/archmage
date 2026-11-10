"""Independent exact-rational models for forced fused/separate study graphs."""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'xtask'))
import powf_vectors as fp


def madd(a, b, c, fused):
    return fp.fma(a, b, c) if fused else fp.add(fp.mul(a, b), c)


def horner(x, coefficients, fused):
    result = fp.bits(coefficients[-1])
    for c in reversed(coefficients[:-1]):
        result = madd(result, x, fp.bits(c), fused)
    return result


def preflight(base, exponent, domain):
    magnitude = exponent & 0x7fffffff
    if magnitude >= fp.INF or (magnitude and magnitude < 0x00800000):
        return 1
    x, n = fp.value(base), fp.value(exponent)
    if domain == 2 and not 0 <= n <= 4:
        return 1
    if domain == 2 and base != 0 and not 2**-16 <= x <= 2**16:
        return 2
    if base & 0x80000000 or base & fp.INF == fp.INF or base < 0x00800000:
        if domain != 0 and base == 0 and n >= 0:
            return 0
        return 2
    return 0


def argument(base, exponent, lowp, fused):
    if lowp:
        delta = base - 0x3f2aaaab
        whole = delta >> 23
        mantissa = base - (whole << 23)
        m = fp.sub(mantissa, fp.ONE)
        numerator = horner(m, [-1.8503833e-6, 1.4287161, .7424587], fused)
        denominator = horner(m, [.99032814, 1.0096718, .17409343], fused)
        log = fp.add(fp.divide(numerator, denominator), fp.bits(float(whole)))
    else:
        adjusted = (base + fp.ONE - 0x3f3504f3) & 0xffffffff
        signed = adjusted - 0x100000000 if adjusted & 0x80000000 else adjusted
        whole = (signed >> 23) - 127
        mantissa = (adjusted & 0x7fffff) + 0x3f3504f3
        y = fp.divide(fp.sub(mantissa, fp.ONE), fp.add(mantissa, fp.ONE))
        p = horner(fp.mul(y, y), [2.88539, .96180076, .57697445, .43441197], fused)
        log = madd(p, y, fp.bits(float(whole)), fused)
    return fp.mul(log, exponent)


def mirror(base, exponent, domain, lowp, fused):
    error = preflight(base, exponent, domain)
    if error:
        return 0, error
    if base == 0:
        return (fp.ONE if fp.value(exponent) == 0 else 0), 0
    t = argument(base, exponent, lowp, fused)
    if domain != 2 and not -126 <= fp.value(t) <= 127.5:
        return 0, 3
    if lowp:
        xi = math.floor(fp.value(t))
        p = horner(fp.sub(t, fp.bits(float(xi))), [1, math.log(2), .2402265, .05550411], fused)
    else:
        xi = min(round(fp.value(t)), 127)
        p = horner(fp.sub(t, fp.bits(float(xi))), [1, math.log(2), .24022646, .055504545, .009618055, .00133337, .00015447], fused)
    return fp.mul(p, (xi + 127) << 23), 0


def gamma_certificates():
    """Monotone RNE interval bounds; include each rounding path explicitly."""
    results = []
    for fused in [False, True]:
        for lowp in [False, True]:
            if lowp:
                m = fp.bits(1/3)  # rounded upward relative to the exact 1/3
                assert fp.value(m) > 1/3
                q_high = madd(fp.bits(.17409343), m, fp.bits(1.0096718), fused)
                q_low = madd(q_high, m ^ 0x80000000, fp.bits(.99032814), fused)
                assert fp.value(q_low) > .6
                p_high = madd(fp.bits(.7424587), m, fp.bits(1.4287161), fused)
                p_high = madd(p_high, m, fp.bits(1.8503833e-6), fused)
                log_bound = fp.add(fp.divide(p_high, q_low), fp.bits(16.))
                for e in [-16, 16]:
                    assert (fp.bits(2.**e) - 0x3f2aaaab) >> 23 == e
            else:
                low, high = 0x3f3504f3, 0x3fb504f2
                num = max(abs(fp.value(fp.sub(low, fp.ONE))), abs(fp.value(fp.sub(high, fp.ONE))))
                y = fp.divide(fp.bits(num), fp.add(low, fp.ONE))
                assert fp.value(y) < .25
                p = horner(fp.mul(y, y), [2.88539, .96180076, .57697445, .43441197], fused)
                log_bound = madd(p, y, fp.bits(16.), fused)
            bound = fp.mul(log_bound, fp.bits(4.))
            assert fp.value(log_bound) < 17 and fp.value(bound) < 68
            results.append(dict(lowp=lowp, fused=fused, log_bound=fp.value(log_bound), argument_bound=fp.value(bound)))
    return results

"""Exact certificates for a homogeneous multiclass LP on fixed FP32 features.

No solver, tensor library, model, filesystem access, or backend dot product is
used here. Call only in the registered CPU job. Coefficient-bearing ``witness``
results belong under CKPT; the caller owns publication and source/input hashes.
"""
from __future__ import annotations

from fractions import Fraction
import math
import struct


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _list(value):
    return value.tolist() if hasattr(value, 'tolist') else value


def _rational(value):
    value = Fraction(value)
    try:
        approximation = float(value)
    except OverflowError:
        approximation = None
    if approximation is not None and not math.isfinite(approximation):
        approximation = None
    return dict(numerator=str(value.numerator), denominator=str(value.denominator),
                float_approx=approximation)


def _dyadic(values):
    """Exact common power-of-two denominator of finite Python floats."""
    ratios = [float(value).as_integer_ratio() for value in values]
    exponent = max((den.bit_length() - 1 for _, den in ratios), default=0)
    return [num << (exponent - (den.bit_length() - 1)) for num, den in ratios], exponent


def _rounded_quotient(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    return quotient + int(twice > denominator or (twice == denominator and quotient % 2))


def binary32_bits(numerator, denominator=1):
    """Round an exact rational once to IEEE binary32, nearest/ties to even."""
    _need(isinstance(numerator, int) and isinstance(denominator, int) and denominator > 0,
          'Integer numerator and positive denominator required')
    sign = int(numerator < 0) << 31
    numerator = abs(numerator)
    if numerator == 0:
        return sign
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        exponent -= int(numerator < (denominator << exponent))
    else:
        exponent -= int((numerator << -exponent) < denominator)
    if exponent < -126:
        mantissa = _rounded_quotient(numerator << 149, denominator)
        return sign | mantissa  # May round to the minimum normal (bit 23).
    shift = 23 - exponent
    mantissa = _rounded_quotient(numerator << shift, denominator) if shift >= 0 else \
        _rounded_quotient(numerator, denominator << -shift)
    if mantissa == (1 << 24):
        exponent += 1
        mantissa >>= 1
    if exponent > 127:
        return sign | 0x7f800000
    return sign | ((exponent + 127) << 23) | (mantissa - (1 << 23))


def _bits_float(bits):
    return struct.unpack('>f', struct.pack('>I', bits))[0]


def _binary32_exact(value):
    value = float(value)
    if not math.isfinite(value):
        return False
    numerator, denominator = value.as_integer_ratio()
    return _bits_float(binary32_bits(numerator, denominator)) == value


def _matrix(flat, rows, width):
    return [flat[row * width:(row + 1) * width] for row in range(rows)]


def _scores(features, weights):
    return [[sum(x * w for x, w in zip(row, candidate)) for candidate in weights]
            for row in features]


def _comparisons(scores, labels):
    return [(i, k, row[labels[i]] - row[k]) for i, row in enumerate(scores)
            for k in range(len(row)) if k != labels[i]]


def _predictions(scores):
    # First class on ties is descriptive only: certificates require strictness.
    return [max(range(len(row)), key=row.__getitem__) for row in scores]


def _fp32_certificate(x_int, x_exp, features, labels, weight_int, weight_den):
    n, width, classes = len(features), len(features[0]), len(weight_int)
    bits = [[binary32_bits(value, weight_den) for value in row] for row in weight_int]
    candidate = [[_bits_float(value) for value in row] for row in bits]
    _need(all(math.isfinite(value) for row in candidate for value in row),
          'Normalized finite-L1 candidate must round to finite binary32')
    flat, exponent = _dyadic([value for row in candidate for value in row])
    w_int = _matrix(flat, classes, width)
    score_int = _scores(x_int, w_int)
    denominator = 1 << (x_exp + exponent)
    comparisons = _comparisons(score_int, labels)
    operations = 2 * width
    u, tiny = Fraction(1, 1 << 24), Fraction(1, 1 << 126)
    _need(operations * u < 1, 'FP32 operation bound requires 2R*u < 1')
    divisor = 1 - operations * u
    gamma = operations * u / divisor
    underflow = operations * tiny / divisor
    max_finite = Fraction(((1 << 24) - 1) << 104)
    feature_subnormal = [[value != 0 and abs(value) < float(tiny) for value in row]
                         for row in features]
    coefficient_subnormal = [[value != 0 and abs(value) < float(tiny) for value in row]
                             for row in candidate]
    errors, absolute_sums, daz_losses, overflow_safe = [], [], [], []
    for i in range(n):
        row_errors, row_sums, row_daz, row_safe = [], [], [], []
        for c in range(classes):
            products = [abs(x * w) for x, w in zip(x_int[i], w_int[c])]
            absolute_sum = Fraction(sum(products), denominator)
            daz = Fraction(sum(value for r, value in enumerate(products)
                               if feature_subnormal[i][r] or coefficient_subnormal[c][r]), denominator)
            row_errors.append(gamma * absolute_sum + underflow + daz)
            row_sums.append(absolute_sum)
            row_daz.append(daz)
            row_safe.append((absolute_sum + operations * tiny) / divisor < max_finite)
        errors.append(row_errors)
        absolute_sums.append(row_sums)
        daz_losses.append(row_daz)
        overflow_safe.append(row_safe)
    lower_bounds = [Fraction(margin, denominator) - errors[i][labels[i]] - errors[i][k]
                    for i, k, margin in comparisons]
    row_certified = [all(overflow_safe[i]) and all(
        Fraction(score_int[i][labels[i]] - score_int[i][k], denominator) >
        errors[i][labels[i]] + errors[i][k] for k in range(classes) if k != labels[i])
        for i in range(n)]
    predictions = _predictions(score_int)
    result = dict(
        available=True, certified_all=all(row_certified), certified_rows=sum(row_certified),
        row_certified=row_certified, exact_reference_predictions=predictions,
        exact_reference_correct=sum(a == b for a, b in zip(predictions, labels)),
        exact_reference_minimum_margin=_rational(Fraction(min(v for _, _, v in comparisons), denominator)),
        minimum_certified_margin_lower_bound=_rational(min(lower_bounds)),
        candidate_l1=_rational(Fraction(sum(abs(v) for v in flat), 1 << exponent)),
        candidate_l1_is_lp_feasibility_certificate=False,
        operations_per_score_bound=operations, unit_roundoff=_rational(u), gamma=_rational(gamma),
        additive_underflow_bound=_rational(underflow),
        maximum_absolute_product_sum=_rational(max(v for row in absolute_sums for v in row)),
        maximum_daz_loss=_rational(max(v for row in daz_losses for v in row)),
        overflow_safe=all(all(row) for row in overflow_safe),
        score_error_bounds=[[_rational(v) for v in row] for row in errors],
        margin_lower_bounds=[dict(scene=i, wrong_class=k, bound=_rational(bound))
                             for (i, k, _), bound in zip(comparisons, lower_bounds)],
        scoring_scope='IEEE binary32 products/additions or FMA, at most2R rounded operations per dot; '
                      'any reduction order; gradual underflow or FTZ and optional DAZ; no TF32 or reduced inputs',
        predictions_are_backend_observations=False)
    witness = dict(bits=bits, values=candidate,
                   exact_score_numerators=[[str(v) for v in row] for row in score_int],
                   exact_score_denominator=str(denominator))
    return result, witness


def certify(features, labels, solver_result, *, n_classes=9):
    """Independently certify candidates; solver success/epsilon never decides.

    The second return value contains classifier coefficients and belongs under
    CKPT. Missing candidates yield indeterminate certificates, not exceptions.
    Malformed feature/label inputs are errors. All exact values serialize as
    numerator/denominator strings with an explicitly approximate display float.
    """
    features, labels = _list(features), _list(labels)
    _need(isinstance(n_classes, int) and not isinstance(n_classes, bool) and n_classes >= 2,
          'At least two classes required')
    _need(isinstance(features, (list, tuple)) and len(features) > 0, 'Nonempty feature matrix required')
    features = [_list(row) for row in features]
    width = len(features[0])
    _need(width > 0 and all(len(row) == width for row in features), 'Rectangular positive-width features required')
    features = [[float(value) for value in row] for row in features]
    _need(all(_binary32_exact(value) for row in features for value in row),
          'Features must be finite exact promotions of binary32 values; no rounding or scaling')
    _need(isinstance(labels, (list, tuple)) and len(labels) == len(features), 'One label per feature row required')
    _need(all(not isinstance(value, bool) and int(value) == value and 0 <= int(value) < n_classes
              for value in labels), 'Integral class labels in range required')
    labels = [int(value) for value in labels]
    n, classes = len(features), n_classes
    flat_x, x_exp = _dyadic([value for row in features for value in row])
    x_int = _matrix(flat_x, n, width)
    comparison_count = n * (classes - 1)
    dimension = classes * width
    max_feature = Fraction(max(abs(value) for value in flat_x), 1 << x_exp)
    certificate = dict(
        status='indeterminate', dimensions=dict(contexts=n, width=width, classes=classes,
                                                coefficients=dimension, comparisons=comparison_count),
        scope='Strict complete-label homogeneous linear recoverability of these fixed features only; '
              'not native-head feasibility, weaker operational accuracy, or generalization',
        feature_transform='none; exact promoted binary32 values',
        maximum_absolute_feature=_rational(max_feature),
        solver_success_is_certificate=False,
        primal=dict(available=False), dual=dict(available=False), fp32=dict(available=False))
    witness = dict(format='integer numerator / positive integer denominator; exact',
                   comparison_order='scene ascending, wrong class ascending excluding target',
                   labels=labels, primal=None, dual=None, fp32=None)
    solver_result = solver_result if isinstance(solver_result, dict) else {}
    candidate = _list(solver_result.get('x'))
    candidate_ok = isinstance(candidate, (list, tuple)) and len(candidate) == 2 * dimension + 1
    if candidate_ok:
        try:
            candidate = [float(value) for value in candidate]
            candidate_ok = all(math.isfinite(value) for value in candidate)
        except (TypeError, ValueError, OverflowError):
            candidate_ok = False
    primal_lower = Fraction(0)
    if candidate_ok:
        split, exponent = _dyadic(candidate[:2 * dimension])
        raw_weights = [split[i] - split[i + dimension] for i in range(dimension)]
        raw_denominator = 1 << exponent
        l1_numerator = sum(abs(value) for value in raw_weights)
        denominator = max(raw_denominator, l1_numerator)
        weights = _matrix(raw_weights, classes, width)
        scores = _scores(x_int, weights)
        comparisons = _comparisons(scores, labels)
        score_denominator = denominator << x_exp
        minimum_margin = Fraction(min(value for _, _, value in comparisons), score_denominator)
        primal_lower = max(Fraction(0), minimum_margin)
        predictions = _predictions(scores)
        radii = []
        for i in range(n):
            row_margins = [(k, scores[i][labels[i]] - scores[i][k]) for k in range(classes) if k != labels[i]]
            if any(margin <= 0 for _, margin in row_margins):
                radii.append(Fraction(0))
            else:
                denominators = [(margin, sum(abs(a - b) for a, b in zip(weights[labels[i]], weights[k])))
                                for k, margin in row_margins]
                _need(all(value > 0 for _, value in denominators), 'Positive margin requires nonzero coefficient difference')
                radii.append(min(Fraction(margin, value << x_exp) for margin, value in denominators))
        certificate['primal'] = dict(
            available=True, strictly_separating=minimum_margin > 0,
            raw_effective_l1=_rational(Fraction(l1_numerator, raw_denominator)),
            normalization_divisor=_rational(Fraction(denominator, raw_denominator)),
            effective_l1=_rational(Fraction(l1_numerator, denominator)),
            split_nonnegative=all(value >= 0 for value in split),
            split_budget=_rational(Fraction(sum(split), raw_denominator)),
            solver_gamma=_rational(Fraction(candidate[-1])),
            minimum_margin=_rational(minimum_margin), lp_lower_bound=_rational(primal_lower),
            margin_relative_to_maximum_feature=_rational(minimum_margin / max_feature) if max_feature else None,
            zero_feature_case=max_feature == 0,
            exact_reference_predictions=predictions,
            exact_reference_correct=sum(a == b for a, b in zip(predictions, labels)),
            margins=[dict(scene=i, wrong_class=k, value=_rational(Fraction(value, score_denominator)))
                     for i, k, value in comparisons],
            per_scene_linf_perturbation_radius=[_rational(value) for value in radii],
            minimum_linf_perturbation_radius=_rational(min(radii)),
            perturbation_scope='Exact real-valued scoring; strict infinity-norm radius; boundary may tie; '
                               'nonpositive original margins give radius zero')
        witness['primal'] = dict(weight_numerators=[[str(v) for v in row] for row in weights],
                                weight_denominator=str(denominator), raw_weight_denominator=str(raw_denominator),
                                raw_split_numerators=[str(v) for v in split],
                                score_numerators=[[str(v) for v in row] for row in scores],
                                score_denominator=str(score_denominator))
        certificate['fp32'], witness['fp32'] = _fp32_certificate(
            x_int, x_exp, features, labels, weights, denominator)
    else:
        certificate['primal']['reason'] = 'Missing, malformed, or nonfinite solver x; no candidate certificate'
    inequality = solver_result.get('ineqlin') or {}
    marginal = _list(inequality.get('marginals')) if isinstance(inequality, dict) else None
    dual_ok = isinstance(marginal, (list, tuple)) and len(marginal) == comparison_count + 1
    if dual_ok:
        try:
            marginal = [float(value) for value in marginal]
            dual_ok = all(math.isfinite(value) for value in marginal)
        except (TypeError, ValueError, OverflowError):
            dual_ok = False
    upper = None
    if dual_ok:
        alpha, exponent = _dyadic([max(0., -value) for value in marginal[:comparison_count]])
        mass = sum(alpha)
        if mass > 0:
            transpose = [[0] * width for _ in range(classes)]
            index = 0
            for i in range(n):
                for k in range(classes):
                    if k == labels[i]:
                        continue
                    coefficient = alpha[index]
                    index += 1
                    for r in range(width):
                        term = coefficient * x_int[i][r]
                        transpose[labels[i]][r] += term
                        transpose[k][r] -= term
            residual = max(abs(value) for row in transpose for value in row)
            upper = Fraction(residual, mass << x_exp)
            certificate['dual'] = dict(
                available=True, nonnegative=True, positive_mass=True,
                clipped_without_epsilon=True,
                alpha_mass=_rational(Fraction(mass, 1 << exponent)),
                transpose_residual_linf=_rational(Fraction(residual, 1 << (exponent + x_exp))),
                lp_upper_bound=_rational(upper), exact_zero_farkas=residual == 0,
                bound_relative_to_maximum_feature=_rational(upper / max_feature) if max_feature else None,
                stationarity_not_required=True)
            witness['dual'] = dict(alpha_numerators=[str(v) for v in alpha], alpha_denominator=str(1 << exponent),
                                   transpose_numerators=[[str(v) for v in row] for row in transpose],
                                   transpose_denominator=str(1 << (exponent + x_exp)))
        else:
            certificate['dual']['reason'] = 'Clipped candidate has zero mass; no upper bound'
    else:
        certificate['dual']['reason'] = 'Missing, malformed, or nonfinite inequality marginals'
    if upper is not None:
        _need(primal_lower <= upper, 'Exact primal/dual bounds contradict; certificate implementation or inputs invalid')
    positive = certificate['primal'].get('strictly_separating', False)
    negative = certificate['dual'].get('exact_zero_farkas', False)
    _need(not (positive and negative), 'Exact positive and negative certificates cannot coexist')
    certificate['status'] = 'strictly_separable' if positive else 'not_strictly_separable' if negative else 'indeterminate'
    certificate['interval'] = dict(lower=_rational(primal_lower), upper=_rational(upper) if upper is not None else None,
                                   exact_gap=_rational(upper - primal_lower) if upper is not None else None,
                                   lower_includes_zero_classifier=True, solver_optimum_claimed=False)
    return certificate, witness


def self_test():
    """Small exact fixtures; no LP solves, backend scoring, or tensor imports."""
    def packet(weights, alpha, gamma=0.):
        flat = [float(value) for row in weights for value in row]
        return dict(x=[max(value, 0.) for value in flat] + [max(-value, 0.) for value in flat] + [gamma],
                    ineqlin=dict(marginals=[-float(value) for value in alpha] + [-1.]))

    separate, _ = certify([[-1.], [1.]], [0, 1], packet([[-.5], [.5]], [.5, .5], 1.), n_classes=2)
    _need(separate['status'] == 'strictly_separable' and separate['fp32']['certified_all']
          and separate['interval']['exact_gap']['numerator'] == '0', 'Separable primal/dual fixture')
    repaired, _ = certify([[-1.], [1.]], [0, 1], packet([[-2.], [2.]], [.5, .5]), n_classes=2)
    _need(repaired['primal']['normalization_divisor']['numerator'] == '4'
          and repaired['primal']['minimum_margin'] == separate['primal']['minimum_margin'], 'Exact L1 repair fixture')
    contradiction, _ = certify([[1.], [1.]], [0, 1], packet([[0.], [0.]], [.5, .5]), n_classes=2)
    _need(contradiction['status'] == 'not_strictly_separable'
          and contradiction['primal']['minimum_linf_perturbation_radius']['numerator'] == '0', 'Exact Farkas/tie fixture')
    near, _ = certify([[1.], [1. + 2.**-23]], [0, 1], packet([[0.], [0.]], [1., 1.]), n_classes=2)
    _need(near['status'] == 'indeterminate' and near['dual']['lp_upper_bound']['numerator'] != '0', 'No approximate-zero rejection')
    subnormal, _ = certify([[-2.**-149], [2.**-149]], [0, 1], packet([[-.5], [.5]], [.5, .5]), n_classes=2)
    _need(subnormal['status'] == 'strictly_separable' and not subnormal['fp32']['certified_all']
          and subnormal['fp32']['maximum_daz_loss']['numerator'] != '0', 'Real margin versus FP32 underflow fixture')
    _need(binary32_bits(1, 1 << 150) == 0 and binary32_bits(3, 1 << 150) == 2
          and binary32_bits((1 << 24) + 1, 1 << 24) == 0x3f800000
          and binary32_bits(-1, 1) == 0xbf800000, 'Exact ties-even normal/subnormal/sign fixtures')
    overflow, _ = certify([[_bits_float(0x7f7fffff)]], [0], packet([[1.], [0.]], [1.]), n_classes=2)
    _need(overflow['status'] == 'strictly_separable' and not overflow['fp32']['overflow_safe'], 'Conservative overflow guard fixture')
    absent, _ = certify([[0.]], [0], {}, n_classes=2)
    _need(absent['status'] == 'indeterminate' and absent['maximum_absolute_feature']['numerator'] == '0', 'Missing candidates fixture')
    return dict(passed=True, solver_calls=0, backend_dot_products=0,
                groups=['separable_primal_dual_fp32', 'exact_l1_repair', 'exact_farkas_tie_radius',
                        'near_zero_indeterminate', 'subnormal_daz_underflow', 'ties_even_rounding',
                        'conservative_overflow_guard', 'missing_candidates_zero_features'])

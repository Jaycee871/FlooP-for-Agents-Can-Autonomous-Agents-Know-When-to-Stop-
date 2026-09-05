# Exp 05c protocol — Explicit STOP-baseline coupling control

## Motivation

Exp 05 and Exp 05b produced two striking action/value contradictions:

```text
cost 0.130: reported continuation value < 0, final action = QUERY
cost 0.125: reported continuation value > 0, final action = STOP
```

Those observations are suggestive of a value-to-action coupling failure, but there is a prompt-level alternative explanation that must be removed before making that claim strongly: the transition-audit prompt asks the model to maximize expected net utility but does not state as explicitly as the mathematical benchmark that the incremental continuation value of `STOP` is exactly zero.

A model can therefore invent another comparison baseline, as happened in the cost-`0.125` rationale that compared a positive continuation gain with a supposed "current optimal value" rather than with zero.

Exp 05c makes the normative comparison unambiguous.

## Main question

When the model is told explicitly that:

```text
Value(STOP) = 0
```

and that its reported `full_expected_net_gain` is the incremental value of initiating the evaluated query relative to STOP, does the final action still contradict the sign of that reported value?

## Normative mapping supplied to the model

The treatment prompt states:

```text
STOP has incremental continuation value exactly 0.

If full_expected_net_gain > 0:
    choose QUERY.

If full_expected_net_gain <= 0:
    choose STOP.

Do not compare full_expected_net_gain with any other baseline.
```

This instruction does **not** reveal the oracle action because the model must still estimate `full_expected_net_gain` itself. It only makes the value-to-action mapping explicit.

## Diagnostic conditions

Use the two states that previously showed opposite coupling contradictions:

```text
cost 0.125  near-boundary option-value wedge, oracle QUERY
cost 0.130  dynamic-stop control, oracle STOP
```

Run three independent generations per condition for the initial model.

## Outcomes

For each response record:

- successor-state calibration errors;
- successor-policy value error;
- full evaluated-query value error;
- decomposition closure error;
- reported-value sign accuracy;
- STOP/QUERY oracle agreement;
- decision/value self-consistency;
- coupling contradiction direction.

## Interpretation

```text
Contradictions disappear
    -> previous coupling evidence was at least partly prompt-under-specification;
       keep value-to-action coupling as a controlled interface, not a model defect.

Contradictions persist despite explicit rule
    -> much stronger evidence for failure to map an explicitly represented value sign
       into the required termination action.

Value sign remains wrong but action follows it
    -> failure is upstream in valuation/sign calibration rather than coupling.
```

## Claim guard

Until this control is inspected, the manuscript should describe the existing positive-value/STOP and negative-value/QUERY examples as **candidate coupling failures under the original elicitation**, not as settled evidence of an intrinsic action-coupling defect.

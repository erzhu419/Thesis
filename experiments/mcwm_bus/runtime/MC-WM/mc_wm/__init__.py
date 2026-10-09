"""MC-WM residual world-model research code.

The submission path learns state/reward/termination corrections, deploys a
sparse linear feature model plus a masked neural remainder, calibrates model
error on disjoint development data, and optionally filters imagined rollouts
with QDelta and audited constraints. No formal OOD error guarantee is claimed.
"""

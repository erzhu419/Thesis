"""Reassess the existing v105 identity receipt without changing its evidence."""


def assess(summary, original_protocol):
    """Correct only the numeric-versus-JSON episode-key comparison."""
    training = summary["training"]
    recorded = summary["identity"]
    budget = summary["budget"]
    actor = summary["actor_identity"]
    p = original_protocol
    observed = training["demand_scale_episode_counts"]
    expected = p["expected_source_episodes"]
    numeric_observed = {float(key): value for key, value in observed.items()}
    numeric_expected = {float(key): value for key, value in expected.items()}
    declared_scales = set(p["source_demand_scales"])
    episodes_equal = (
        set(numeric_observed) == set(numeric_expected) == declared_scales
        and numeric_observed == numeric_expected
    )
    checks = {
        "actor_exact": recorded["actor_exact"] is True
        and actor["all_pass"] is True
        and actor["differing_field_count"] == 0
        and actor["differing_fields"] == []
        and actor["max_abs_difference"] == 0,
        "historical_seed": recorded["historical_seed"] is True
        and training["seed"] == p["training_seed"],
        "historical_endpoint": recorded["historical_endpoint"] is True
        and training["total_policy_updates"] == p["saved_terminal_updates"]
        and summary["terminal_updates_at_exit"] == p["saved_terminal_updates"],
        "original_source_events": recorded["original_source_events"] is True
        and training["source_events"] == p["expected_source_events"]
        and budget["source_events"] == p["expected_source_events"],
        "original_update_budget": recorded["original_update_budget"] is True
        and training["continuation_updates"] == p["new_source_continuation_updates"]
        and budget["source_continuation_updates"] == p["new_source_continuation_updates"],
        "original_source_episodes": episodes_equal,
        "original_model_mixture": recorded["original_model_mixture"] is True
        and training["model_td_coefficient"] == p["model_td_coefficient"]
        and training["real_td_coefficient"] == p["real_td_coefficient"],
        "target_labels_unused": recorded["target_labels_unused"] is True
        and training["target_online_training_events"] == 0
        and budget["target_feedback_labels"] == 0,
    }
    return {
        "all_pass": all(checks.values()),
        "checks": checks,
        "failures": [key for key, passed in checks.items() if not passed],
        "episode_counts": {
            "observed": observed,
            "expected": expected,
            "numeric_keys_equal": set(numeric_observed) == set(numeric_expected),
            "declared_scales": sorted(declared_scales),
            "missing_zero_keys_filled": False,
        },
        "original_gate_error": {
            "confirmed": recorded["original_source_episodes"] is False and episodes_equal,
            "reason": "The live sampler uses float keys; the JSON protocol uses string keys.",
            "runtime_float_key_example": repr(numeric_observed),
            "protocol_string_key_example": repr(expected),
            "raw_comparison": numeric_observed == expected,
            "normalized_comparison": episodes_equal,
        },
        "evidence_scope": "Receipt adjudication only. The other seven original checks must remain true and agree with available receipt facts; this does not perform checkpoint restoration or a next-update test.",
    }

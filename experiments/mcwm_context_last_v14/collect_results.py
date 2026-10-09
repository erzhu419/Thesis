"""Report this branch only; reused controls are not fresh training repetitions."""

from last_common import HERE, P, CONTEXTS, module, initializer_folder, policy_folder

collector = module("last_confirmation_collect", HERE.parent / "mcwm_seed_confirmation_v13/collect_results.py")
collector.HERE, collector.P, collector.CONTEXTS = HERE, P, CONTEXTS
collector.initializer_folder, collector.policy_folder = initializer_folder, policy_folder
summarize, select_development = collector.summarize, collector.select_development
server_analysis = collector.server_analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    server_analysis(parser.parse_args().smoke)

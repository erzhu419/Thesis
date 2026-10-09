"""Separate submissions and summaries; explicit resident main-control references."""

from last_common import (HERE, P, CONTEXTS, module, bundle_path, initializer_folder, policy_folder)

main_launch = module("last_confirmation_launch", HERE.parent / "mcwm_seed_confirmation_v13/launch.py")
main_launch.HERE, main_launch.P, main_launch.CONTEXTS = HERE, P, CONTEXTS
main_launch.bundle_path, main_launch.initializer_folder, main_launch.policy_folder = bundle_path, initializer_folder, policy_folder
placements, jobs, task = main_launch.placements, main_launch.jobs, main_launch.task
collect, handoff, stage = main_launch.collect, main_launch.handoff, main_launch.stage


if __name__ == "__main__":
    main_launch.main()

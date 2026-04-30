from __future__ import annotations

import random

from lfothello.tests import (
    test_case_1_initial_actions,
    test_case_2_apply_move,
    test_case_3_limited_flip_rule
)
from lfothello.experiments import (
    experiment_baseline_vs_random,
    experiment_baseline_vs_tt,
    experiment_vary_k_baseline_vs_random,
    experiment_fixed_tt_vs_id_tt
)

# random seed for reproduceability
SEED = 42


def main() -> None:
    """
    Main driver for the program. Runs the test cases and experiments and prints
    out results/statistics.
    """
    # set random seed for reproduceability
    random.seed(SEED)

    # test cases
    test_case_1_initial_actions()
    print()
    test_case_2_apply_move()
    print()
    test_case_3_limited_flip_rule()
    print()

    # experiments
    experiment_baseline_vs_random(
        num_games=30,
        depth=3,
        flip_limit=2,
        verbose_each_game=False
    )
    print()
    experiment_baseline_vs_tt(
        num_games=10,
        depth=4,
        flip_limit=2,
        verbose_each_game=False
    )
    print()
    experiment_vary_k_baseline_vs_random(
        k_values=[1, 2, 3, 4],
        num_games_per_k=30,
        depth=3,
        verbose_each_game=False,
    )
    print()
    experiment_fixed_tt_vs_id_tt(
        num_games=30,
        fixed_depth=4,
        id_max_depth=10,
        time_limit_seconds=1.0,
        flip_limit=2,
        verbose_each_game=False,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import statistics
import random
import numpy as np

from lfothello.ai import MinimaxAlphaBetaPlayer, RandomPlayer
from lfothello.game import Board
from lfothello.types import Cell, DiskColor


def test_case_1_initial_actions() -> None:
    """
    Test that the initial board state generates the correct number of legal moves.

    This test verifies that:
    - The standard Othello starting position is initialized correctly.
    - Legal move generation works as expected for the opening position.
    - The number of valid moves for the starting player (BLACK) is exactly 4.
    """
    print("=== TEST CASE 1: Initial legal moves ===")
    board = Board(flip_limit=2)
    board.display_state()
    actions = board.get_actions()
    print(f"Number of legal moves for BLACK: {len(actions)}")
    print("Moves:", actions)
    # Standard opening should have 4 legal moves.
    assert len(actions) == 4, "Expected 4 legal opening moves."


def test_case_2_apply_move() -> None:
    """
    Test that applying a valid move correctly updates the board state.

    This test verifies that:
    - A legal move can be selected and applied.
    - The board state is updated correctly after the move.
    - Disk flipping follows standard Othello rules.
    - Disk counts reflect the expected outcome.
    """
    print("=== TEST CASE 2: Apply one move ===")
    board = Board(flip_limit=2)
    state = board.get_state()

    actions = board.get_actions(state)
    move = actions[0]
    print("Applying move:", move)

    next_state = board.apply_action(state, move)
    board.display_state(next_state)

    black, white = board.disk_counts(next_state)
    print(f"Disk counts after move -> Black: {black}, White: {white}")

    # After first legal Othello move, board should have 4 black and 1 white.
    assert black == 4 and white == 1, "Unexpected disk counts after first move."


def test_case_3_limited_flip_rule() -> None:
    """
    Test the Limited-Flip Othello rule (k-flip constraint).

    This test constructs a custom board configuration where a move would
    normally flip multiple opponent disks in a straight line. It verifies
    that only the closest k disks are flipped according to the specified
    flip limit.
    """
    print("=== TEST CASE 3: Limited-flip rule ===")

    flip_limit = 2
    board = np.full((8, 8), Cell.EMPTY, dtype=np.int8)

    # Create a row like:
    # B W W W W .   on row 3
    #
    # Black to move at (3, 5). In standard Othello, this would flip
    # all four white disks at (3,1), (3,2), (3,3), (3,4).
    # In Limited-Flip Othello with k=2, only the two closest disks
    # to the placed disk should flip: (3,4) and (3,3).

    board[3, 0] = Cell.BLACK
    board[3, 1] = Cell.WHITE
    board[3, 2] = Cell.WHITE
    board[3, 3] = Cell.WHITE
    board[3, 4] = Cell.WHITE

    game = Board(initial=board, flip_limit=flip_limit)
    game.state = (board.copy(), DiskColor.BLACK)

    print("Before move:")
    game.display_state()

    move = (3, 5, DiskColor.BLACK)
    assert game.is_legal_action(game.state, move), "Move should be legal."

    next_state = game.apply_action(game.state, move)

    print("After move:")
    game.display_state(next_state)

    next_board, next_to_move = next_state

    # The placed disk should be black
    assert next_board[3, 5] == Cell.BLACK

    # Only the two closest whites should flip
    assert next_board[3, 4] == Cell.BLACK
    assert next_board[3, 3] == Cell.BLACK

    # These should remain white because of the flip limit
    assert next_board[3, 2] == Cell.WHITE
    assert next_board[3, 1] == Cell.WHITE

    # Original anchor stays black
    assert next_board[3, 0] == Cell.BLACK

    # Turn should switch
    assert next_to_move == DiskColor.WHITE

    print("Limited-flip rule test passed.")


def play_game_with_agent_stats(
    black_player,
    white_player,
    tracked_players: list,
    flip_limit: int = 2,
    verbose: bool = False,
):
    """
    Simulate a full game between two agents while collecting performance statistics.

    This function runs a complete Limited-Flip Othello game using the provided
    black and white players. During the game, it tracks search-related statistics
    (e.g., nodes expanded, alpha-beta cutoffs, search time, and transposition
    table hits) for any players specified in `tracked_players`.

    Parameters
    ----------
    black_player : Player
        The agent playing as BLACK.
    white_player : Player
        The agent playing as WHITE.
    tracked_players : list
        List of player objects for which statistics should be recorded.
        Typically includes one or both agents participating in the game.
    flip_limit : int, default=2
        Maximum number of disks flipped per direction (k in Limited-Flip Othello).
    verbose : bool, default=False
        If True, prints the board state at each step for debugging or visualization.

    Returns
    -------
    result : GameResult
        Final game outcome, including disk counts and winner.
    stats_by_player : dict
        Dictionary mapping each tracked player to a dictionary of accumulated
        statistics
    """
    board = Board(flip_limit=flip_limit)

    stats_by_player = {
        player: {
            "nodes": 0,
            "cutoffs": 0,
            "time": 0.0,
            "tt_hits": 0,
        }
        for player in tracked_players
    }

    while not board.is_terminal(board.state):
        if verbose:
            board.display_state()

        state = board.get_state()
        _, to_move = state

        current_player = black_player if to_move == DiskColor.BLACK else white_player
        actions = board.get_actions(state)

        if not actions:
            board.state = board.pass_turn(board.state)
            continue

        move = current_player.get_next_move(board, state)

        if current_player in stats_by_player and hasattr(current_player, "stats"):
            stats_by_player[current_player]["nodes"] += current_player.stats.nodes_expanded
            stats_by_player[current_player]["cutoffs"] += current_player.stats.cutoffs
            stats_by_player[current_player]["time"] += current_player.stats.elapsed_seconds

            if hasattr(current_player.stats, "tt_hits"):
                stats_by_player[current_player]["tt_hits"] += current_player.stats.tt_hits

        if move is None:
            board.state = board.pass_turn(board.state)
            continue

        board.make_move(move)

    if verbose:
        board.display_state()

    result = board.get_result()
    return result, stats_by_player


def experiment_baseline_vs_random(
    num_games: int = 10,
    depth: int = 3,
    flip_limit: int = 2,
    verbose_each_game: bool = False,
) -> None:
    """
    Run an experiment comparing the baseline minimax agent against a random agent.

    This experiment evaluates the correctness and performance of the baseline
    MinimaxAlphaBetaPlayer (without enhancements) by playing multiple games
    against a RandomPlayer. It reports both game outcomes and search statistics.

    Parameters
    ----------
    num_games : int, default=10
        Number of games to simulate.
    depth : int, default=3
        Maximum search depth for the baseline agent.
    flip_limit : int, default=2
        Maximum number of disks flipped per direction (k in Limited-Flip Othello).
    verbose_each_game : bool, default=False
        If True, prints the board state during each game.
    """
    print("=== EXPERIMENT: Baseline (no enhancements) vs Random ===")

    baseline_wins = 0
    random_wins = 0
    draws = 0

    nodes_list: list[int] = []
    cutoffs_list: list[int] = []
    time_list: list[float] = []

    for game_idx in range(num_games):
        baseline_color = random.choice([DiskColor.BLACK, DiskColor.WHITE])
        random_color = baseline_color.opponent()

        baseline = MinimaxAlphaBetaPlayer(
            color=baseline_color,
            max_depth=depth,
            use_transposition_table=False,
            use_iterative_deepening=False,
        )
        random_player = RandomPlayer(random_color)

        if baseline_color == DiskColor.BLACK:
            black_player = baseline
            white_player = random_player
        else:
            black_player = random_player
            white_player = baseline

        result, stats_by_player = play_game_with_agent_stats(
            black_player=black_player,
            white_player=white_player,
            tracked_players=[baseline],
            flip_limit=flip_limit,
            verbose=verbose_each_game,
        )

        baseline_stats = stats_by_player[baseline]
        game_nodes = baseline_stats["nodes"]
        game_cutoffs = baseline_stats["cutoffs"]
        game_time = baseline_stats["time"]

        nodes_list.append(game_nodes)
        cutoffs_list.append(game_cutoffs)
        time_list.append(game_time)

        if result.winner == baseline_color:
            baseline_wins += 1
        elif result.winner is None:
            draws += 1
        else:
            random_wins += 1

        winner_str = "DRAW" if result.winner is None else (
            "BLACK" if result.winner == DiskColor.BLACK else "WHITE"
        )

        print(
            f"Game {game_idx + 1}: "
            f"Baseline as {'BLACK' if baseline_color == DiskColor.BLACK else 'WHITE'} | "
            f"Black={result.black_disks}, White={result.white_disks}, "
            f"Winner={winner_str}, "
            f"Nodes={game_nodes}, Cutoffs={game_cutoffs}, Time={game_time:.6f}s"
        )

    avg_nodes = statistics.mean(nodes_list)
    std_nodes = statistics.stdev(nodes_list) if len(nodes_list) > 1 else 0.0

    avg_cutoffs = statistics.mean(cutoffs_list)
    std_cutoffs = statistics.stdev(cutoffs_list) if len(cutoffs_list) > 1 else 0.0

    avg_time = statistics.mean(time_list)
    std_time = statistics.stdev(time_list) if len(time_list) > 1 else 0.0

    print("\n--- Summary ---")
    print(f"Baseline wins: {baseline_wins}")
    print(f"Random wins:   {random_wins}")
    print(f"Draws:         {draws}")
    print(f"Baseline win rate: {baseline_wins / num_games:.2%}")

    print("\nSearch Performance (mean +- std):")
    print(f"Nodes expanded: {avg_nodes:.2f} +- {std_nodes:.2f}")
    print(f"Alpha-beta cutoffs: {avg_cutoffs:.2f} +- {std_cutoffs:.2f}")
    print(f"Search time (s): {avg_time:.6f} +- {std_time:.6f}")


def experiment_baseline_vs_tt(
    num_games: int = 10,
    depth: int = 4,
    flip_limit: int = 2,
    verbose_each_game: bool = False,
) -> None:
    """
    Compare the baseline minimax agent against a version enhanced with
    transposition tables (TT).

    This experiment evaluates the impact of transposition tables on both
    playing performance and search efficiency. The two agents are identical
    except that the TT-enabled agent caches previously evaluated states
    and uses them to avoid redundant computation and improve move ordering.

    Parameters
    ----------
    num_games : int, default=10
        Number of games to simulate.
    depth : int, default=4
        Maximum search depth for both agents.
    flip_limit : int, default=2
        Maximum number of disks flipped per direction (k in Limited-Flip Othello).
    verbose_each_game : bool, default=False
        If True, prints the board state during each game.
    """
    print("=== EXPERIMENT: Baseline vs Baseline + Transposition Tables ===")

    baseline_wins = 0
    tt_wins = 0
    draws = 0

    # Per-game tracking
    baseline_nodes_list = []
    baseline_cutoffs_list = []
    baseline_time_list = []

    tt_nodes_list = []
    tt_cutoffs_list = []
    tt_time_list = []
    tt_hits_list = []

    for game_idx in range(num_games):
        # Randomize roles
        tt_color = random.choice([DiskColor.BLACK, DiskColor.WHITE])
        baseline_color = tt_color.opponent()

        baseline = MinimaxAlphaBetaPlayer(
            color=baseline_color,
            max_depth=depth,
            use_transposition_table=False,
            use_iterative_deepening=False,
        )

        tt_agent = MinimaxAlphaBetaPlayer(
            color=tt_color,
            max_depth=depth,
            use_transposition_table=True,
            use_iterative_deepening=False,
        )

        if baseline_color == DiskColor.BLACK:
            black_player = baseline
            white_player = tt_agent
        else:
            black_player = tt_agent
            white_player = baseline

        result, stats = play_game_with_agent_stats(
            black_player=black_player,
            white_player=white_player,
            tracked_players=[baseline, tt_agent],
            flip_limit=flip_limit,
            verbose=verbose_each_game,
        )

        baseline_stats = stats[baseline]
        tt_stats = stats[tt_agent]

        # Store per-game values
        baseline_nodes_list.append(baseline_stats["nodes"])
        baseline_cutoffs_list.append(baseline_stats["cutoffs"])
        baseline_time_list.append(baseline_stats["time"])

        tt_nodes_list.append(tt_stats["nodes"])
        tt_cutoffs_list.append(tt_stats["cutoffs"])
        tt_time_list.append(tt_stats["time"])
        tt_hits_list.append(tt_stats["tt_hits"])

        # Win tracking
        if result.winner == baseline_color:
            baseline_wins += 1
        elif result.winner == tt_color:
            tt_wins += 1
        else:
            draws += 1

        winner_str = "DRAW" if result.winner is None else (
            "BLACK" if result.winner == DiskColor.BLACK else "WHITE"
        )

        print(
            f"Game {game_idx + 1}: "
            f"Baseline as {'BLACK' if baseline_color == DiskColor.BLACK else 'WHITE'}, "
            f"Baseline + TT as {'WHITE' if baseline_color == DiskColor.BLACK else 'BLACK'} | "
            f"Black={result.black_disks}, White={result.white_disks}, "
            f"Winner={winner_str}"
        )
        print(
            f"  Baseline -> Nodes={baseline_stats['nodes']}, "
            f"Cutoffs={baseline_stats['cutoffs']}, Time={baseline_stats['time']:.6f}s"
        )
        print(
            f"  TT Agent -> Nodes={tt_stats['nodes']}, "
            f"Cutoffs={tt_stats['cutoffs']}, Time={tt_stats['time']:.6f}s, "
            f"TT Hits={tt_stats['tt_hits']}"
        )

    # Compute statistics
    def mean_std(lst):
        return (
            statistics.mean(lst),
            statistics.stdev(lst) if len(lst) > 1 else 0.0,
        )

    b_nodes_mean, b_nodes_std = mean_std(baseline_nodes_list)
    b_cut_mean, b_cut_std = mean_std(baseline_cutoffs_list)
    b_time_mean, b_time_std = mean_std(baseline_time_list)

    tt_nodes_mean, tt_nodes_std = mean_std(tt_nodes_list)
    tt_cut_mean, tt_cut_std = mean_std(tt_cutoffs_list)
    tt_time_mean, tt_time_std = mean_std(tt_time_list)
    tt_hits_mean, tt_hits_std = mean_std(tt_hits_list)

    print("\n--- Summary ---")
    print(f"Baseline wins: {baseline_wins}")
    print(f"TT agent wins: {tt_wins}")
    print(f"Draws:         {draws}")

    print("\nBaseline (mean +- std):")
    print(f"  Nodes:   {b_nodes_mean:.2f} +- {b_nodes_std:.2f}")
    print(f"  Cutoffs: {b_cut_mean:.2f} +- {b_cut_std:.2f}")
    print(f"  Time:    {b_time_mean:.6f} +- {b_time_std:.6f}")

    print("\nTT Agent (mean +- std):")
    print(f"  Nodes:   {tt_nodes_mean:.2f} +- {tt_nodes_std:.2f}")
    print(f"  Cutoffs: {tt_cut_mean:.2f} +- {tt_cut_std:.2f}")
    print(f"  Time:    {tt_time_mean:.6f} +- {tt_time_std:.6f}")
    print(f"  TT Hits: {tt_hits_mean:.2f} +- {tt_hits_std:.2f}")


def experiment_vary_k_baseline_vs_random(
    k_values: list[int] = [1, 2, 3, 4],
    num_games_per_k: int = 10,
    depth: int = 3,
    verbose_each_game: bool = False,
) -> None:
    """
    Evaluate how the Limited-Flip parameter (k) affects baseline agent performance.

    This experiment measures the impact of varying the flip limit k on both
    gameplay outcomes and search performance. For each value of k, the baseline
    minimax agent (with alpha-beta pruning only) plays multiple games against
    a random agent, and performance metrics are recorded.

    Parameters
    ----------
    k_values : list[int], default=[1, 2, 3, 4]
        List of flip-limit values (k) to evaluate.
    num_games_per_k : int, default=10
        Number of games to simulate for each value of k.
    depth : int, default=3
        Maximum search depth for the baseline agent.
    verbose_each_game : bool, default=False
        If True, prints the board state during each game.
    """
    print("=== EXPERIMENT: Effect of varying flip limit k ===")

    all_results = []

    for k in k_values:
        baseline_wins = 0
        random_wins = 0
        draws = 0

        nodes_list: list[int] = []
        cutoffs_list: list[int] = []
        time_list: list[float] = []
        margin_list: list[int] = []

        print(f"\n--- k = {k} ---")

        for game_idx in range(num_games_per_k):
            baseline_color = random.choice([DiskColor.BLACK, DiskColor.WHITE])
            random_color = baseline_color.opponent()

            baseline = MinimaxAlphaBetaPlayer(
                color=baseline_color,
                max_depth=depth,
                use_transposition_table=False,
                use_iterative_deepening=False,
            )
            random_player = RandomPlayer(random_color)

            if baseline_color == DiskColor.BLACK:
                black_player = baseline
                white_player = random_player
            else:
                black_player = random_player
                white_player = baseline

            result, stats_by_player = play_game_with_agent_stats(
                black_player=black_player,
                white_player=white_player,
                tracked_players=[baseline],
                flip_limit=k,
                verbose=verbose_each_game,
            )

            baseline_stats = stats_by_player[baseline]
            game_nodes = baseline_stats["nodes"]
            game_cutoffs = baseline_stats["cutoffs"]
            game_time = baseline_stats["time"]

            if baseline_color == DiskColor.BLACK:
                margin = result.black_disks - result.white_disks
            else:
                margin = result.white_disks - result.black_disks

            nodes_list.append(game_nodes)
            cutoffs_list.append(game_cutoffs)
            time_list.append(game_time)
            margin_list.append(margin)

            if result.winner == baseline_color:
                baseline_wins += 1
            elif result.winner is None:
                draws += 1
            else:
                random_wins += 1

            winner_str = "DRAW" if result.winner is None else (
                "BLACK" if result.winner == DiskColor.BLACK else "WHITE"
            )

            print(
                f"Game {game_idx + 1}: "
                f"Baseline as {'BLACK' if baseline_color == DiskColor.BLACK else 'WHITE'} | "
                f"Black={result.black_disks}, White={result.white_disks}, "
                f"Winner={winner_str}, "
                f"Nodes={game_nodes}, Cutoffs={game_cutoffs}, Time={game_time:.6f}s, "
                f"Margin={margin}"
            )

        avg_nodes = statistics.mean(nodes_list)
        std_nodes = statistics.stdev(nodes_list) if len(nodes_list) > 1 else 0.0

        avg_cutoffs = statistics.mean(cutoffs_list)
        std_cutoffs = statistics.stdev(cutoffs_list) if len(cutoffs_list) > 1 else 0.0

        avg_time = statistics.mean(time_list)
        std_time = statistics.stdev(time_list) if len(time_list) > 1 else 0.0

        avg_margin = statistics.mean(margin_list)
        std_margin = statistics.stdev(margin_list) if len(margin_list) > 1 else 0.0

        win_rate = baseline_wins / num_games_per_k

        all_results.append(
            {
                "k": k,
                "baseline_wins": baseline_wins,
                "random_wins": random_wins,
                "draws": draws,
                "win_rate": win_rate,
                "avg_nodes": avg_nodes,
                "std_nodes": std_nodes,
                "avg_cutoffs": avg_cutoffs,
                "std_cutoffs": std_cutoffs,
                "avg_time": avg_time,
                "std_time": std_time,
                "avg_margin": avg_margin,
                "std_margin": std_margin,
            }
        )

        print("\nSummary:")
        print(f"  Baseline wins: {baseline_wins}")
        print(f"  Random wins:   {random_wins}")
        print(f"  Draws:         {draws}")
        print(f"  Win rate:      {win_rate:.2%}")

        print("  Performance (mean +- std):")
        print(f"    Nodes:       {avg_nodes:.2f} +- {std_nodes:.2f}")
        print(f"    Cutoffs:     {avg_cutoffs:.2f} +- {std_cutoffs:.2f}")
        print(f"    Time (s):    {avg_time:.6f} +- {std_time:.6f}")
        print(f"    Disk margin: {avg_margin:.2f} +- {std_margin:.2f}")

    print("\n=== OVERALL RESULTS BY k ===")
    for row in all_results:
        print(
            f"k={row['k']}: "
            f"win_rate={row['win_rate']:.2%}, "
            f"nodes={row['avg_nodes']:.2f} +- {row['std_nodes']:.2f}, "
            f"cutoffs={row['avg_cutoffs']:.2f} +- {row['std_cutoffs']:.2f}, "
            f"time={row['avg_time']:.6f} +- {row['std_time']:.6f}s, "
            f"margin={row['avg_margin']:.2f} +- {row['std_margin']:.2f}"
        )


def main() -> None:
    """
    Main driver for the program. Runs the test cases and experiments and prints
    out results/statistics.
    """
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


if __name__ == "__main__":
    main()
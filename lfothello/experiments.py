from __future__ import annotations

import statistics
import random

from lfothello.ai import MinimaxAlphaBetaPlayer, RandomPlayer
from lfothello.game import Board
from lfothello.types import DiskColor


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
            "moves": 0,
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
            stats_by_player[current_player]["moves"] += 1

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


def experiment_fixed_tt_vs_id_tt(
    num_games: int = 10,
    fixed_depth: int = 4,
    id_max_depth: int = 10,
    time_limit_seconds: float = 2.0,
    flip_limit: int = 2,
    verbose_each_game: bool = False,
) -> None:
    """
    Compare fixed-depth alpha-beta + TT against iterative deepening + TT.

    Fixed TT searches to a fixed depth, e.g. depth=4.
    ID+TT searches progressively deeper up to id_max_depth, but stops after
    time_limit_seconds.
    """
    print("=== EXPERIMENT: Fixed TT vs Iterative Deepening + TT ===")

    fixed_tt_wins = 0
    id_tt_wins = 0
    draws = 0

    fixed_nodes = []
    fixed_time = []
    fixed_cutoffs = []
    fixed_tt_hits = []
    fixed_moves = []

    id_nodes = []
    id_time = []
    id_cutoffs = []
    id_tt_hits = []
    id_depths = []
    id_moves = []

    for game_idx in range(num_games):
        id_tt_color = random.choice([DiskColor.BLACK, DiskColor.WHITE])
        fixed_tt_color = id_tt_color.opponent()

        fixed_tt_agent = MinimaxAlphaBetaPlayer(
            color=fixed_tt_color,
            max_depth=fixed_depth,
            use_transposition_table=True,
            use_iterative_deepening=False,
        )

        id_tt_agent = MinimaxAlphaBetaPlayer(
            color=id_tt_color,
            max_depth=id_max_depth,
            use_transposition_table=True,
            use_iterative_deepening=True,
            time_limit_seconds=time_limit_seconds,
        )

        if fixed_tt_color == DiskColor.BLACK:
            black_player = fixed_tt_agent
            white_player = id_tt_agent
        else:
            black_player = id_tt_agent
            white_player = fixed_tt_agent

        result, stats = play_game_with_agent_stats(
            black_player=black_player,
            white_player=white_player,
            tracked_players=[fixed_tt_agent, id_tt_agent],
            flip_limit=flip_limit,
            verbose=verbose_each_game,
        )

        fixed_stats = stats[fixed_tt_agent]
        id_stats = stats[id_tt_agent]

        fixed_nodes.append(fixed_stats["nodes"])
        fixed_time.append(fixed_stats["time"])
        fixed_cutoffs.append(fixed_stats["cutoffs"])
        fixed_tt_hits.append(fixed_stats["tt_hits"])
        fixed_moves.append(fixed_stats["moves"])

        id_nodes.append(id_stats["nodes"])
        id_time.append(id_stats["time"])
        id_cutoffs.append(id_stats["cutoffs"])
        id_tt_hits.append(id_stats["tt_hits"])
        id_moves.append(id_stats["moves"])
        id_depths.append(id_tt_agent.stats.max_depth_reached)

        if result.winner == fixed_tt_color:
            fixed_tt_wins += 1
        elif result.winner == id_tt_color:
            id_tt_wins += 1
        else:
            draws += 1

        winner_str = "DRAW" if result.winner is None else (
            "BLACK" if result.winner == DiskColor.BLACK else "WHITE"
        )

        print(
            f"Game {game_idx + 1}: "
            f"Fixed TT as {'BLACK' if fixed_tt_color == DiskColor.BLACK else 'WHITE'}, "
            f"ID+TT as {'BLACK' if id_tt_color == DiskColor.BLACK else 'WHITE'} | "
            f"Black={result.black_disks}, White={result.white_disks}, "
            f"Winner={winner_str}"
        )

        print(
            f"  Fixed TT -> Nodes={fixed_stats['nodes']}, "
            f"Time={fixed_stats['time']:.4f}s, "
            f"Cutoffs={fixed_stats['cutoffs']}, "
            f"TT Hits={fixed_stats['tt_hits']}"
        )

        print(
            f"  ID+TT    -> Nodes={id_stats['nodes']}, "
            f"Time={id_stats['time']:.4f}s, "
            f"Cutoffs={id_stats['cutoffs']}, "
            f"TT Hits={id_stats['tt_hits']}, "
            f"Depth Reached={id_tt_agent.stats.max_depth_reached}"
        )

    def mean_std(values):
        return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0

    fixed_nodes_mean, fixed_nodes_std = mean_std(fixed_nodes)
    fixed_time_mean, fixed_time_std = mean_std(fixed_time)
    fixed_cutoffs_mean, fixed_cutoffs_std = mean_std(fixed_cutoffs)
    fixed_hits_mean, fixed_hits_std = mean_std(fixed_tt_hits)

    id_nodes_mean, id_nodes_std = mean_std(id_nodes)
    id_time_mean, id_time_std = mean_std(id_time)
    id_cutoffs_mean, id_cutoffs_std = mean_std(id_cutoffs)
    id_hits_mean, id_hits_std = mean_std(id_tt_hits)
    id_depth_mean, id_depth_std = mean_std(id_depths)

    fixed_avg_time_per_move = [
        t / m if m > 0 else 0
        for t, m in zip(fixed_time, fixed_moves)
    ]

    id_avg_time_per_move = [
        t / m if m > 0 else 0
        for t, m in zip(id_time, id_moves)
    ]

    fixed_tpm_mean, fixed_tpm_std = mean_std(fixed_avg_time_per_move)
    id_tpm_mean, id_tpm_std = mean_std(id_avg_time_per_move)

    print("\n--- Summary ---")
    print(f"Fixed TT wins: {fixed_tt_wins}")
    print(f"ID+TT wins:    {id_tt_wins}")
    print(f"Draws:         {draws}")

    print(f"\nFixed TT, depth = {fixed_depth} (mean +- std):")
    print(f"  Nodes:   {fixed_nodes_mean:.2f} +- {fixed_nodes_std:.2f}")
    print(f"  Cutoffs: {fixed_cutoffs_mean:.2f} +- {fixed_cutoffs_std:.2f}")
    print(f"  Time:    {fixed_time_mean:.6f} +- {fixed_time_std:.6f}")
    print(f"  Time/move: {fixed_tpm_mean:.6f} +- {fixed_tpm_std:.6f}")
    print(f"  TT Hits: {fixed_hits_mean:.2f} +- {fixed_hits_std:.2f}")

    print(f"\nIterative Deepening + TT, max depth = {id_max_depth} (mean +- std):")
    print(f"  Nodes:             {id_nodes_mean:.2f} +- {id_nodes_std:.2f}")
    print(f"  Cutoffs:           {id_cutoffs_mean:.2f} +- {id_cutoffs_std:.2f}")
    print(f"  Time:              {id_time_mean:.6f} +- {id_time_std:.6f}")
    print(f"  Time/move:         {id_tpm_mean:.6f} +- {id_tpm_std:.6f}")
    print(f"  TT Hits:           {id_hits_mean:.2f} +- {id_hits_std:.2f}")
    print(f"  Max depth reached: {id_depth_mean:.2f} +- {id_depth_std:.2f}")
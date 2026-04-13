from __future__ import annotations

import random
import numpy as np

from lfothello.ai import MinimaxAlphaBetaPlayer, RandomPlayer
from lfothello.game import Board
from lfothello.types import Cell, DiskColor


def test_case_1_initial_actions() -> None:
    print("=== TEST CASE 1: Initial legal moves ===")
    board = Board(flip_limit=2)
    board.display_state()
    actions = board.get_actions()
    print(f"Number of legal moves for BLACK: {len(actions)}")
    print("Moves:", actions)
    # Standard opening should have 4 legal moves.
    assert len(actions) == 4, "Expected 4 legal opening moves."


def test_case_2_apply_move() -> None:
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


def play_game_with_stats(
    black_player,
    white_player,
    baseline_player,
    flip_limit: int = 2,
    verbose: bool = False,
):
    """
    Play one full game and accumulate stats for the baseline player.
    """
    board = Board(flip_limit=flip_limit)

    baseline_total_nodes = 0
    baseline_total_cutoffs = 0
    baseline_total_time = 0.0
    baseline_total_tt_hits = 0

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

        if current_player is baseline_player and hasattr(current_player, "stats"):
            baseline_total_nodes += current_player.stats.nodes_expanded
            baseline_total_cutoffs += current_player.stats.cutoffs
            baseline_total_time += current_player.stats.elapsed_seconds

            if hasattr(current_player.stats, "tt_hits"):
                baseline_total_tt_hits += current_player.stats.tt_hits

        if move is None:
            board.state = board.pass_turn(board.state)
            continue

        board.make_move(move)

    if verbose:
        board.display_state()

    result = board.get_result()
    return (
        result,
        baseline_total_nodes,
        baseline_total_cutoffs,
        baseline_total_time,
        baseline_total_tt_hits,
    )


def experiment_baseline_vs_random(
    num_games: int = 10,
    depth: int = 3,
    flip_limit: int = 2,
    verbose_each_game: bool = False,
) -> None:
    print("=== EXPERIMENT: Baseline (no enhancements) vs Random (randomized roles) ===")

    baseline_wins = 0
    random_wins = 0
    draws = 0

    total_nodes = 0
    total_cutoffs = 0
    total_time = 0.0

    for game_idx in range(num_games):
        baseline_color = random.choice([DiskColor.BLACK, DiskColor.WHITE])

        baseline = MinimaxAlphaBetaPlayer(
            color=baseline_color,
            max_depth=depth,
            use_transposition_table=False,
            use_iterative_deepening=False,
        )

        random_color = DiskColor.WHITE if baseline_color == DiskColor.BLACK else DiskColor.BLACK
        random_player = RandomPlayer(random_color)

        if baseline_color == DiskColor.BLACK:
            black_player = baseline
            white_player = random_player
        else:
            black_player = random_player
            white_player = baseline

        result, game_nodes, game_cutoffs, game_time, _ = play_game_with_stats(
            black_player=black_player,
            white_player=white_player,
            baseline_player=baseline,
            flip_limit=flip_limit,
            verbose=verbose_each_game,
        )

        total_nodes += game_nodes
        total_cutoffs += game_cutoffs
        total_time += game_time

        if result.winner == baseline_color:
            baseline_wins += 1
        elif result.winner is None:
            draws += 1
        else:
            random_wins += 1

        print(
            f"Game {game_idx + 1}: "
            f"Baseline as {'BLACK' if baseline_color == DiskColor.BLACK else 'WHITE'} | "
            f"Black={result.black_disks}, White={result.white_disks}, "
            f"Winner={result.winner}, "
            f"Nodes={game_nodes}, Cutoffs={game_cutoffs}, Time={game_time:.4f}s"
        )

    print("\n--- Summary ---")
    print(f"Baseline wins: {baseline_wins}")
    print(f"Random wins:   {random_wins}")
    print(f"Draws:         {draws}")
    print(f"Baseline win rate: {baseline_wins / num_games:.2%}")
    print(f"Average nodes expanded per game: {total_nodes / num_games:.2f}")
    print(f"Average alpha-beta cutoffs per game: {total_cutoffs / num_games:.2f}")
    print(f"Average search time per game: {total_time / num_games:.4f}s")


def main() -> None:
    test_case_1_initial_actions()
    print()
    test_case_2_apply_move()
    print()
    test_case_3_limited_flip_rule()
    print()
    experiment_baseline_vs_random(
        num_games=10,
        depth=3,
        flip_limit=2,
        verbose_each_game=False
    )


if __name__ == "__main__":
    main()
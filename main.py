from __future__ import annotations

import numpy as np

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
    # board[3, 5] stays EMPTY

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


def main() -> None:
    test_case_1_initial_actions()
    print()
    test_case_2_apply_move()
    print()
    test_case_3_limited_flip_rule()


if __name__ == "__main__":
    main()
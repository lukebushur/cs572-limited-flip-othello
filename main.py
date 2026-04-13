from __future__ import annotations

from lfothello.game import Board


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


def main() -> None:
    test_case_1_initial_actions()
    print()
    test_case_2_apply_move()


if __name__ == "__main__":
    main()
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


def main() -> None:
    test_case_1_initial_actions()


if __name__ == "__main__":
    main()
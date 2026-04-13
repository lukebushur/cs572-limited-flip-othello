from __future__ import annotations

import random
from typing import Optional

from .game import Board
from .types import Action, DiskColor, State


class Player:
    def __init__(self, color: DiskColor) -> None:
        self.color = color

    def get_next_move(self, board: Board, state: State) -> Optional[Action]:
        raise NotImplementedError


class RandomPlayer(Player):
    def get_next_move(self, board: Board, state: State) -> Optional[Action]:
        actions = board.get_actions(state)
        if not actions:
            return None
        return random.choice(actions)


class CLIPlayer(Player):
    def get_next_move(self, board: Board, state: State) -> Optional[Action]:
        actions = board.get_actions(state)
        if not actions:
            return None

        while True:
            raw = input("Enter move as: row col\n> ").strip()
            try:
                row_str, col_str = raw.split()
                row, col = int(row_str), int(col_str)
                candidate = (row, col, self.color)
                if candidate in actions:
                    return candidate
                print("Illegal move. Legal moves are:")
                for act in actions:
                    print(f"  {act}")
            except ValueError:
                print("Invalid input. Example: 2 3")
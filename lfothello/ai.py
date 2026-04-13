from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional

from .game import Board
from .types import Action, BoardState, DiskColor, State


@dataclass
class SearchStats:
    nodes_expanded: int = 0
    cutoffs: int = 0
    max_depth_reached: int = 0
    elapsed_seconds: float = 0.0
    tt_hits: int = 0


@dataclass
class TTEntry:
    depth: int
    value: float
    best_action: Optional[Action]


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


class MinimaxAlphaBetaPlayer(Player):
    def __init__(
        self,
        color: DiskColor,
        max_depth: int = 4,
        use_transposition_table: bool = True,
        use_iterative_deepening: bool = True,
        time_limit_seconds: float = 2.0,
    ) -> None:
        super().__init__(color)
        self.max_depth = max_depth
        self.use_transposition_table = use_transposition_table
        self.use_iterative_deepening = use_iterative_deepening
        self.time_limit_seconds = time_limit_seconds

        self.stats = SearchStats()
        self._tt: dict[tuple[bytes, int, int], TTEntry] = {}
        self._move_order_table: dict[tuple[bytes, int], Action] = {}
        self._search_start_time = 0.0

    def reset_stats(self) -> None:
        self.stats = SearchStats()

    def clear_tt(self) -> None:
        self._tt.clear()
        self._move_order_table.clear()

    def get_next_move(self, board: Board, state: State) -> Optional[Action]:
        self.reset_stats()
        self._search_start_time = time.perf_counter()

        actions = board.get_actions(state)
        if not actions:
            self.stats.elapsed_seconds = time.perf_counter() - self._search_start_time
            return None

        if self.use_iterative_deepening:
            best_action = self._iterative_deepening_search(board, state)
        else:
            _, best_action = self._alphabeta(
                board=board,
                state=state,
                depth=self.max_depth,
                alpha=-math.inf,
                beta=math.inf,
                maximizing=(state[1] == self.color),
            )

        self.stats.elapsed_seconds = time.perf_counter() - self._search_start_time
        return best_action

    def _time_exceeded(self) -> bool:
        if not self.use_iterative_deepening:
            return False
        return (time.perf_counter() - self._search_start_time) >= self.time_limit_seconds

    def _board_key(self, board_state: BoardState, to_move: DiskColor, depth: int) -> tuple[bytes, int, int]:
        return (board_state.tobytes(), int(to_move), depth)

    def _move_order_key(
        self,
        board_state: BoardState,
        to_move: DiskColor,
    ) -> tuple[bytes, int]:
        return (board_state.tobytes(), int(to_move))

    def _ordered_actions(
        self,
        board: Board,
        state: State,
        actions: list[Action],
        tt_best_action: Optional[Action] = None,
    ) -> list[Action]:
        corners = {(0, 0), (0, 7), (7, 0), (7, 7)}

        def score(action: Action) -> tuple[int, int]:
            r, c, _ = action
            corner_bonus = 100 if (r, c) in corners else 0
            flip_bonus = len(board.get_flips_for_move(state, action))
            return (corner_bonus, flip_bonus)

        ordered = sorted(actions, key=score, reverse=True)

        if tt_best_action is not None and tt_best_action in ordered:
            ordered.remove(tt_best_action)
            ordered.insert(0, tt_best_action)

        return ordered

    def _iterative_deepening_search(self, board: Board, state: State) -> Optional[Action]:
        best_action: Optional[Action] = None
        depth = 1

        while depth <= self.max_depth:
            if self._time_exceeded():
                break

            try:
                value, action = self._alphabeta(
                    board=board,
                    state=state,
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing=(state[1] == self.color),
                )
                if action is not None:
                    best_action = action
                self.stats.max_depth_reached = max(self.stats.max_depth_reached, depth)
            except TimeoutError:
                break

            depth += 1

        if best_action is None:
            actions = board.get_actions(state)
            return actions[0] if actions else None

        return best_action

    def _alphabeta(
        self,
        board: Board,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> tuple[float, Optional[Action]]:
        if self._time_exceeded():
            raise TimeoutError

        self.stats.nodes_expanded += 1
        self.stats.max_depth_reached = max(
            self.stats.max_depth_reached,
            self.max_depth - depth if self.max_depth >= depth else 0,
        )

        board_state, to_move = state

        key = None
        move_key = None

        if self.use_transposition_table:
            key = self._board_key(board_state, to_move, depth)
            move_key = self._move_order_key(board_state, to_move)

            entry = self._tt.get(key)
            if entry is not None:
                self.stats.tt_hits += 1
                return entry.value, entry.best_action

        if depth == 0 or board.is_terminal(state):
            value = board.evaluate(state, self.color)
            if self.use_transposition_table and key is not None:
                self._tt[key] = TTEntry(depth=depth, value=value, best_action=None)
            return value, None

        actions = board.get_actions(state)

        if not actions:
            next_state = board.pass_turn(state)
            value, _ = self._alphabeta(
                board=board,
                state=next_state,
                depth=depth,
                alpha=alpha,
                beta=beta,
                maximizing=not maximizing,
            )
            if self.use_transposition_table and key is not None:
                self._tt[key] = TTEntry(depth=depth, value=value, best_action=None)
            return value, None

        tt_best_action = None
        if self.use_transposition_table and move_key is not None:
            tt_best_action = self._move_order_table.get(move_key)

        actions = self._ordered_actions(board, state, actions, tt_best_action=tt_best_action)

        best_action: Optional[Action] = None

        if maximizing:
            value = -math.inf
            for action in actions:
                child_state = board.apply_action(state, action)
                child_value, _ = self._alphabeta(
                    board=board,
                    state=child_state,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=False,
                )
                if child_value > value:
                    value = child_value
                    best_action = action
                alpha = max(alpha, value)
                if beta <= alpha:
                    self.stats.cutoffs += 1
                    break
        else:
            value = math.inf
            for action in actions:
                child_state = board.apply_action(state, action)
                child_value, _ = self._alphabeta(
                    board=board,
                    state=child_state,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=True,
                )
                if child_value < value:
                    value = child_value
                    best_action = action
                beta = min(beta, value)
                if beta <= alpha:
                    self.stats.cutoffs += 1
                    break

        if self.use_transposition_table:
            if key is not None:
                self._tt[key] = TTEntry(depth=depth, value=value, best_action=best_action)
            if move_key is not None and best_action is not None:
                self._move_order_table[move_key] = best_action

        return value, best_action
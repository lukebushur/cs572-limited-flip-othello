from lfothello.game import Board


def main():
    game = Board(None)
    game.display_state()
    game.get_actions()


if __name__ == "__main__":
    main()

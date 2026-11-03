"""Watch a saved Snake agent play without training it."""
import argparse
from pathlib import Path
import pygame
import torch
from neural_network.brain import Brain
from snake_game.snake_game import Main
from vizualizer.network import NetworkVisualizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).parent / 'snake_brain_1500.pth')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--games', type=int, default=5)
    parser.add_argument('--no-visualizer', action='store_true')
    args = parser.parse_args()
    brain = Brain()
    brain.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    brain.eval()
    game = Main()
    visualizer = None if args.no_visualizer else NetworkVisualizer(brain)
    clock = pygame.time.Clock()
    try:
        for episode in range(1, args.games + 1):
            state = game.reset()
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.WINDOWCLOSE:
                        if visualizer and event.window.id == visualizer.window.id:
                            visualizer.handle_event(event)
                        else:
                            return
                action = brain.choose_action(state, epsilon=0.0)
                if visualizer:
                    visualizer.update(state, action, 0, 0, episode, game.score)
                state, reward, done = game.step(action)
                game.draw()
                clock.tick(args.fps)
            print(f'Game {episode}: score {game.score}')
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()

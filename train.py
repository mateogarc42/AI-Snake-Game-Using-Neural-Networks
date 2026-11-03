"""Train a Snake agent with experience replay."""
import argparse
import os
import random
import time
from pathlib import Path
import pygame
import torch
from neural_network.brain import Brain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--fps', type=int, default=100, help='0 means uncapped')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-visualizer', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'checkpoints')
    args = parser.parse_args()
    if args.episodes < 1 or args.save_every < 1:
        parser.error('episodes and save-every must be positive')
    if args.headless:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    from snake_game.snake_game import Main
    from vizualizer.network import NetworkVisualizer
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    brain = Brain()
    game = Main()
    visualizer = None if args.headless or args.no_visualizer else NetworkVisualizer(brain)
    clock = pygame.time.Clock()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        for episode in range(1, args.episodes + 1):
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
                action = brain.choose_action(state)
                next_state, reward, done = game.step(action)
                brain.remember(state, action, reward, next_state, done)
                brain.learn()
                if visualizer:
                    visualizer.update(state, action, reward, brain.epsilon, episode, game.score)
                state = next_state
                if not args.headless:
                    game.draw()
                    clock.tick(args.fps)
            brain.epsilon = max(brain.epsilon_min, brain.epsilon * brain.epsilon_decay)
            print(f'Episode {episode}/{args.episodes}: score {game.score}, epsilon {brain.epsilon:.3f}', flush=True)
            if episode % args.save_every == 0:
                torch.save(brain.state_dict(), args.output / f'snake_brain_{episode}.pth')
    finally:
        torch.save(brain.state_dict(), args.output / 'snake_brain.pth')
        pygame.quit()
        print(f'Model saved in {args.output}. Elapsed: {time.time() - start:.1f}s')


if __name__ == '__main__':
    main()

# Neural Network Snake

Python, PyTorch, and Pygame Snake agent, based on https://github.com/Yadhu10nm/Snake_Game_using_NN.

## Setup on macOS

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

In VS Code select `.venv/bin/python` with **Python: Select Interpreter**.
Run and Debug includes **Watch trained Snake** and **Train Snake**.

## Watch the included model

```sh
.venv/bin/python play.py
```

Opens the game and neural network visualization. Defaults to five games using the included 1500-episode weights. Close the game to stop. Close just the visualization to continue playing.

## Train your own

```sh
.venv/bin/python train.py
.venv/bin/python train.py --headless --episodes 2000
.venv/bin/python play.py --model checkpoints/snake_brain.pth
```

Training saves weights every 500 episodes and on exit to `checkpoints/`. These are model weights, not full training-resume snapshots. `--fps 0` removes the rendering speed limit; headless training runs without windows. Run either script with `--help` for options.

The network maps 11 danger, direction, and food indicators through hidden layers of 32, 64, and 32 neurons to three actions: straight, right, left. Training uses epsilon-greedy exploration and replay memory. Episodes end on collision or after too many moves without food. Saved models may still perform poorly; training quality varies.

# <img src=".\img\logo.png"/>

`AlphaGoWeak` is go game bot based on deep learning. Her principle is similar to that of `AlphaGo` but can run in personal computer easily. I made her just for fun. She has some interesting specialties as follows:

- [x] Downloading, unpacking and parsing **SGF** (Smart Game Format) data automatically.
- [x] Fully **Win32**, **Linux**, **Mac** platform support.
- [x] Multiprocessing optimization.
- [x] **GPU** computing.
- [x] **GUI** support.
- [x] Support for **GTP** (Go Text Protocol).
- [x] Support for command line tools.
- [x] Experimental AlphaGo Zero-style MCTS self-play.
- [x] Support for **MCST** (Monte Carlo Search Tree).

## Model Structure

### AlphaGo Weak v1.0

The model structure and supervised training loop are implemented with [**PyTorch**](https://pytorch.org/). This kind of network structure is very efficient, and can work half a time.


<img src=".\img\model.png"/>


The input of chessboard grid contains 11 features, which is few but very representative. Quite a lot of scholars like a large number of feature layers, and this will make the training network get twice the result with half the effort. But I think too many feature layers will not only reduce the efficiency of network operation, but also exert too much human influence and make the network rigid. So I used a few feature layers. Although the accuracy of prediction is less than that of multi feature layer, the accuracy of policy network can still reach about 42% just after a few hours of training. The project also includes an experimental AlphaGo Zero-style MCTS self-play command with replay-buffer training, but it should not be treated as a validated full reinforcement-learning pipeline.

| Feature name      | Number of planes | Description                                                         |
|  ---------------  | ---------------- | ------------------------------------------------------------------- |
| Stone color       | 2                | Whether there is next player / opposite player's stone.             |
| Valid position    | 1                | Whether there is a valid position.                                  |
| Player Liberties  | 4                | (1,2,3,>=4) Number of liberties (empty adjacent points) of the player's stones have. |
| Opposite Liberties| 4                | (1,2,3,>=4) Number of liberties of the opposite player's stones have. |



## Training

1. Assume you are in `<project_folder>`.

2. Run command `pip install -e .` to install the package and the `alphago` command. This installs [PyTorch](https://pytorch.org/) and the other prerequisites. PyTorch will use CUDA automatically when a compatible GPU build is installed.
   
3. Run command `alphago download --type u-go` to download datasets from the Internet. If the network speed is too slow, you may need to use an accelerator. Mirroring is not supported currently.

4. Run command `alphago train --init 4096` to train your model. The `.pt` weights file is saved and loaded automatically.

5. Run `alphago self-play --steps 1 --epochs 1 --simulations 1 --max-moves 2 --record-games 1 --replay-buffer-size 128` to smoke-test the experimental AlphaGo Zero-style self-play loop, replay-buffer training, and one sample SGF.

6. More commands are not written because they are experimental at now.

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src=".\img\PolicyNetwork-Output_accuracy.svg"/>    <br>    <div style="color:#ffa500; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999999;    padding: 2px;">PolicyNetwork Accuracy</div> </center>


## Usage

Currently the project for `AlphaGoWeak` has not been finished, so it can only be executed by modifying the source code.


## Reference

- [**Leela Zero**](http://zero.sjeng.org/)
- [**KataGo**](https://github.com/lightvector/KataGo)
- [**GNU Go**](http://www.gnu.org/software/gnugo)
- [**Pachi**](https://github.com/pasky/pachi)
- [**Leela**](https://www.sjeng.org/leela.html)
- [**Sabaki**](https://github.com/SabakiHQ/Sabaki)

# ![AlphaGoWeak: A mini version  AlphaGo](.\img\logo.png)

`AlphaGoWeak` is go game bot based on deep learning. Her principle is similar to that of `AlphaGo` but can run in personal computer easily. I made her just for fun. She has some interesting specialties as follows:

- [x] Downloading, unpacking, extracting **SGF** (Smart Game Format) data automatically.
- [x] Fully **Win32**, **Linux**, **Mac** platform support.
- [x] Multiprocessing optimization.
- [x] **GPU** computing.
- [x] **GUI** support.
- [ ] Support for **GTP** (Go Text Protocol).
- [ ] Reinforcement learning.
- [ ] Support for **MCST** (Monte Carlo Search Tree).
- [ ] Support for command line tools.

## Model Structure

The model structure is generated by [**Keras**](https://keras.io/). This kind of network structure is very efficient, and can work half a time.


![model](.\img\model.png)


The input of chessboard grid contains 9 features, which is few but very representative. Quite a lot of scholars like a large number of feature layers, and this will make the training network get twice the result with half the effort. But I think too many feature layers will not only reduce the efficiency of network operation, but also exert too much human influence and make the network rigid. So I used a few feature layers. Although the accuracy is less than that of multi feature layer, the accuracy of policy network can reach about 8% just after a few hours of training, and the accuracy of value network can reach 60%. Considering that reinforcement learning will be carried out next, the loss of accuracy is acceptable.

| Feature name   | Number of planes | Description                                                  |
| -------------- | ---------------- | ------------------------------------------------------------ |
| Stone color    | 3                | Whether there is black stone/ white stone/ next player stone. |
| Valid position | 1                | Whether there is a valid position.                           |
| Sensibleness   | 1                | Whether a move does not fill its own eyes.                   |
| Liberties      | 4                | Number of liberties (empty adjacent points) of the string of stones this move belongs to. |




## Training

1. Assume you are in `<project_folder>`.

2. Run command `python go_data.py` to download data from Internet. If the network speed is too slow, you may need to use an accelerator because mirroring is not supported currently.

3. Run command `python alpha_go_weak.py` to train your model. The weights file is saved and loaded automatically in this way.

4. The default training logging directory is `<project_folder>/.data/models/alpha-go-weak/logs`. You can type `tensorboard --logdir <log_folder>` to see training history. 

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src=".\img\PolicyNetwork-Output_accuracy.svg"/>    <br>    <div style="color:#ffa500; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999999;    padding: 2px;">PolicyNetwork Accuracy</div> </center>

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src=".\img\ValueNetwork-Output_accuracy.svg"/>    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">ValueNetwork Accuracy</div> </center>



## Usage

Currently the project for `AlphaGoWeak` has not been finished, so it can only be executed by modifying the source code.


## Reference

- [**Leela Zero**](http://zero.sjeng.org/)
- [**KataGo**](https://github.com/lightvector/KataGo)
- [**GNU Go**](http://www.gnu.org/software/gnugo)
- [**Pachi**](https://github.com/pasky/pachi)
- [**Leela**](https://www.sjeng.org/leela.html)
- [**Sabaki**](https://github.com/SabakiHQ/Sabaki)
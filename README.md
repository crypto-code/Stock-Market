# Stock-Market
Stock Market Prediction &amp; Trading Bot using AI

## Stock Market Prediction using an LSTM Network
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work. They work tremendously well on a large variety of problems, and are now widely used. LSTMs are explicitly designed to avoid the vanishing gradient problem. 

<p align="center">
<img src="https://github.com/crypto-code/Stock-Market/blob/master/assets/lstm_model.png" align="middle" />  </p>

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer. LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

For more info check out this [article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Stock Market Agent using Evolution Strategy Agent

Even though the name sounds fancy but under the hood, it’s perhaps the simplest algorithm you can devise for exploring a landscape. Consider an agent in an environment (like Pong) that’s implemented via a neural network. It takes pixels in the input layer and outputs probabilities of actions available to it (move the paddle up, down or do nothing).

<p align="center">
<img src="https://github.com/crypto-code/Stock-Market/blob/master/assets/evolve_agent.png" align="middle" />  </p>

Our task in reinforcement learning is to find the parameters (weights and biases) of the neural network (weights and biases) that make the agent win more often and hence get more rewards. 

For more info check out this [article](https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f)

## Requirements
* Python 3.6.2 (https://www.python.org/downloads/release/python-362/)
* Numpy (https://pypi.org/project/numpy/)
* Tensorflow (https://pypi.org/project/tensorflow/)
* Keras (https://pypi.org/project/Keras/)
* Seaborn (https://pypi.org/project/seaborn/)
* Yahoo-Finance (https://pypi.org/project/yahoo-finance/)
* Pandas (https://pypi.org/project/pandas/)
* Matplotlib (https://pypi.org/project/matplotlib/)

## Usage
* For Stock Market Predictions run main-predict.py:
```
python main-predict.py --help

usage: main-predict.py [-h] --symbol SYMBOL [--period PERIOD]
                       [--epochs EPOCHS] [--sims SIMS]

Train Stock Market Predictor

optional arguments:
  -h, --help       show this help message and exit
  --symbol SYMBOL  Symbol of Stock to use
  --period PERIOD  Data period to download Valid periods are: nd, nmo, ny, max
                   (n is integer)
  --epochs EPOCHS  Number of training epochs
  --sims SIMS      Number of Simulations
```

* For Stock Market Agent run main-agent.py:
```
python main-agent.py --help

usage: main-agent.py [-h] --symbol SYMBOL [--period PERIOD] [--epochs EPOCHS]
                     [--initial INITIAL] [--skip SKIP]

Stock Market Agent

optional arguments:
  -h, --help         show this help message and exit
  --symbol SYMBOL    Symbol of Stock to use
  --period PERIOD    Data period to download Valid periods are: nd, nmo, ny,
                     max (n is integer)
  --epochs EPOCHS    Number of training epochs
  --initial INITIAL  Initial Money Available
  --skip SKIP        Number of days to skip in between
```

## Results

* Stock Market Prediction:
<p align="center">
<img src="https://github.com/crypto-code/Stock-Market/blob/master/assets/Prediction.png" align="middle" />  </p>

* Stock Market Agent:
<p align="center">
<img src="https://github.com/crypto-code/Stock-Market/blob/master/assets/Agent.png" align="middle" />  </p>



# G00D LUCK

For doubts email me at:
atinsaki@gmail.com

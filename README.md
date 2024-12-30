# BeyondTheRainbowICLR
Repository for the ICLR Paper "BeyondTheRainbow"

In this paper, we show Rainbow DQN extended with 6 extensions.

Here are our results on the Atari-5 and 60 game subsets:

<img width="1464" alt="IndivAblationsGithub" src="https://github.com/user-attachments/assets/5006518b-24e8-48da-94bb-a3f7b595c361">
<img width="644" alt="Indiv60Github" src="https://github.com/user-attachments/assets/40d9fdcd-f648-4f6e-816e-6e5ebd3053c3">

For our results, we provide a .csv file named results.csv, containing the results for each game. For each game, there are 200 evaluations, one for each million frames. Per evaluation, there are 100 episode scores.

In order to run our results, first install our environment via the requirements.txt file (all code was tested on Python 3.11.0):

run: pip install -r requirements.txt

Download your correct version of pytorch here: https://pytorch.org/

(We Use PyTorch version 2.1.2, with cuda v1.21)

After installing the environment, you can use main.py to perform runs. The default game is NameThisGame, however you can change this using the command line argument --game "GameName". (ie --game Breakout)

There are also many other command line arguments worth checking out, so have a look at main.py to see.

Also note by default we use 64 parallel environments, and even more for evaluation. This can be quite CPU and RAM intensive.
Furthermore, the full Replay Buffer and environments can use up to 45GB of RAM, so beware. If this is an issue, try reducing the number of evaluation environments, training environments or replay buffer size.

Also note that although we use a custom atari environment, this is exactly the same as the standard by default. We also however add a --life_info option, which passes a terminal to the agent on life loss, but does not reset the episode. Using this will drastically improve performance on games with lives.

# Report

The algorithm used for the project is Deep Deterministic Policy Gradient (DDPG), an Actor-Critic method based on https://arxiv.org/abs/1509.02971 

Following augmentations are made:

- Fixed-Q target
- Double network
- Soft-updates
- Experience replay. 

Ornstein–Uhlenbeck process-generated noise is also used.

### Hyperparameters

BUFFER_SIZE = int(1e5)  # replay buffer size

BATCH_SIZE = 64        # minibatch size

GAMMA = 0.99            # discount factor

TAU = 1e-3              # for soft update of target parameters

LR_ACTOR = 1e-4         # learning rate of the actor 

LR_CRITIC = 1e-4        # learning rate of the critic

WEIGHT_DECAY = 0        # L2 weight decay

### Neural Network Architectures:

NN architecture for the Actor:

Input nodes (33 nodes, based on state size)

Fully Connected Layer with ReLU activation (256 nodes)

Fully Connected Layer with ReLU activation (128 nodes)

Output nodes with tanh activation (4 nodes)

NN architecture for the Critic:

Input nodes (33 nodes, based on state size)

Fully Connected Layer with ReLU activation (256 nodes) in concatenation with 4 more nodes (action)

Fully Connected Layer with ReLU activation (128 nodes)

Output node (1 node)


### Plot of rewards

Agent is able to receive an average reward (over 100 episodes) of at least 30:

Environment solved in 106 episodes. 

Average score: 30.00

![alt text](https://github.com/AlperTekeli/udacity-continuous-control/blob/main/score.png)

### Ideas for Future Work

Following algorithms can also be used for this task and their performance can be compared to DDPG. 

- Proximal Policy Optimization (PPO)
- Trust Region Policy Optimization (TRPO)
- Truncated Natural Policy Gradient (TNPG)
- Distributed Distributional Deterministic Policy Gradients (D4PG) 

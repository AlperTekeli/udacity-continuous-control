# Report

The algorithm used for the project is Deep Deterministic Policy Gradient (DDPG), an Actor-Critic method based on https://arxiv.org/abs/1509.02971 

Here is the pseudo-code for the algorithm:

![alt text](https://github.com/AlperTekeli/udacity-continuous-control/blob/main/pseudo.png)


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

It was quite challenging to achieve stable learning with DDPG, especially with regards to hyperparameter tuning. 

Based on paper "Benchmarking Deep Reinforcement Learning for Continuous Control" - https://arxiv.org/pdf/1604.06778.pdf
Following algorithms can also be used for this task and their performance can be compared to DDPG. 

- Trust Region Policy Optimization (TRPO)
TRPO is a trust region-based policy gradient method.
"This algorithm allows more precise control on the expected policy improvement than TNPG through the introduction of a surrogate loss." (1)
Please refer to "Schulman, J., Levine, S., Abbeel, P., Jordan, M. I., and Moritz, P. Trust region policy optimization. In ICML, pp. 1889–1897,
2015a" for details regarding TRPO.

- Proximal Policy Optimization (PPO)
I am also eager to apply PPO, which is denoted as the default reinforcement learning algorithm at OpenAI because of its ease of use and good performance. - https://openai.com/blog/openai-baselines-ppo/

You may find the PPO paper here: https://arxiv.org/abs/1707.06347

"PPO strikes a balance between ease of implementation, sample complexity, and ease of tuning, trying to compute an update at each step that minimizes the cost function while ensuring the deviation from the previous policy is relatively small." (2)

- Distributed Distributional Deterministic Policy Gradients (D4PG) - ![paper](https://openreview.net/forum?id=SyZipzbCb)


### References
[1] "Benchmarking Deep Reinforcement Learning for Continuous Control" - https://arxiv.org/pdf/1604.06778.pdf
[2] https://openai.com/blog/openai-baselines-ppo/

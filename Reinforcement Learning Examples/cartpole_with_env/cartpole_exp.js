import * as sm from "@shumai/shumai"
import {CartPoleEnv} from './cart_pole_env.js';

const env = new CartPoleEnv()
const eps = Number.EPSILON

function randomGame () {
    // each episode is just a random play of the cart game
    for (let episode = 0; episode < 10; episode++) {
        env.reset()
        for (let t = 0; t < 100; t++) {
            // env.render()
            const action = env.action_space.sample()
            const [next_state, reward, done, info] = env.step(action)

            // console.log(t, next_state.valueOf(), reward, done, info, action)
            if (done) {
                break
            }
        }
    }
}

// randomGame()

// define dense layer

class Dense {
    constructor (inp_dim, out_dim, activation) {
        this.weight = sm.randn([inp_dim, out_dim])
        this.bias = sm.randn([1, out_dim])
        this.weight.requires_grad = true
        this.bias.requires_grad = true
    }

    forward(x) {
        x = this.weight.matmul(x)
        x = x.add(this.bias)
        x = activation(x)
        return x
    }
}

function relu(x) {
    return x.maximum(sm.scalar(0))
}

function mse(a, b) {
    const c = a.sub(b)
    return c.mul(c).mean()
}

class rlModel {
  constructor(input_shape, action_space) {
    let X_input = sm.randn([input_shape])
    let X = Dense(input_shape, 512, relu())(X_input)
    X = Dense(512, 256, relu())(X)
    X = Dense(256, 64, relu())(X)
    X = Dense(64, action_space, sm.linear())(X)
    return X
  }

  summary() {

  }

  compile() {
    const lr = 0.00025
    const optimize = (...args) => {
      const upd = (v) => {
        const o = v.detach().add(v.grad.detach().mul(sm.scalar(-lr)))
        o.eval()
        o.requires_grad = true
        v.grad = null
        return o
      }
      const opt = (l) => {
        for (let key in l) {
          const t = l[key]
          if (t.constructor === sm.Tensor && t.requires_grad) {
            l[key] = upd(t)
          }
        }
      }
      for (let a of args) {
        opt(a)
      }
    }

    const zero_grad = (...args) => {
      const zg = (l) => {
        for (let key in l) {
          const t = l[key]
          if (t.constructor === sm.Tensor && t.requires_grad) {
            t.grad = null
          }
        }
      }
      for (let a of args) {
        zg(a)
      }
    }

  }

  step() {
    zero_grad(l0, l1, l2)
    const t0 = performance.now()
    const y_hat = rlModel(x)
    const t1 = performance.now()
    const l = mse(y, y_hat)
    floss = l
    const t2 = performance.now()
    const stat = l.backward()
    const t3 = performance.now()
    optimize(l0, l1, l2)
    const t4 = performance.now()
    if (show_timing) {
      console.log('fwd', t1 - t0)
      console.log('mse', t2 - t1)
      console.log('bwd', t3 - t2)
      const tot = stat[4] - stat[0]
      const b0 = stat[1] - stat[0]
      const b1 = stat[2] - stat[1]
      const b2 = stat[3] - stat[2]
      const b3 = stat[4] - stat[3]
      console.log('  create jacobian', Math.round((100 * b0) / tot) + '%')
      console.log('  reverse graph', Math.round((100 * b1) / tot) + '%')
      console.log('  toposort', Math.round((100 * b2) / tot) + '%')
      console.log('  exec grad', Math.round((100 * b3) / tot) + '%')
      for (let k of Object.keys(stat[5])) {
        console.log('    ', k, ':', stat[5][k], '(count, ms)')
      }
      console.log('opt', t4 - t3)
      console.log('(whole step:', t4 - t0, ')')
      console.log('curr bytes used', Number(sm.bytesUsed()) / 1e6, 'MB')
    }
  }

}

const show_timing = false
let floss = 0



// def OurModel(input_shape, action_space):

//     model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

//     model.summary()
//     return model



function step() {


step()
const traint0 = performance.now()
for (let i = 0; i < iters; ++i) {
  step()
}
const traint1 = performance.now()
console.log('train at', iters / ((traint1 - traint0) / 1e3), 'iters/sec')
console.log('final loss', floss.toFloat32())

function remeber(state, action, reward, next_state, done) {
  this.memory.append((state, action, reward, next_state, done))
}

// def remember(self, state, action, reward, next_state, done):
//     self.memory.append((state, action, reward, next_state, done))
//     if len(self.memory) > self.train_start:
//         if self.epsilon > self.epsilon_min:
//             self.epsilon *= self.epsilon_decay

// def replay(self):
//     if len(self.memory) < self.train_start:
//         return
//     # Randomly sample minibatch from the memory
//     minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

//     state = np.zeros((self.batch_size, self.state_size))
//     next_state = np.zeros((self.batch_size, self.state_size))
//     action, reward, done = [], [], []

//     # do this before prediction
//     # for speedup, this could be done on the tensor level
//     # but easier to understand using a loop
//     for i in range(self.batch_size):
//         state[i] = minibatch[i][0]
//         action.append(minibatch[i][1])
//         reward.append(minibatch[i][2])
//         next_state[i] = minibatch[i][3]
//         done.append(minibatch[i][4])

//     # do batch prediction to save speed
//     target = self.model.predict(state)
//     target_next = self.model.predict(next_state)

//     for i in range(self.batch_size):
//         # correction on the Q value for the action used
//         if done[i]:
//             target[i][action[i]] = reward[i]
//         else:
//             # Standard - DQN
//             # DQN chooses the max Q value among next actions
//             # selection and evaluation of action is on the target Q Network
//             # Q_max = max_a' Q_target(s', a')
//             target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

//     # Train the Neural Network with batches
//     self.model.fit(state, target, batch_size=self.batch_size, verbose=0)




//     import os
//     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
//     import random
//     import gym
//     import numpy as np
//     from collections import deque
//     from keras.models import Model, load_model
//     from keras.layers import Input, Dense
//     from keras.optimizers import Adam, RMSprop


//     def OurModel(input_shape, action_space):
//         X_input = Input(input_shape)

//         # 'Dense' is the basic form of a neural network layer
//         # Input Layer of state size(4) and Hidden Layer with 512 nodes
//         X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

//         # Hidden layer with 256 nodes
//         X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

//         # Hidden layer with 64 nodes
//         X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

//         # Output Layer with # of actions: 2 nodes (left, right)
//         X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

//         model = Model(inputs = X_input, outputs = X, name='CartPole DQN model')
//         model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

//         model.summary()
//         return model

//     class DQNAgent:
//         def __init__(self):
//             self.env = gym.make('CartPole-v1')
//             # by default, CartPole-v1 has max episode steps = 500
//             self.state_size = self.env.observation_space.shape[0]
//             self.action_size = self.env.action_space.n
//             self.EPISODES = 1000
//             self.memory = deque(maxlen=2000)

//             self.gamma = 0.95    # discount rate
//             self.epsilon = 1.0  # exploration rate
//             self.epsilon_min = 0.001
//             self.epsilon_decay = 0.999
//             self.batch_size = 64
//             self.train_start = 1000

//             # create main model
//             self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)

//         def remember(self, state, action, reward, next_state, done):
//             self.memory.append((state, action, reward, next_state, done))
//             if len(self.memory) > self.train_start:
//                 if self.epsilon > self.epsilon_min:
//                     self.epsilon *= self.epsilon_decay

//         def act(self, state):
//             if np.random.random() <= self.epsilon:
//                 return random.randrange(self.action_size)
//             else:
//                 return np.argmax(self.model.predict(state))

//         def replay(self):
//             if len(self.memory) < self.train_start:
//                 return
//             # Randomly sample minibatch from the memory
//             minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

//             state = np.zeros((self.batch_size, self.state_size))
//             next_state = np.zeros((self.batch_size, self.state_size))
//             action, reward, done = [], [], []

//             # do this before prediction
//             # for speedup, this could be done on the tensor level
//             # but easier to understand using a loop
//             for i in range(self.batch_size):
//                 state[i] = minibatch[i][0]
//                 action.append(minibatch[i][1])
//                 reward.append(minibatch[i][2])
//                 next_state[i] = minibatch[i][3]
//                 done.append(minibatch[i][4])

//             # do batch prediction to save speed
//             target = self.model.predict(state)
//             target_next = self.model.predict(next_state)

//             for i in range(self.batch_size):
//                 # correction on the Q value for the action used
//                 if done[i]:
//                     target[i][action[i]] = reward[i]
//                 else:
//                     # Standard - DQN
//                     # DQN chooses the max Q value among next actions
//                     # selection and evaluation of action is on the target Q Network
//                     # Q_max = max_a' Q_target(s', a')
//                     target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

//             # Train the Neural Network with batches
//             self.model.fit(state, target, batch_size=self.batch_size, verbose=0)


//         def load(self, name):
//             self.model = load_model(name)

//         def save(self, name):
//             self.model.save(name)

//         def run(self):
//             for e in range(self.EPISODES):
//                 state = self.env.reset()
//                 state = np.reshape(state, [1, self.state_size])
//                 done = False
//                 i = 0
//                 while not done:
//                     self.env.render()
//                     action = self.act(state)
//                     next_state, reward, done, _ = self.env.step(action)
//                     next_state = np.reshape(next_state, [1, self.state_size])
//                     if not done or i == self.env._max_episode_steps-1:
//                         reward = reward
//                     else:
//                         reward = -100
//                     self.remember(state, action, reward, next_state, done)
//                     state = next_state
//                     i += 1
//                     if done:
//                         print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
//                         if i == 500:
//                             print("Saving trained model as cartpole-dqn.h5")
//                             self.save("cartpole-dqn.h5")
//                             return
//                     self.replay()

//         def test(self):
//             self.load("cartpole-dqn.h5")
//             for e in range(self.EPISODES):
//                 state = self.env.reset()
//                 state = np.reshape(state, [1, self.state_size])
//                 done = False
//                 i = 0
//                 while not done:
//                     self.env.render()
//                     action = np.argmax(self.model.predict(state))
//                     next_state, reward, done, _ = self.env.step(action)
//                     state = np.reshape(next_state, [1, self.state_size])
//                     i += 1
//                     if done:
//                         print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
//                         break

//     if __name__ == "__main__":
//         agent = DQNAgent()
//         agent.run()
//         #agent.test()

import * as tf from '@tensorflow/tfjs';

import {createDeepQNetwork} from './dqn';
import {getRandomAction, SnakeGame, NUM_ACTIONS, ALL_ACTIONS, getStateTensor} from './snake_game';
import {ReplayMemory} from './replay_memory';
import { assertPositiveInteger } from './utils';

export class SnakeGameAgent {
  constructor(game, config) {
    this.game = game;

    this.epsilonInit = config.epsilonInit;
    this.epsilonFinal = config.epsilonFinal;
    this.epsilonDecayFrames = config.epsilonDecayFrames;
    this.epsilonIncrement_ = (this.epsilonFinal - this.epsilonInit) /
        this.epsilonDecayFrames;

    this.onlineNetwork =
        createDeepQNetwork(game.height,  game.width, NUM_ACTIONS);
    this.targetNetwork =
        createDeepQNetwork(game.height,  game.width, NUM_ACTIONS);
    // Freeze taget network: it's weights are updated only through copying from
    // the online network.
    this.targetNetwork.trainable = false;

    this.optimizer = tf.train.adam(config.learningRate);

    this.replayBufferSize = config.replayBufferSize;
    this.replayMemory = new ReplayMemory(config.replayBufferSize);
    this.frameCount = 0;
    this.reset();
  }

  reset() {
    this.cumulativeReward_ = 0;
    this.fruitsEaten_ = 0;
    this.game.reset();
  }

  playStep() {
    this.epsilon = this.frameCount >= this.epsilonDecayFrames ?
        this.epsilonFinal :
        this.epsilonInit + this.epsilonIncrement_  * this.frameCount;
    this.frameCount++;

    // The epsilon-greedy algorithm.
    let action;
    const state = this.game.getState();
    if (Math.random() < this.epsilon) {
      // Pick an action at random.
      action = getRandomAction();
    } else {
      // Greedily pick an action based on online DQN output.
      tf.tidy(() => {
        const stateTensor =
            getStateTensor(state, this.game.height, this.game.width)
        action = ALL_ACTIONS[
            this.onlineNetwork.predict(stateTensor).argMax(-1).dataSync()[0]];
      });
    }

    const {state: nextState, reward, done, fruitEaten} = this.game.step(action);

    this.replayMemory.append([state, action, reward, done, nextState]);

    this.cumulativeReward_ += reward;
    if (fruitEaten) {
      this.fruitsEaten_++;
    }
    const output = {
      action,
      cumulativeReward: this.cumulativeReward_,
      done,
      fruitsEaten: this.fruitsEaten_
    };
    if (done) {
      this.reset();
    }
    return output;
  }

  trainOnReplayBatch(batchSize, gamma, optimizer) {
    // Get a batch of examples from the replay buffer.
    const batch = this.replayMemory.sample(batchSize);
    const lossFunction = () => tf.tidy(() => {
      const stateTensor = getStateTensor(
          batch.map(example => example[0]), this.game.height, this.game.width);
      const actionTensor = tf.tensor1d(
          batch.map(example => example[1]), 'int32');
      const qs = this.onlineNetwork.apply(stateTensor, {training: true})
          .mul(tf.oneHot(actionTensor, NUM_ACTIONS)).sum(-1);

      const rewardTensor = tf.tensor1d(batch.map(example => example[2]));
      const nextStateTensor = getStateTensor(
          batch.map(example => example[4]), this.game.height, this.game.width);
      const nextMaxQTensor =
          this.targetNetwork.predict(nextStateTensor).max(-1);
      const doneMask = tf.scalar(1).sub(
          tf.tensor1d(batch.map(example => example[3])).asType('float32'));
      const targetQs =
          rewardTensor.add(nextMaxQTensor.mul(doneMask).mul(gamma));
      return tf.losses.meanSquaredError(targetQs, qs);
    });

    // TODO(cais): Remove the second argument when `variableGrads()` obeys the
    // trainable flag.
    const grads =
        tf.variableGrads(lossFunction, this.onlineNetwork.getWeights());
    optimizer.applyGradients(grads.grads);
    tf.dispose(grads);
    // TODO(cais): Return the loss value here?
  }
}

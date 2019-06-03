import * as fs from 'fs';

import * as argparse from 'argparse';
import {mkdir} from 'shelljs';

let tf;

import {SnakeGameAgent} from './agent';
import {copyWeights} from './dqn';
import {SnakeGame} from './snake_game';

class MovingAverager {
  constructor(bufferLength) {
    this.buffer = [];
    for (let i = 0; i < bufferLength; ++i) {
      this.buffer.push(null);
    }
  }

  append(x) {
    this.buffer.shift();
    this.buffer.push(x);
  }

  average() {
    return this.buffer.reduce((x, prev) => x + prev) / this.buffer.length;
  }
}

export async function train(
    agent, batchSize, gamma, learningRate, cumulativeRewardThreshold,
    maxNumFrames, syncEveryFrames, savePath, logDir) {
  let summaryWriter;
  if (logDir != null) {
    summaryWriter = tf.node.summaryFileWriter(logDir);
  }

  for (let i = 0; i < agent.replayBufferSize; ++i) {
    agent.playStep();
  }

  // Moving averager: cumulative reward across 100 most recent 100 episodes.
  const rewardAverager100 = new MovingAverager(100);
  // Moving averager: fruits eaten across 100 most recent 100 episodes.
  const eatenAverager100 = new MovingAverager(100);

  const optimizer = tf.train.adam(learningRate);
  let tPrev = new Date().getTime();
  let frameCountPrev = agent.frameCount;
  let averageReward100Best = -Infinity;
  while (true) {
    agent.trainOnReplayBatch(batchSize, gamma, optimizer);
    const {cumulativeReward, done, fruitsEaten} = agent.playStep();
    if (done) {
      const t = new Date().getTime();
      const framesPerSecond =
          (agent.frameCount - frameCountPrev) / (t - tPrev) * 1e3;
      tPrev = t;
      frameCountPrev = agent.frameCount;

      rewardAverager100.append(cumulativeReward);
      eatenAverager100.append(fruitsEaten);
      const averageReward100 = rewardAverager100.average();
      const averageEaten100 = eatenAverager100.average();

      console.log(
          `Frame #${agent.frameCount}: ` +
          `cumulativeReward100=${averageReward100.toFixed(1)}; ` +
          `eaten100=${averageEaten100.toFixed(2)} ` +
          `(epsilon=${agent.epsilon.toFixed(3)}) ` +
          `(${framesPerSecond.toFixed(1)} frames/s)`);
      if (summaryWriter != null) {
        summaryWriter.scalar(
            'cumulativeReward100', averageReward100, agent.frameCount);
        summaryWriter.scalar('eaten100', averageEaten100, agent.frameCount);
        summaryWriter.scalar('epsilon', agent.epsilon, agent.frameCount);
        summaryWriter.scalar(
            'framesPerSecond', framesPerSecond, agent.frameCount);
      }
      if (averageReward100 >= cumulativeRewardThreshold ||
          agent.frameCount >= maxNumFrames) {
        // TODO(cais): Save online network.
        break;
      }
      if (averageReward100 > averageReward100Best) {
        averageReward100Best = averageReward100;
        if (savePath != null) {
          if (!fs.existsSync(savePath)) {
            mkdir('-p', savePath);
          }
          await agent.onlineNetwork.save(`file://${savePath}`);
          console.log(`Saved DQN to ${savePath}`);
        }
      }
    }
    if (agent.frameCount % syncEveryFrames === 0) {
      copyWeights(agent.targetNetwork, agent.onlineNetwork);
      console.log('Sync\'ed weights from online network to target network');
    }
  }
}

export function parseArguments() {
  const parser = new argparse.ArgumentParser({
    description: 'Training script for a DQN that plays the snake game'
  });
  parser.addArgument('--savePath', {
    type: 'string',
    defaultValue: './models/dqn',
    help: 'File path to which the online DQN will be saved after training.'
  });
  parser.addArgument('--logDir', {
    type: 'string',
    defaultValue: null,
    help: 'Path to the directory for writing TensorBoard logs in.'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArguments();
  tf = require('@tensorflow/tfjs-node');
  console.log(`args: ${JSON.stringify(args, null, 2)}`);

  const game = new SnakeGame({
    height: 5,
    width: 5,
    numFruits: 1,
    initLen: 2
  });
  const agent = new SnakeGameAgent(game, {
    replayBufferSize: 5000,
    epsilonInit: 0.8,
    epsilonFinal: 0.01,
    epsilonDecayFrames: 50000
  });

  let batchSize = 64;
  let gamma = 0.99;
  let lr = 1e-3;
  let network_sync = 500;
  let r_threshold = 100;
  let train_steps = 50000;
  await train(
      agent, batchSize, gamma, lr,
      r_threshold, train_steps,
      network_sync, args.savePath, args.logDir);
}

if (require.main === module) {
  main();
}

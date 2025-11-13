/**
 * GhostEncoder Neural Network Layers
 * Production-grade layer implementations
 * 
 * @module ghostencoder/layers
 * @author Ghost Key Team
 * @license MIT
 */

import { Tensor } from './tensor.js';
import { getActivation, getActivationDerivative } from './activations.js';

/**
 * Dense (Fully Connected) Layer
 */
export class Dense {
  constructor(inputSize, outputSize, options = {}) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activation = options.activation || 'relu';
    this.useBias = options.useBias !== undefined ? options.useBias : true;
    
    // Initialize weights with He initialization (good for ReLU)
    this.weights = Tensor.heRandom([inputSize, outputSize]);
    
    // Initialize biases to small positive values
    this.biases = this.useBias ? Tensor.ones([outputSize]).multiply(0.01) : null;
    
    // For training
    this.input = null;
    this.output = null;
    this.weightGradients = null;
    this.biasGradients = null;
    
    // Activation functions
    this.activationFn = getActivation(this.activation);
    this.activationDerivative = getActivationDerivative(this.activation);
  }

  /**
   * Forward pass
   */
  forward(input) {
    this.input = input;
    
    // Linear transformation: output = input @ weights + bias
    let output = input.matmul(this.weights);
    
    if (this.useBias) {
      // Broadcast bias addition
      const biasData = new Float32Array(output.size);
      const [batchSize, features] = output.shape;
      
      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < features; j++) {
          biasData[i * features + j] = output.data[i * features + j] + this.biases.data[j];
        }
      }
      
      output = new Tensor(biasData, output.shape);
    }
    
    // Apply activation
    this.output = output.apply(this.activationFn);
    
    return this.output;
  }

  /**
   * Backward pass
   */
  backward(outputGradient, learningRate) {
    // Apply activation derivative
    const activationGrad = this.output.apply(this.activationDerivative);
    const delta = outputGradient.multiply(activationGrad);
    
    // Compute weight gradients: input^T @ delta
    this.weightGradients = this.input.transpose().matmul(delta);
    
    // Compute bias gradients
    if (this.useBias) {
      this.biasGradients = delta.sum(0);
    }
    
    // Compute input gradient: delta @ weights^T
    const inputGradient = delta.matmul(this.weights.transpose());
    
    // Update weights and biases
    this.weights = this.weights.subtract(this.weightGradients.multiply(learningRate));
    
    if (this.useBias) {
      this.biases = this.biases.subtract(this.biasGradients.multiply(learningRate));
    }
    
    return inputGradient;
  }

  /**
   * Get parameters
   */
  getParameters() {
    return {
      weights: this.weights.clone(),
      biases: this.useBias ? this.biases.clone() : null
    };
  }

  /**
   * Set parameters
   */
  setParameters(params) {
    this.weights = params.weights.clone();
    if (this.useBias && params.biases) {
      this.biases = params.biases.clone();
    }
  }
}

/**
 * Batch Normalization Layer
 */
export class BatchNormalization {
  constructor(numFeatures, options = {}) {
    this.numFeatures = numFeatures;
    this.epsilon = options.epsilon || 1e-5;
    this.momentum = options.momentum || 0.9;
    
    // Learnable parameters
    this.gamma = Tensor.ones([numFeatures]);
    this.beta = Tensor.zeros([numFeatures]);
    
    // Running statistics
    this.runningMean = Tensor.zeros([numFeatures]);
    this.runningVar = Tensor.ones([numFeatures]);
    
    // For training
    this.input = null;
    this.normalized = null;
    this.mean = null;
    this.variance = null;
    this.gammaGradients = Tensor.zeros([numFeatures]);
    this.betaGradients = Tensor.zeros([numFeatures]);
  }

  /**
   * Forward pass
   */
  forward(input, training = true) {
    this.input = input;
    const [batchSize, features] = input.shape;
    
    if (training) {
      // Compute batch statistics
      this.mean = input.mean(0);
      
      // Compute variance
      const centered = new Float32Array(input.size);
      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < features; j++) {
          const idx = i * features + j;
          centered[idx] = input.data[idx] - this.mean.data[j];
        }
      }
      
      const centeredTensor = new Tensor(centered, input.shape);
      this.variance = centeredTensor.multiply(centeredTensor).mean(0);
      
      // Update running statistics
      this.runningMean = this.runningMean.multiply(this.momentum)
        .add(this.mean.multiply(1 - this.momentum));
      this.runningVar = this.runningVar.multiply(this.momentum)
        .add(this.variance.multiply(1 - this.momentum));
    } else {
      this.mean = this.runningMean;
      this.variance = this.runningVar;
    }
    
    // Normalize
    const normalized = new Float32Array(input.size);
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < features; j++) {
        const idx = i * features + j;
        const std = Math.sqrt(this.variance.data[j] + this.epsilon);
        normalized[idx] = (input.data[idx] - this.mean.data[j]) / std;
      }
    }
    
    this.normalized = new Tensor(normalized, input.shape);
    
    // Scale and shift
    const output = new Float32Array(input.size);
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < features; j++) {
        const idx = i * features + j;
        output[idx] = this.gamma.data[j] * this.normalized.data[idx] + this.beta.data[j];
      }
    }
    
    return new Tensor(output, input.shape);
  }

  /**
   * Backward pass
   */
  backward(outputGradient, learningRate) {
    const [batchSize, features] = outputGradient.shape;
    const gradData = outputGradient.data;
    const inputData = this.input.data;
    const meanData = this.mean.data;
    const varianceData = this.variance.data;
    const normalizedData = this.normalized.data;
    const gammaData = this.gamma.data;

    const gammaGrad = new Float32Array(features);
    const betaGrad = new Float32Array(features);
    const dx = new Float32Array(outputGradient.size);

    for (let j = 0; j < features; j++) {
      let dgamma = 0;
      let dbeta = 0;
      for (let i = 0; i < batchSize; i++) {
        const idx = i * features + j;
        const grad = gradData[idx];
        dgamma += grad * normalizedData[idx];
        dbeta += grad;
      }
      gammaGrad[j] = dgamma;
      betaGrad[j] = dbeta;
    }

    const invBatch = 1 / batchSize;

    for (let j = 0; j < features; j++) {
      const invStd = 1 / Math.sqrt(varianceData[j] + this.epsilon);
      let sumDxhat = 0;
      let sumDxhatXm = 0;

      for (let i = 0; i < batchSize; i++) {
        const idx = i * features + j;
        const xm = inputData[idx] - meanData[j];
        const dxhat = gradData[idx] * gammaData[j];
        sumDxhat += dxhat;
        sumDxhatXm += dxhat * xm;
      }

      const coeff = sumDxhat * invBatch;
      const coeff2 = sumDxhatXm * invBatch * invStd * invStd;

      for (let i = 0; i < batchSize; i++) {
        const idx = i * features + j;
        const xm = inputData[idx] - meanData[j];
        const dxhat = gradData[idx] * gammaData[j];
        dx[idx] = invStd * (dxhat - coeff - xm * coeff2);
      }
    }

    this.gamma = this.gamma.subtract(new Tensor(gammaGrad, [features]).multiply(learningRate));
    this.beta = this.beta.subtract(new Tensor(betaGrad, [features]).multiply(learningRate));

    this.gammaGradients = new Tensor(gammaGrad, [features]);
    this.betaGradients = new Tensor(betaGrad, [features]);

    return new Tensor(dx, outputGradient.shape);
  }
}

/**
 * Dropout Layer (for regularization)
 */
export class Dropout {
  constructor(rate = 0.5) {
    this.rate = rate;
    this.mask = null;
  }

  /**
   * Forward pass
   */
  forward(input, training = true) {
    if (!training) {
      return input;
    }
    
    // Create dropout mask
    const mask = new Float32Array(input.size);
    const scale = 1 / (1 - this.rate);
    
    for (let i = 0; i < input.size; i++) {
      mask[i] = Math.random() > this.rate ? scale : 0;
    }
    
    this.mask = new Tensor(mask, input.shape);
    
    return input.multiply(this.mask);
  }

  /**
   * Backward pass
   */
  backward(outputGradient) {
    if (!this.mask) {
      return outputGradient;
    }
    return outputGradient.multiply(this.mask);
  }
}

/**
 * Residual Connection (Skip Connection)
 */
export class ResidualBlock {
  constructor(layers) {
    this.layers = layers;
    this.input = null;
  }

  /**
   * Forward pass
   */
  forward(input, training = true) {
    this.input = input;
    let output = input;
    
    for (const layer of this.layers) {
      output = layer.forward(output, training);
    }
    
    // Add skip connection
    return output.add(this.input);
  }

  /**
   * Backward pass
   */
  backward(outputGradient, learningRate) {
    let gradient = outputGradient;
    
    for (let i = this.layers.length - 1; i >= 0; i--) {
      gradient = this.layers[i].backward(gradient, learningRate);
    }
    
    // Add gradient from skip connection
    return gradient.add(outputGradient);
  }
}

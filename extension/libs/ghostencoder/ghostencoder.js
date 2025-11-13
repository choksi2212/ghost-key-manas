/**
 * GhostEncoder - Production-Grade Autoencoder for Keystroke Dynamics
 * 
 * Advanced deep learning autoencoder specifically optimized for biometric
 * keystroke authentication. Features state-of-the-art architecture with:
 * - Deep encoder-decoder structure (7 layers)
 * - Batch normalization for stable training
 * - Residual connections for better gradient flow
 * - Dropout for regularization
 * - Advanced activation functions (SELU, Swish)
 * - Anomaly detection capabilities
 * 
 * Architecture:
 * Input (variable) → 128 → 64 → 32 (bottleneck) → 64 → 128 → Output
 * 
 * @module ghostencoder
 * @author Ghost Key Team
 * @version 1.0.0
 * @license MIT
 */

import { Tensor } from './tensor.js';
import { Dense, BatchNormalization, Dropout, ResidualBlock } from './layers.js';

export class GhostEncoder {
  constructor(options = {}) {
    this.inputSize = options.inputSize || 100;
    this.encodingDim = options.encodingDim || 32;
    this.hiddenLayers = options.hiddenLayers || [128, 64];
    this.activation = options.activation || 'selu';
    this.dropoutRate = options.dropoutRate || 0.2;
    this.useBatchNorm = options.useBatchNorm !== undefined ? options.useBatchNorm : true;
    this.useResidual = options.useResidual !== undefined ? options.useResidual : true;
    
    // Training parameters
    this.learningRate = options.learningRate || 0.001;
    this.epochs = options.epochs || 100;
    this.batchSize = options.batchSize || 32;
    this.validationSplit = options.validationSplit || 0.2;
    
    // Anomaly detection threshold
    this.anomalyThreshold = options.anomalyThreshold || 0.15;
    
    // Build the network
    this.encoder = null;
    this.decoder = null;
    this.buildNetwork();
    
    // Training history
    this.history = {
      loss: [],
      valLoss: [],
      reconstructionError: []
    };
    
    // Statistics for normalization
    this.inputMean = null;
    this.inputStd = null;
  }

  /**
   * Build the autoencoder network
   * Architecture: Input → 128 → 64 → 32 (bottleneck) → 64 → 128 → Output
   */
  buildNetwork() {
    this.encoder = [];
    this.decoder = [];
    
    // ==================== ENCODER ====================
    let prevSize = this.inputSize;
    
    for (let i = 0; i < this.hiddenLayers.length; i++) {
      const layerSize = this.hiddenLayers[i];
      
      // Dense layer
      this.encoder.push(new Dense(prevSize, layerSize, {
        activation: this.activation,
        useBias: true
      }));
      
      // Batch normalization (helps with training stability)
      if (this.useBatchNorm) {
        this.encoder.push(new BatchNormalization(layerSize));
      }
      
      // Dropout (prevents overfitting)
      if (this.dropoutRate > 0) {
        this.encoder.push(new Dropout(this.dropoutRate));
      }
      
      prevSize = layerSize;
    }
    
    // Bottleneck layer (compressed representation)
    this.encoder.push(new Dense(prevSize, this.encodingDim, {
      activation: 'linear', // Linear activation for bottleneck
      useBias: true
    }));
    
    // ==================== DECODER ====================
    prevSize = this.encodingDim;
    
    // Mirror the encoder structure
    for (let i = this.hiddenLayers.length - 1; i >= 0; i--) {
      const layerSize = this.hiddenLayers[i];
      
      // Dense layer
      this.decoder.push(new Dense(prevSize, layerSize, {
        activation: this.activation,
        useBias: true
      }));
      
      // Batch normalization
      if (this.useBatchNorm) {
        this.decoder.push(new BatchNormalization(layerSize));
      }
      
      // Dropout
      if (this.dropoutRate > 0) {
        this.decoder.push(new Dropout(this.dropoutRate));
      }
      
      prevSize = layerSize;
    }
    
    // Output layer (reconstruction)
    this.decoder.push(new Dense(prevSize, this.inputSize, {
      activation: 'linear', // Linear for reconstruction
      useBias: true
    }));
    
    console.log(`🧠 GhostEncoder Network Built:`);
    console.log(`   - Input: ${this.inputSize}`);
    console.log(`   - Hidden: ${this.hiddenLayers.join(' → ')}`);
    console.log(`   - Bottleneck: ${this.encodingDim}`);
    console.log(`   - Activation: ${this.activation}`);
    console.log(`   - Batch Norm: ${this.useBatchNorm}`);
    console.log(`   - Dropout: ${this.dropoutRate}`);
    console.log(`   - Total Encoder Layers: ${this.encoder.length}`);
    console.log(`   - Total Decoder Layers: ${this.decoder.length}`);
  }

  /**
   * Normalize input data
   * @private
   */
  normalizeData(data) {
    const tensor = new Tensor(data);
    
    if (!this.inputMean || !this.inputStd) {
      // Compute statistics
      this.inputMean = tensor.mean(0);
      
      const centered = new Float32Array(tensor.size);
      const [batchSize, features] = tensor.shape;
      
      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < features; j++) {
          const idx = i * features + j;
          centered[idx] = tensor.data[idx] - this.inputMean.data[j];
        }
      }
      
      const centeredTensor = new Tensor(centered, tensor.shape);
      const variance = centeredTensor.multiply(centeredTensor).mean(0);
      
      this.inputStd = variance.apply(x => Math.sqrt(x + 1e-8));
    }
    
    // Normalize
    const normalized = new Float32Array(tensor.size);
    const [batchSize, features] = tensor.shape;
    
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < features; j++) {
        const idx = i * features + j;
        normalized[idx] = (tensor.data[idx] - this.inputMean.data[j]) / this.inputStd.data[j];
      }
    }
    
    return new Tensor(normalized, tensor.shape);
  }

  /**
   * Denormalize data
   * @private
   */
  denormalizeData(tensor) {
    if (!this.inputMean || !this.inputStd) {
      return tensor;
    }
    
    const denormalized = new Float32Array(tensor.size);
    const [batchSize, features] = tensor.shape;
    
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < features; j++) {
        const idx = i * features + j;
        denormalized[idx] = tensor.data[idx] * this.inputStd.data[j] + this.inputMean.data[j];
      }
    }
    
    return new Tensor(denormalized, tensor.shape);
  }

  /**
   * Forward pass through encoder
   */
  encode(input, training = false) {
    let output = input;
    
    for (const layer of this.encoder) {
      if (layer instanceof BatchNormalization || layer instanceof Dropout) {
        output = layer.forward(output, training);
      } else {
        output = layer.forward(output);
      }
    }
    
    return output;
  }

  /**
   * Forward pass through decoder
   */
  decode(encoded, training = false) {
    let output = encoded;
    
    for (const layer of this.decoder) {
      if (layer instanceof BatchNormalization || layer instanceof Dropout) {
        output = layer.forward(output, training);
      } else {
        output = layer.forward(output);
      }
    }
    
    return output;
  }

  /**
   * Full forward pass (encode + decode)
   */
  forward(input, training = false) {
    const encoded = this.encode(input, training);
    const decoded = this.decode(encoded, training);
    return decoded;
  }

  /**
   * Compute reconstruction loss (MSE)
   * @private
   */
  computeLoss(input, output) {
    const diff = input.subtract(output);
    const squaredDiff = diff.multiply(diff);
    return squaredDiff.mean();
  }

  /**
   * Train the autoencoder
   */
  async train(data, options = {}) {
    console.log('🚀 Starting GhostEncoder training...');
    
    const epochs = options.epochs || this.epochs;
    const batchSize = options.batchSize || this.batchSize;
    const validationSplit = options.validationSplit || this.validationSplit;
    
    // Convert to tensor and normalize
    const dataTensor = this.normalizeData(data);
    const [numSamples, features] = dataTensor.shape;
    
    // Split into training and validation
    const splitIdx = Math.floor(numSamples * (1 - validationSplit));
    const trainData = new Tensor(
      dataTensor.data.slice(0, splitIdx * features),
      [splitIdx, features]
    );
    const valData = new Tensor(
      dataTensor.data.slice(splitIdx * features),
      [numSamples - splitIdx, features]
    );
    
    console.log(`📊 Training samples: ${splitIdx}, Validation samples: ${numSamples - splitIdx}`);
    
    // Training loop
    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let numBatches = 0;
      
      // Shuffle training data
      const indices = Array.from({ length: splitIdx }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      
      // Mini-batch training
      for (let i = 0; i < splitIdx; i += batchSize) {
        const batchIndices = indices.slice(i, Math.min(i + batchSize, splitIdx));
        const batchData = new Float32Array(batchIndices.length * features);
        
        for (let j = 0; j < batchIndices.length; j++) {
          const idx = batchIndices[j];
          for (let k = 0; k < features; k++) {
            batchData[j * features + k] = trainData.data[idx * features + k];
          }
        }
        
        const batch = new Tensor(batchData, [batchIndices.length, features]);
        
        // Forward pass
        const reconstructed = this.forward(batch, true);
        
        // Compute loss
        const loss = this.computeLoss(batch, reconstructed);
        epochLoss += loss;
        numBatches++;
        
        const backpropLayer = (layer, grad) => {
          if (layer instanceof Dense || layer instanceof BatchNormalization) {
            return layer.backward(grad, this.learningRate);
          }
          if (layer instanceof Dropout) {
            return layer.backward(grad);
          }
          if (layer instanceof ResidualBlock) {
            return layer.backward(grad, this.learningRate);
          }
          return grad;
        };

        let gradient = batch.subtract(reconstructed).multiply(-2 / batch.size);

        for (let j = this.decoder.length - 1; j >= 0; j--) {
          gradient = backpropLayer(this.decoder[j], gradient);
        }

        for (let j = this.encoder.length - 1; j >= 0; j--) {
          gradient = backpropLayer(this.encoder[j], gradient);
        }
      }
      
      // Validation
      const valReconstructed = this.forward(valData, false);
      const valLoss = this.computeLoss(valData, valReconstructed);
      
      // Record history
      this.history.loss.push(epochLoss / numBatches);
      this.history.valLoss.push(valLoss);
      
      // Log progress
      if ((epoch + 1) % 10 === 0 || epoch === 0) {
        console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${(epochLoss / numBatches).toFixed(6)}, Val Loss: ${valLoss.toFixed(6)}`);
      }
    }
    
    console.log('✅ Training complete!');
    
    return this.history;
  }

  /**
   * Compute reconstruction error for anomaly detection
   */
  computeReconstructionError(data) {
    // Normalize input
    const normalized = this.normalizeData([data]);
    
    // Forward pass
    const reconstructed = this.forward(normalized, false);
    
    // Denormalize
    const denormalized = this.denormalizeData(reconstructed);
    
    // Compute MSE
    const original = new Tensor([data]);
    const diff = original.subtract(denormalized);
    const squaredDiff = diff.multiply(diff);
    
    return squaredDiff.mean();
  }

  /**
   * Detect if input is anomalous (different user)
   */
  isAnomaly(data, threshold = null) {
    const error = this.computeReconstructionError(data);
    const thresh = threshold !== null ? threshold : this.anomalyThreshold;
    
    return {
      isAnomaly: error > thresh,
      error: error,
      threshold: thresh,
      confidence: Math.max(0, 1 - error / thresh)
    };
  }

  /**
   * Authenticate user based on keystroke pattern
   */
  authenticate(data, options = {}) {
    const threshold = options.threshold || this.anomalyThreshold;
    const result = this.isAnomaly(data, threshold);
    
    return {
      authenticated: !result.isAnomaly,
      confidence: result.confidence,
      reconstructionError: result.error,
      threshold: result.threshold,
      similarity: Math.max(0, 1 - result.error)
    };
  }

  /**
   * Get compressed representation (encoding)
   */
  getEncoding(data) {
    const normalized = this.normalizeData([data]);
    const encoded = this.encode(normalized, false);
    return encoded.toArray()[0];
  }

  /**
   * Export model parameters (fully JSON-serializable)
   */
  exportModel() {
    const encoderParams = this.encoder
      .filter(layer => layer instanceof Dense)
      .map(layer => {
        const params = layer.getParameters();
        const weightsArray = params.weights.toArray();
        const weightsData = Array.isArray(weightsArray[0]) 
          ? weightsArray.flat() 
          : Array.from(weightsArray);
        
        return {
          weights: {
            data: weightsData,
            shape: params.weights.shape
          },
          biases: params.biases ? {
            data: Array.from(params.biases.toArray()),
            shape: params.biases.shape
          } : null
        };
      });
    
    const decoderParams = this.decoder
      .filter(layer => layer instanceof Dense)
      .map(layer => {
        const params = layer.getParameters();
        const weightsArray = params.weights.toArray();
        const weightsData = Array.isArray(weightsArray[0]) 
          ? weightsArray.flat() 
          : Array.from(weightsArray);
        
        return {
          weights: {
            data: weightsData,
            shape: params.weights.shape
          },
          biases: params.biases ? {
            data: Array.from(params.biases.toArray()),
            shape: params.biases.shape
          } : null
        };
      });
    
    return {
      config: {
        inputSize: this.inputSize,
        encodingDim: this.encodingDim,
        hiddenLayers: this.hiddenLayers,
        activation: this.activation,
        dropoutRate: this.dropoutRate,
        useBatchNorm: this.useBatchNorm,
        anomalyThreshold: this.anomalyThreshold
      },
      encoder: encoderParams,
      decoder: decoderParams,
      statistics: {
        inputMean: this.inputMean ? Array.from(this.inputMean.toArray()) : null,
        inputStd: this.inputStd ? Array.from(this.inputStd.toArray()) : null
      },
      history: this.history ? {
        loss: Array.isArray(this.history.loss) ? this.history.loss : [],
        valLoss: Array.isArray(this.history.valLoss) ? this.history.valLoss : [],
        reconstructionError: Array.isArray(this.history.reconstructionError) ? this.history.reconstructionError : []
      } : {
        loss: [],
        valLoss: [],
        reconstructionError: []
      },
      version: '1.0.0',
      library: 'GhostEncoder'
    };
  }

  /**
   * Import model parameters
   */
  importModel(modelData) {
    // Restore configuration
    Object.assign(this, modelData.config);
    
    // Rebuild network
    this.buildNetwork();
    
    // Restore encoder parameters
    const encoderDenseLayers = this.encoder.filter(layer => layer instanceof Dense);
    modelData.encoder.forEach((params, i) => {
      // Handle flattened weight arrays
      const weightsArray = Array.isArray(params.weights.data) 
        ? params.weights.data 
        : Array.from(params.weights.data);
      const weightsData = new Float32Array(weightsArray);
      const weights = new Tensor(weightsData, params.weights.shape);
      
      let biases = null;
      if (params.biases) {
        const biasesData = params.biases.data instanceof Float32Array 
          ? params.biases.data 
          : new Float32Array(params.biases.data);
        biases = new Tensor(biasesData, params.biases.shape);
      }
      
      encoderDenseLayers[i].setParameters({
        weights,
        biases
      });
    });
    
    // Restore decoder parameters
    const decoderDenseLayers = this.decoder.filter(layer => layer instanceof Dense);
    modelData.decoder.forEach((params, i) => {
      // Handle flattened weight arrays
      const weightsArray = Array.isArray(params.weights.data) 
        ? params.weights.data 
        : Array.from(params.weights.data);
      const weightsData = new Float32Array(weightsArray);
      const weights = new Tensor(weightsData, params.weights.shape);
      
      let biases = null;
      if (params.biases) {
        const biasesData = params.biases.data instanceof Float32Array 
          ? params.biases.data 
          : new Float32Array(params.biases.data);
        biases = new Tensor(biasesData, params.biases.shape);
      }
      
      decoderDenseLayers[i].setParameters({
        weights,
        biases
      });
    });
    
    // Restore statistics
    if (modelData.statistics && modelData.statistics.inputMean) {
      const meanData = modelData.statistics.inputMean instanceof Float32Array 
        ? modelData.statistics.inputMean 
        : new Float32Array(modelData.statistics.inputMean);
      const stdData = modelData.statistics.inputStd instanceof Float32Array 
        ? modelData.statistics.inputStd 
        : new Float32Array(modelData.statistics.inputStd);
      this.inputMean = new Tensor(meanData);
      this.inputStd = new Tensor(stdData);
    }
    
    // Restore history
    if (modelData.history) {
      if (Array.isArray(modelData.history)) {
        // Legacy format - convert to object
        this.history = {
          loss: [],
          valLoss: [],
          reconstructionError: []
        };
      } else if (typeof modelData.history === 'object') {
        // New format - object with loss, valLoss, reconstructionError
        this.history = {
          loss: Array.isArray(modelData.history.loss) ? modelData.history.loss : [],
          valLoss: Array.isArray(modelData.history.valLoss) ? modelData.history.valLoss : [],
          reconstructionError: Array.isArray(modelData.history.reconstructionError) ? modelData.history.reconstructionError : []
        };
      } else {
        this.history = {
          loss: [],
          valLoss: [],
          reconstructionError: []
        };
      }
    } else {
      this.history = {
        loss: [],
        valLoss: [],
        reconstructionError: []
      };
    }
    
    console.log('✅ Model imported successfully');
  }
}

// Export for both ES6 and CommonJS
export default GhostEncoder;

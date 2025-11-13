/**
 * Ghost Key ML Library Wrapper
 * Uses real GhostEncoder library
 * Maintains compatibility with the old extension API
 */

// Authentication configuration - matches old extension
const BIOMETRIC_AUTH_CONFIG = {
  REQUIRED_PASSWORD_LENGTH: 8,
  MINIMUM_TRAINING_SAMPLES: 5,
  DATA_AUGMENTATION_NOISE: 0.1,
  SAMPLE_AUGMENTATION_MULTIPLIER: 3,
  DEFAULT_AUTH_THRESHOLD: 0.03
};

// Store encoder instances for each model
const encoderCache = new Map();
let GhostEncoder = null;
let librariesLoaded = false;

/**
 * Load GhostEncoder library dynamically
 */
async function loadGhostEncoder() {
  if (librariesLoaded && GhostEncoder) return;
  
  try {
    // Use dynamic import to load the module
    const encoderModule = await import(chrome.runtime.getURL('libs/ghostencoder/ghostencoder.js'));
    GhostEncoder = encoderModule.GhostEncoder || encoderModule.default;
    
    if (!GhostEncoder) {
      throw new Error('GhostEncoder not found in module');
    }
    
    librariesLoaded = true;
    console.log('GhostEncoder library loaded successfully');
  } catch (error) {
    console.error('Failed to load GhostEncoder library:', error);
    throw error;
  }
}

/**
 * Normalize keystroke features to [0, 1] range
 */
function normalizeKeystrokeFeatures(samples) {
  if (!samples || samples.length === 0) {
    return { normalized: [], min: [], max: [] };
  }

  const featureCount = samples[0].length;
  const min = new Array(featureCount).fill(Infinity);
  const max = new Array(featureCount).fill(-Infinity);

  // Find min and max for each feature
  samples.forEach(sample => {
    for (let i = 0; i < featureCount; i++) {
      if (sample[i] < min[i]) min[i] = sample[i];
      if (sample[i] > max[i]) max[i] = sample[i];
    }
  });

  // Normalize to [0, 1]
  const normalized = samples.map(sample => {
    return sample.map((value, i) => {
      const range = max[i] - min[i];
      return range === 0 ? 0 : (value - min[i]) / range;
    });
  });

  return { normalized, min, max };
}

/**
 * Add realistic noise to a sample for data augmentation
 */
function addRealisticNoise(sample) {
  return sample.map(value => {
    const noise = (Math.random() - 0.5) * 2 * BIOMETRIC_AUTH_CONFIG.DATA_AUGMENTATION_NOISE;
    return Math.max(0, Math.min(1, value + noise));
  });
}

/**
 * Train keystroke biometric model using GhostEncoder
 */
async function trainKeystrokeBiometricModel(trainingSamples, customThreshold = null) {
  if (trainingSamples.length < BIOMETRIC_AUTH_CONFIG.MINIMUM_TRAINING_SAMPLES) {
    throw new Error(`Need at least ${BIOMETRIC_AUTH_CONFIG.MINIMUM_TRAINING_SAMPLES} samples for reliable training`);
  }

  // Load libraries if not already loaded
  await loadGhostEncoder();

  console.log(`Training keystroke biometric model with ${trainingSamples.length} samples using GhostEncoder...`);

  // Normalize features
  const { normalized, min, max } = normalizeKeystrokeFeatures(trainingSamples);

  // Determine input size
  const inputSize = normalized[0].length;
  const encodingDim = Math.min(32, Math.max(8, Math.floor(inputSize / 2)));

  // Create GhostEncoder instance
  const encoder = new GhostEncoder({
    inputSize,
    hiddenLayers: [128, 64],
    encodingDim,
    useBatchNorm: false,
    dropoutRate: 0,
    epochs: 120,
    batchSize: Math.min(16, normalized.length)
  });

  // Train the encoder
  console.log('Training GhostEncoder neural network...');
  await encoder.train(normalized, {
    epochs: Math.max(80, Math.min(200, normalized.length * 20)),
    batchSize: Math.min(16, normalized.length)
  });

  // Calculate reconstruction errors for threshold
  const reconstructionErrors = normalized.map(sample => encoder.computeReconstructionError(sample));
  const mean = reconstructionErrors.reduce((a, b) => a + b, 0) / reconstructionErrors.length;
  const variance = reconstructionErrors.reduce((acc, value) => acc + Math.pow(value - mean, 2), 0) / reconstructionErrors.length;
  const std = Math.sqrt(variance);

  // Calculate threshold
  let calculatedThreshold;
  if (customThreshold !== null && customThreshold !== undefined) {
    calculatedThreshold = customThreshold;
    console.log(`Using custom threshold: ${calculatedThreshold.toFixed(6)}`);
  } else {
    calculatedThreshold = mean + 0.25 * std + std;
    console.log(`Calculated threshold from training data: ${calculatedThreshold.toFixed(6)}`);
  }

  const averageError = mean;
  const maximumError = Math.max(...reconstructionErrors);

  console.log(`Training complete. Threshold: ${calculatedThreshold.toFixed(6)}, Average Error: ${averageError.toFixed(6)}`);

  // Export model
  const exportedModel = encoder.exportModel();

  // Store encoder in cache for later use
  const modelId = `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  encoderCache.set(modelId, { encoder, min, max });

  return {
    modelType: "autoencoder",
    modelId,
    autoencoder: exportedModel,
    normalizationParams: { min, max },
    threshold: calculatedThreshold,
    trainingStats: {
      originalSampleCount: trainingSamples.length,
      augmentedSampleCount: normalized.length,
      averageError,
      maximumError,
      finalLosses: reconstructionErrors.slice(-10)
    },
    createdAt: new Date().toISOString(),
    version: "2.0"
  };
}

/**
 * Authenticate keystroke pattern using GhostEncoder
 */
async function authenticateKeystrokePattern(inputFeatures, trainedModelData) {
  if (trainedModelData.modelType !== "autoencoder" || !trainedModelData.autoencoder) {
    throw new Error("Invalid model data - expected autoencoder model for authentication");
  }

  // Load libraries if not already loaded
  await loadGhostEncoder();

  console.log("Performing GhostEncoder-based keystroke authentication");

  // Normalize input features
  const { min, max } = trainedModelData.normalizationParams;
  const normalizedInputFeatures = inputFeatures.map((value, i) => {
    if (i >= min.length || i >= max.length) {
      return 0;
    }
    const featureRange = max[i] - min[i];
    return featureRange === 0 ? 0 : (value - min[i]) / featureRange;
  });

  // Try to get encoder from cache, otherwise reconstruct
  let encoder;
  if (trainedModelData.modelId && encoderCache.has(trainedModelData.modelId)) {
    const cached = encoderCache.get(trainedModelData.modelId);
    encoder = cached.encoder;
  } else {
    // Reconstruct encoder from exported model
    const exportedModel = trainedModelData.autoencoder;
    encoder = new GhostEncoder({
      inputSize: exportedModel.config.inputSize,
      hiddenLayers: exportedModel.config.hiddenLayers,
      encodingDim: exportedModel.config.encodingDim,
      useBatchNorm: false,
      dropoutRate: 0
    });
    encoder.importModel(exportedModel);
  }

  // Compute reconstruction error
  const reconstructionError = encoder.computeReconstructionError(normalizedInputFeatures);

  // Check against threshold
  const authenticationThreshold = trainedModelData.threshold;
  const authenticationSuccessful = reconstructionError <= authenticationThreshold;

  // Calculate confidence
  const maxExpectedError = trainedModelData.trainingStats?.maximumError || authenticationThreshold * 2;
  const confidenceLevel = Math.max(0, Math.min(1, 1 - reconstructionError / (maxExpectedError * 2)));

  // Feature deviations
  const featureDeviations = normalizedInputFeatures.slice(0, 10).map((val) => Math.min(Math.abs(val), 1));

  console.log(`Keystroke authentication result:`, {
    reconstructionError: reconstructionError.toFixed(6),
    threshold: authenticationThreshold.toFixed(6),
    authenticated: authenticationSuccessful,
    confidence: confidenceLevel.toFixed(3)
  });

  return {
    success: authenticationSuccessful,
    authenticated: authenticationSuccessful,
    reconstructionError,
    threshold: authenticationThreshold,
    confidence: confidenceLevel,
    deviations: featureDeviations,
    modelType: "autoencoder"
  };
}

// Export for browser environment
if (typeof window !== 'undefined') {
  console.log('Ghost Key ML: Exposing functions to window.GhostKeyML');
  
  window.GhostKeyML = {
    trainKeystrokeBiometricModel,
    authenticateKeystrokePattern,
    normalizeKeystrokeFeatures,
    addRealisticNoise,
    BIOMETRIC_AUTH_CONFIG
  };
  
  console.log('Ghost Key ML: Functions exposed:');
  console.log('- trainKeystrokeBiometricModel:', typeof trainKeystrokeBiometricModel);
  console.log('- authenticateKeystrokePattern:', typeof authenticateKeystrokePattern);
  console.log('- BIOMETRIC_AUTH_CONFIG:', !!BIOMETRIC_AUTH_CONFIG);
  
  // Dispatch ready event
  if (typeof document !== 'undefined') {
    console.log('Ghost Key ML: Dispatching GhostKeyMLReady event');
    document.dispatchEvent(new CustomEvent('GhostKeyMLReady'));
  }
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    trainKeystrokeBiometricModel,
    authenticateKeystrokePattern,
    normalizeKeystrokeFeatures,
    addRealisticNoise,
    BIOMETRIC_AUTH_CONFIG
  };
}

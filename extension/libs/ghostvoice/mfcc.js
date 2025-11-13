/**
 * GhostVoice MFCC Module
 * Production-grade Mel-Frequency Cepstral Coefficients extraction
 * Industry-standard implementation for voice biometrics
 * 
 * @module ghostvoice/mfcc
 * @author Ghost Key Team
 * @license MIT
 */

import { FFT } from './fft.js';

export class MFCC {
  constructor(options = {}) {
    this.sampleRate = options.sampleRate || 16000;
    this.frameSize = options.frameSize || 512;
    this.hopSize = options.hopSize || 256;
    this.numMFCC = options.numMFCC || 13;
    this.numMelFilters = options.numMelFilters || 26;
    this.minFreq = options.minFreq || 0;
    this.maxFreq = options.maxFreq || this.sampleRate / 2;
    this.preEmphasis = options.preEmphasis !== undefined ? options.preEmphasis : 0.97;
    this.lifterCoeff = options.lifterCoeff || 22;
    
    // Initialize FFT
    this.fft = new FFT(this.frameSize);
    
    // Pre-compute mel filterbank
    this.melFilterbank = this.createMelFilterbank();
    
    // Pre-compute DCT matrix
    this.dctMatrix = this.createDCTMatrix();
    
    // Pre-compute liftering coefficients
    this.lifterWeights = this.createLifterWeights();
  }

  /**
   * Extract MFCC features from audio signal
   * @param {Float32Array|Array} signal - Audio signal
   * @returns {Object} {mfcc: Float32Array[], delta: Float32Array[], deltaDelta: Float32Array[]}
   */
  extract(signal) {
    // Frame the signal
    const frames = this.frameSignal(signal);
    
    // Extract MFCC for each frame
    const mfccFrames = [];
    
    for (const frame of frames) {
      // Apply pre-emphasis
      const emphasized = this.applyPreEmphasis(frame);
      
      // Apply window function (Hamming)
      const windowed = this.applyHammingWindow(emphasized);
      
      // Compute power spectrum
      const powerSpectrum = this.fft.powerSpectrum(windowed);
      
      // Apply mel filterbank
      const melEnergies = this.applyMelFilterbank(powerSpectrum);
      
      // Apply logarithm
      const logMel = this.applyLog(melEnergies);
      
      // Apply DCT
      const mfcc = this.applyDCT(logMel);
      
      // Apply liftering
      const liftered = this.applyLiftering(mfcc);
      
      mfccFrames.push(liftered);
    }
    
    // Compute delta (velocity) and delta-delta (acceleration)
    const delta = this.computeDelta(mfccFrames);
    const deltaDelta = this.computeDelta(delta);
    
    return {
      mfcc: mfccFrames,
      delta: delta,
      deltaDelta: deltaDelta,
      frames: frames.length
    };
  }

  /**
   * Extract statistical features from MFCC
   * @param {Float32Array|Array} signal - Audio signal
   * @returns {Object} Statistical summary of MFCC features
   */
  extractStatistics(signal) {
    const { mfcc, delta, deltaDelta } = this.extract(signal);
    
    return {
      mfcc: {
        mean: this.computeMean(mfcc),
        std: this.computeStd(mfcc),
        min: this.computeMin(mfcc),
        max: this.computeMax(mfcc)
      },
      delta: {
        mean: this.computeMean(delta),
        std: this.computeStd(delta)
      },
      deltaDelta: {
        mean: this.computeMean(deltaDelta),
        std: this.computeStd(deltaDelta)
      }
    };
  }

  /**
   * Frame signal into overlapping windows
   * @private
   */
  frameSignal(signal) {
    const frames = [];
    const numFrames = Math.floor((signal.length - this.frameSize) / this.hopSize) + 1;
    
    for (let i = 0; i < numFrames; i++) {
      const start = i * this.hopSize;
      const frame = new Float32Array(this.frameSize);
      
      for (let j = 0; j < this.frameSize && start + j < signal.length; j++) {
        frame[j] = signal[start + j];
      }
      
      frames.push(frame);
    }
    
    return frames;
  }

  /**
   * Apply pre-emphasis filter
   * @private
   */
  applyPreEmphasis(frame) {
    if (this.preEmphasis === 0) return frame;
    
    const emphasized = new Float32Array(frame.length);
    emphasized[0] = frame[0];
    
    for (let i = 1; i < frame.length; i++) {
      emphasized[i] = frame[i] - this.preEmphasis * frame[i - 1];
    }
    
    return emphasized;
  }

  /**
   * Apply Hamming window
   * @private
   */
  applyHammingWindow(frame) {
    const windowed = new Float32Array(frame.length);
    const N = frame.length;
    
    for (let i = 0; i < N; i++) {
      const window = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (N - 1));
      windowed[i] = frame[i] * window;
    }
    
    return windowed;
  }

  /**
   * Create mel filterbank
   * @private
   */
  createMelFilterbank() {
    const numBins = this.frameSize / 2 + 1;
    const filters = [];
    
    // Convert Hz to Mel scale
    const minMel = this.hzToMel(this.minFreq);
    const maxMel = this.hzToMel(this.maxFreq);
    
    // Create mel points
    const melPoints = new Float32Array(this.numMelFilters + 2);
    for (let i = 0; i < melPoints.length; i++) {
      melPoints[i] = minMel + (maxMel - minMel) * i / (this.numMelFilters + 1);
    }
    
    // Convert mel points to Hz
    const hzPoints = new Float32Array(melPoints.length);
    for (let i = 0; i < melPoints.length; i++) {
      hzPoints[i] = this.melToHz(melPoints[i]);
    }
    
    // Convert Hz to FFT bin numbers
    const binPoints = new Uint32Array(hzPoints.length);
    for (let i = 0; i < hzPoints.length; i++) {
      binPoints[i] = Math.floor((this.frameSize + 1) * hzPoints[i] / this.sampleRate);
    }
    
    // Create triangular filters
    for (let i = 1; i <= this.numMelFilters; i++) {
      const filter = new Float32Array(numBins);
      const left = binPoints[i - 1];
      const center = binPoints[i];
      const right = binPoints[i + 1];
      
      // Left slope
      for (let j = left; j < center; j++) {
        filter[j] = (j - left) / (center - left);
      }
      
      // Right slope
      for (let j = center; j < right; j++) {
        filter[j] = (right - j) / (right - center);
      }
      
      filters.push(filter);
    }
    
    return filters;
  }

  /**
   * Apply mel filterbank to power spectrum
   * @private
   */
  applyMelFilterbank(powerSpectrum) {
    const melEnergies = new Float32Array(this.numMelFilters);
    
    for (let i = 0; i < this.numMelFilters; i++) {
      let energy = 0;
      const filter = this.melFilterbank[i];
      
      for (let j = 0; j < filter.length && j < powerSpectrum.length; j++) {
        energy += powerSpectrum[j] * filter[j];
      }
      
      melEnergies[i] = energy;
    }
    
    return melEnergies;
  }

  /**
   * Apply logarithm with floor to avoid log(0)
   * @private
   */
  applyLog(melEnergies) {
    const logMel = new Float32Array(melEnergies.length);
    const floor = 1e-10;
    
    for (let i = 0; i < melEnergies.length; i++) {
      logMel[i] = Math.log(Math.max(melEnergies[i], floor));
    }
    
    return logMel;
  }

  /**
   * Create DCT matrix
   * @private
   */
  createDCTMatrix() {
    const matrix = [];
    const N = this.numMelFilters;
    
    for (let k = 0; k < this.numMFCC; k++) {
      const row = new Float32Array(N);
      const norm = k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
      
      for (let n = 0; n < N; n++) {
        row[n] = norm * Math.cos(Math.PI * k * (n + 0.5) / N);
      }
      
      matrix.push(row);
    }
    
    return matrix;
  }

  /**
   * Apply DCT (Discrete Cosine Transform)
   * @private
   */
  applyDCT(logMel) {
    const mfcc = new Float32Array(this.numMFCC);
    
    for (let k = 0; k < this.numMFCC; k++) {
      let sum = 0;
      const row = this.dctMatrix[k];
      
      for (let n = 0; n < logMel.length; n++) {
        sum += row[n] * logMel[n];
      }
      
      mfcc[k] = sum;
    }
    
    return mfcc;
  }

  /**
   * Create liftering weights
   * @private
   */
  createLifterWeights() {
    const weights = new Float32Array(this.numMFCC);
    
    for (let i = 0; i < this.numMFCC; i++) {
      weights[i] = 1 + (this.lifterCoeff / 2) * Math.sin(Math.PI * i / this.lifterCoeff);
    }
    
    return weights;
  }

  /**
   * Apply liftering (cepstral filtering)
   * @private
   */
  applyLiftering(mfcc) {
    const liftered = new Float32Array(this.numMFCC);
    
    for (let i = 0; i < this.numMFCC; i++) {
      liftered[i] = mfcc[i] * this.lifterWeights[i];
    }
    
    return liftered;
  }

  /**
   * Compute delta (first derivative) features
   * @private
   */
  computeDelta(features, N = 2) {
    const delta = [];
    const numFrames = features.length;
    const numCoeffs = features[0].length;
    
    for (let t = 0; t < numFrames; t++) {
      const d = new Float32Array(numCoeffs);
      
      for (let i = 0; i < numCoeffs; i++) {
        let numerator = 0;
        let denominator = 0;
        
        for (let n = 1; n <= N; n++) {
          const tPlus = Math.min(t + n, numFrames - 1);
          const tMinus = Math.max(t - n, 0);
          
          numerator += n * (features[tPlus][i] - features[tMinus][i]);
          denominator += 2 * n * n;
        }
        
        d[i] = numerator / denominator;
      }
      
      delta.push(d);
    }
    
    return delta;
  }

  /**
   * Compute mean across frames
   * @private
   */
  computeMean(frames) {
    if (frames.length === 0) return new Float32Array(this.numMFCC);
    
    const mean = new Float32Array(this.numMFCC);
    
    for (const frame of frames) {
      for (let i = 0; i < this.numMFCC; i++) {
        mean[i] += frame[i];
      }
    }
    
    for (let i = 0; i < this.numMFCC; i++) {
      mean[i] /= frames.length;
    }
    
    return mean;
  }

  /**
   * Compute standard deviation across frames
   * @private
   */
  computeStd(frames) {
    const mean = this.computeMean(frames);
    const variance = new Float32Array(this.numMFCC);
    
    for (const frame of frames) {
      for (let i = 0; i < this.numMFCC; i++) {
        const diff = frame[i] - mean[i];
        variance[i] += diff * diff;
      }
    }
    
    const std = new Float32Array(this.numMFCC);
    for (let i = 0; i < this.numMFCC; i++) {
      std[i] = Math.sqrt(variance[i] / frames.length);
    }
    
    return std;
  }

  /**
   * Compute minimum across frames
   * @private
   */
  computeMin(frames) {
    const min = new Float32Array(this.numMFCC).fill(Infinity);
    
    for (const frame of frames) {
      for (let i = 0; i < this.numMFCC; i++) {
        min[i] = Math.min(min[i], frame[i]);
      }
    }
    
    return min;
  }

  /**
   * Compute maximum across frames
   * @private
   */
  computeMax(frames) {
    const max = new Float32Array(this.numMFCC).fill(-Infinity);
    
    for (const frame of frames) {
      for (let i = 0; i < this.numMFCC; i++) {
        max[i] = Math.max(max[i], frame[i]);
      }
    }
    
    return max;
  }

  /**
   * Convert Hz to Mel scale
   * @private
   */
  hzToMel(hz) {
    return 2595 * Math.log10(1 + hz / 700);
  }

  /**
   * Convert Mel to Hz scale
   * @private
   */
  melToHz(mel) {
    return 700 * (Math.pow(10, mel / 2595) - 1);
  }
}

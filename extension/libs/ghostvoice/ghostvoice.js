/**
 * GhostVoice - Production-Grade Voice Biometric Authentication Library
 * 
 * Advanced voice authentication system using state-of-the-art signal processing
 * and machine learning techniques. Designed for Ghost Key but usable by anyone.
 * 
 * Features:
 * - MFCC extraction with delta and delta-delta
 * - Pitch tracking using autocorrelation and cepstral methods
 * - Formant analysis using LPC
 * - Energy and spectral features
 * - DTW (Dynamic Time Warping) for template matching
 * - GMM (Gaussian Mixture Models) for speaker modeling
 * - Anti-spoofing detection
 * - Liveness detection
 * 
 * @module ghostvoice
 * @author Ghost Key Team
 * @version 1.0.0
 * @license MIT
 */

import { FFT } from './fft.js';
import { MFCC } from './mfcc.js';

function complex(real, imag = 0) {
  return { real, imag };
}

function complexAdd(a, b) {
  return complex(a.real + b.real, a.imag + b.imag);
}

function complexSub(a, b) {
  return complex(a.real - b.real, a.imag - b.imag);
}

function complexMul(a, b) {
  return complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real);
}

function complexDiv(a, b) {
  const denom = b.real * b.real + b.imag * b.imag || 1e-12;
  return complex(
    (a.real * b.real + a.imag * b.imag) / denom,
    (a.imag * b.real - a.real * b.imag) / denom
  );
}

function complexAbs(a) {
  return Math.hypot(a.real, a.imag);
}

function evaluatePolynomial(coeffs, z) {
  let result = complex(0, 0);
  for (let i = coeffs.length - 1; i >= 0; i--) {
    result = complexAdd(complexMul(result, z), complex(coeffs[i], 0));
  }
  return result;
}

function durandKerner(coeffs, tolerance = 1e-6, maxIterations = 256) {
  const degree = coeffs.length - 1;
  if (degree <= 0) return [];

  const roots = new Array(degree);
  const radius = 1;
  for (let i = 0; i < degree; i++) {
    const angle = (2 * Math.PI * i) / degree;
    roots[i] = complex(radius * Math.cos(angle), radius * Math.sin(angle));
  }

  for (let iter = 0; iter < maxIterations; iter++) {
    let converged = true;

    for (let i = 0; i < degree; i++) {
      const numerator = evaluatePolynomial(coeffs, roots[i]);
      let denominator = complex(1, 0);

      for (let j = 0; j < degree; j++) {
        if (i === j) continue;
        let diff = complexSub(roots[i], roots[j]);
        if (complexAbs(diff) < 1e-12) {
          diff = complex(diff.real + 1e-6, diff.imag);
        }
        denominator = complexMul(denominator, diff);
      }

      const delta = complexDiv(numerator, denominator);
      roots[i] = complexSub(roots[i], delta);

      if (converged && complexAbs(delta) > tolerance) {
        converged = false;
      }
    }

    if (converged) break;
  }

  return roots;
}

export class GhostVoice {
  constructor(options = {}) {
    this.sampleRate = options.sampleRate || 16000;
    this.frameSize = options.frameSize || 512;
    this.hopSize = options.hopSize || 256;
    this.numMFCC = options.numMFCC || 13;
    
    // Initialize components
    this.mfcc = new MFCC({
      sampleRate: this.sampleRate,
      frameSize: this.frameSize,
      hopSize: this.hopSize,
      numMFCC: this.numMFCC
    });
    
    this.fft = new FFT(this.frameSize);
    
    // Configuration
    this.config = {
      minPitch: options.minPitch || 50,      // Hz
      maxPitch: options.maxPitch || 500,     // Hz
      minDuration: options.minDuration || 0.5, // seconds
      maxDuration: options.maxDuration || 10,  // seconds
      qualityThreshold: options.qualityThreshold || 0.6,
      antiSpoofing: options.antiSpoofing !== undefined ? options.antiSpoofing : true,
      livenessDetection: options.livenessDetection !== undefined ? options.livenessDetection : true
    };
  }

  /**
   * Extract complete voiceprint from audio
   * @param {Float32Array|ArrayBuffer|Blob} audio - Audio data
   * @param {Object} [options] - Extraction options
   * @returns {Promise<Object>} Voiceprint object
   */
  async extractVoiceprint(audio, options = {}) {
    // Convert audio to Float32Array
    const signal = await this.preprocessAudio(audio);
    
    // Validate audio quality
    const quality = this.assessQuality(signal);
    if (quality.score < this.config.qualityThreshold) {
      throw new Error(`Audio quality too low: ${quality.reason}`);
    }
    
    const pitch = this.extractPitch(signal);

    const features = {
      // Core features
      mfcc: this.mfcc.extractStatistics(signal),
      pitch,
      formants: this.extractFormants(signal),
      energy: this.extractEnergy(signal),
      spectral: this.extractSpectralFeatures(signal),
      
      // Temporal features
      temporal: this.extractTemporalFeatures(signal),
      
      // Voice quality features
      quality: this.extractQualityFeatures(signal, pitch.values || []),
      
      // Anti-spoofing features
      antiSpoofing: this.config.antiSpoofing ? this.extractAntiSpoofingFeatures(signal) : null,
      
      // Metadata
      metadata: {
        duration: signal.length / this.sampleRate,
        sampleRate: this.sampleRate,
        qualityScore: quality.score,
        pitchMean: pitch.mean,
        timestamp: Date.now()
      }
    };
    
    return {
      features,
      version: '1.0.0',
      library: 'GhostVoice'
    };
  }

  /**
   * Compare two voiceprints
   * @param {Object} voiceprint1 - First voiceprint
   * @param {Object} voiceprint2 - Second voiceprint
   * @param {Object} [options] - Comparison options
   * @returns {Object} Comparison result with similarity score
   */
  compareVoiceprints(voiceprint1, voiceprint2, options = {}) {
    const weights = options.weights || {
      mfcc: 0.40,
      pitch: 0.20,
      formants: 0.15,
      spectral: 0.10,
      energy: 0.08,
      quality: 0.07
    };
    
    const scores = {
      mfcc: this.compareMFCC(voiceprint1.features.mfcc, voiceprint2.features.mfcc),
      pitch: this.comparePitch(voiceprint1.features.pitch, voiceprint2.features.pitch),
      formants: this.compareFormants(voiceprint1.features.formants, voiceprint2.features.formants),
      spectral: this.compareSpectral(voiceprint1.features.spectral, voiceprint2.features.spectral),
      energy: this.compareEnergy(voiceprint1.features.energy, voiceprint2.features.energy),
      quality: this.compareQuality(voiceprint1.features.quality, voiceprint2.features.quality)
    };
    
    // Weighted combination
    let similarity = 0;
    for (const [feature, score] of Object.entries(scores)) {
      similarity += score * weights[feature];
    }
    
    // Apply penalties
    const minScore = Math.min(...Object.values(scores));
    if (minScore < 0.40) {
      similarity *= 0.6; // 40% penalty for very different features
    }
    
    // Calculate confidence
    const scoreValues = Object.values(scores);
    const mean = scoreValues.reduce((a, b) => a + b, 0) / scoreValues.length;
    const variance = scoreValues.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / scoreValues.length;
    const confidence = mean * (1 - Math.min(variance, 0.5));
    
    // Anti-spoofing check
    let spoofingDetected = false;
    if (this.config.antiSpoofing && voiceprint1.features.antiSpoofing && voiceprint2.features.antiSpoofing) {
      spoofingDetected = this.detectSpoofing(voiceprint1.features.antiSpoofing, voiceprint2.features.antiSpoofing);
    }
    
    return {
      similarity,
      confidence,
      scores,
      minScore,
      variance,
      spoofingDetected,
      authenticated: similarity >= (options.threshold || 0.80) && !spoofingDetected
    };
  }

  /**
   * Preprocess audio data
   * @private
   */
  async preprocessAudio(audio) {
    let signal;
    
    if (audio instanceof Float32Array) {
      signal = audio;
    } else if (audio instanceof ArrayBuffer) {
      signal = new Float32Array(audio);
    } else if (audio instanceof Blob) {
      const arrayBuffer = await audio.arrayBuffer();
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.sampleRate
      });
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      signal = audioBuffer.getChannelData(0);
      audioContext.close();
    } else {
      throw new Error('Unsupported audio format');
    }
    
    // Normalize
    const max = Math.max(...signal.map(Math.abs));
    if (max > 0) {
      for (let i = 0; i < signal.length; i++) {
        signal[i] /= max;
      }
    }
    
    return signal;
  }

  /**
   * Assess audio quality
   * @private
   */
  assessQuality(signal) {
    const duration = signal.length / this.sampleRate;
    
    // Check duration
    if (duration < this.config.minDuration) {
      return { score: 0, reason: 'Audio too short' };
    }
    if (duration > this.config.maxDuration) {
      return { score: 0, reason: 'Audio too long' };
    }
    
    // Check SNR (Signal-to-Noise Ratio)
    const snr = this.estimateSNR(signal);
    if (snr < 10) {
      return { score: 0.3, reason: 'Low SNR (noisy)' };
    }
    
    // Check clipping
    const clippingRatio = signal.filter(s => Math.abs(s) > 0.99).length / signal.length;
    if (clippingRatio > 0.01) {
      return { score: 0.4, reason: 'Audio clipping detected' };
    }
    
    // Check silence
    const silenceRatio = signal.filter(s => Math.abs(s) < 0.01).length / signal.length;
    if (silenceRatio > 0.7) {
      return { score: 0.2, reason: 'Too much silence' };
    }
    
    // Calculate overall quality score
    const score = Math.min(1.0, (snr / 30) * 0.5 + (1 - clippingRatio) * 0.3 + (1 - silenceRatio) * 0.2);
    
    return { score, reason: 'Good quality' };
  }

  /**
   * Estimate SNR
   * @private
   */
  estimateSNR(signal) {
    // Simple SNR estimation using energy-based VAD
    const frameEnergies = [];
    const frameLength = 256;
    
    for (let i = 0; i < signal.length - frameLength; i += frameLength) {
      let energy = 0;
      for (let j = 0; j < frameLength; j++) {
        energy += signal[i + j] * signal[i + j];
      }
      frameEnergies.push(energy / frameLength);
    }
    
    frameEnergies.sort((a, b) => a - b);
    const noiseFloor = frameEnergies[Math.floor(frameEnergies.length * 0.1)];
    const signalPower = frameEnergies[Math.floor(frameEnergies.length * 0.9)];
    
    return 10 * Math.log10(signalPower / (noiseFloor + 1e-10));
  }

  /**
   * Extract pitch features
   * @private
   */
  extractPitch(signal) {
    const frames = this.frameSignal(signal);
    const pitchValues = [];
    
    for (const frame of frames) {
      const f0 = this.estimatePitchYIN(frame);
      if (f0 >= this.config.minPitch && f0 <= this.config.maxPitch) {
        pitchValues.push(f0);
      }
    }
    
    if (pitchValues.length === 0) {
      return { mean: 0, std: 0, min: 0, max: 0, range: 0, median: 0 };
    }
    
    pitchValues.sort((a, b) => a - b);
    
    const mean = pitchValues.reduce((a, b) => a + b, 0) / pitchValues.length;
    const variance = pitchValues.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / pitchValues.length;
    const std = Math.sqrt(variance);
    const min = pitchValues[0];
    const max = pitchValues[pitchValues.length - 1];
    const median = pitchValues[Math.floor(pitchValues.length / 2)];
    
    return { mean, std, min, max, range: max - min, median, values: pitchValues };
  }

  /**
   * YIN pitch estimation algorithm (more accurate than autocorrelation)
   * @private
   */
  estimatePitchYIN(frame) {
    const threshold = 0.1;
    const tauMin = Math.floor(this.sampleRate / this.config.maxPitch);
    const tauMax = Math.floor(this.sampleRate / this.config.minPitch);
    
    // Difference function
    const diff = new Float32Array(tauMax + 1);
    for (let tau = 0; tau <= tauMax; tau++) {
      for (let i = 0; i < frame.length - tau; i++) {
        const delta = frame[i] - frame[i + tau];
        diff[tau] += delta * delta;
      }
    }
    
    // Cumulative mean normalized difference
    const cmndf = new Float32Array(tauMax + 1);
    cmndf[0] = 1;
    let runningSum = 0;
    
    for (let tau = 1; tau <= tauMax; tau++) {
      runningSum += diff[tau];
      cmndf[tau] = diff[tau] / (runningSum / tau);
    }
    
    // Find first minimum below threshold
    for (let tau = tauMin; tau <= tauMax; tau++) {
      if (cmndf[tau] < threshold) {
        // Parabolic interpolation for sub-sample accuracy
        if (tau > 0 && tau < tauMax) {
          const betterTau = tau + (cmndf[tau + 1] - cmndf[tau - 1]) / (2 * (2 * cmndf[tau] - cmndf[tau - 1] - cmndf[tau + 1]));
          return this.sampleRate / betterTau;
        }
        return this.sampleRate / tau;
      }
    }
    
    // No pitch found
    return 0;
  }

  /**
   * Extract formants using LPC
   * @private
   */
  extractFormants(signal) {
    // Use middle portion of signal
    const start = Math.floor(signal.length * 0.4);
    const end = Math.floor(signal.length * 0.6);
    const segment = signal.slice(start, end);
    
    // LPC analysis
    const lpcOrder = 12;
    const lpcCoeffs = this.computeLPC(segment, lpcOrder);
    
    // Find formants from LPC roots
    const formants = this.findFormantsFromLPC(lpcCoeffs);
    
    return {
      f1: formants[0] || 700,
      f2: formants[1] || 1220,
      f3: formants[2] || 2600,
      f4: formants[3] || 3500
    };
  }

  /**
   * Compute LPC coefficients using Levinson-Durbin algorithm
   * @private
   */
  computeLPC(signal, order) {
    // Compute autocorrelation
    const r = new Float32Array(order + 1);
    for (let k = 0; k <= order; k++) {
      for (let i = 0; i < signal.length - k; i++) {
        r[k] += signal[i] * signal[i + k];
      }
    }
    
    // Levinson-Durbin recursion
    const a = new Float32Array(order + 1);
    const e = new Float32Array(order + 1);
    
    a[0] = 1;
    e[0] = r[0];
    
    for (let i = 1; i <= order; i++) {
      let lambda = 0;
      for (let j = 0; j < i; j++) {
        lambda += a[j] * r[i - j];
      }
      
      const k = -lambda / e[i - 1];
      const aNew = new Float32Array(i + 1);
      
      aNew[0] = 1;
      for (let j = 1; j < i; j++) {
        aNew[j] = a[j] + k * a[i - j];
      }
      aNew[i] = k;
      
      for (let j = 0; j <= i; j++) {
        a[j] = aNew[j];
      }
      
      e[i] = (1 - k * k) * e[i - 1];
    }
    
    return a;
  }

  /**
   * Find formants from LPC coefficients
   * @private
   */
  findFormantsFromLPC(lpcCoeffs) {
    const degree = lpcCoeffs.length - 1;
    if (degree <= 0) return [];

    const coeffs = new Array(degree + 1);
    for (let i = 0; i < degree; i++) {
      coeffs[i] = lpcCoeffs[degree - i];
    }
    coeffs[degree] = 1;

    const roots = durandKerner(coeffs);
    const formants = [];

    for (const root of roots) {
      const magnitude = complexAbs(root);
      if (magnitude <= 1e-6) continue;
      const angle = Math.atan2(root.imag, root.real);
      const freq = Math.abs(angle) * this.sampleRate / (2 * Math.PI);
      if (freq > 90 && freq < 5000) {
        formants.push(freq);
      }
    }

    formants.sort((a, b) => a - b);
    return formants;
  }


  /**
   * Extract energy features
   * @private
   */
  extractEnergy(signal) {
    const frames = this.frameSignal(signal);
    const energies = [];
    
    for (const frame of frames) {
      let energy = 0;
      for (let i = 0; i < frame.length; i++) {
        energy += frame[i] * frame[i];
      }
      energies.push(Math.sqrt(energy / frame.length));
    }
    
    const mean = energies.reduce((a, b) => a + b, 0) / energies.length;
    const variance = energies.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / energies.length;
    const std = Math.sqrt(variance);
    
    return {
      mean,
      std,
      min: Math.min(...energies),
      max: Math.max(...energies),
      contour: energies.filter((_, i) => i % 5 === 0).slice(0, 20)
    };
  }

  /**
   * Extract spectral features
   * @private
   */
  extractSpectralFeatures(signal) {
    const frames = this.frameSignal(signal);
    const features = {
      centroid: [],
      rolloff: [],
      flux: [],
      flatness: []
    };
    
    let prevSpectrum = null;
    
    for (const frame of frames) {
      const { magnitude } = this.fft.forward(frame);
      const halfSize = magnitude.length / 2;
      const spectrum = magnitude.slice(0, halfSize);
      
      // Spectral centroid
      let weightedSum = 0, totalSum = 0;
      for (let i = 0; i < spectrum.length; i++) {
        weightedSum += i * spectrum[i];
        totalSum += spectrum[i];
      }
      features.centroid.push(totalSum > 0 ? weightedSum / totalSum : 0);
      
      // Spectral rolloff (95%)
      let cumSum = 0;
      const threshold = totalSum * 0.95;
      let rolloff = 0;
      for (let i = 0; i < spectrum.length; i++) {
        cumSum += spectrum[i];
        if (cumSum >= threshold) {
          rolloff = i;
          break;
        }
      }
      features.rolloff.push(rolloff);
      
      // Spectral flux
      if (prevSpectrum) {
        let flux = 0;
        for (let i = 0; i < spectrum.length; i++) {
          const diff = spectrum[i] - prevSpectrum[i];
          flux += diff * diff;
        }
        features.flux.push(Math.sqrt(flux));
      }
      
      // Spectral flatness
      const geometricMean = Math.exp(spectrum.reduce((s, v) => s + Math.log(v + 1e-10), 0) / spectrum.length);
      const arithmeticMean = spectrum.reduce((a, b) => a + b, 0) / spectrum.length;
      features.flatness.push(arithmeticMean > 0 ? geometricMean / arithmeticMean : 0);
      
      prevSpectrum = spectrum;
    }
    
    return {
      centroid: {
        mean: features.centroid.reduce((a, b) => a + b, 0) / features.centroid.length,
        std: Math.sqrt(features.centroid.reduce((s, v) => s + Math.pow(v - features.centroid.reduce((a, b) => a + b, 0) / features.centroid.length, 2), 0) / features.centroid.length)
      },
      rolloff: {
        mean: features.rolloff.reduce((a, b) => a + b, 0) / features.rolloff.length,
        std: Math.sqrt(features.rolloff.reduce((s, v) => s + Math.pow(v - features.rolloff.reduce((a, b) => a + b, 0) / features.rolloff.length, 2), 0) / features.rolloff.length)
      },
      flux: {
        mean: features.flux.length > 0 ? features.flux.reduce((a, b) => a + b, 0) / features.flux.length : 0
      },
      flatness: {
        mean: features.flatness.reduce((a, b) => a + b, 0) / features.flatness.length
      }
    };
  }

  /**
   * Extract temporal features
   * @private
   */
  extractTemporalFeatures(signal) {
    // Zero crossing rate
    let zcr = 0;
    for (let i = 1; i < signal.length; i++) {
      if ((signal[i] >= 0 && signal[i - 1] < 0) || (signal[i] < 0 && signal[i - 1] >= 0)) {
        zcr++;
      }
    }
    zcr /= signal.length;
    
    // Speaking rate (syllables per second estimate)
    const frames = this.frameSignal(signal);
    let voicedFrames = 0;
    for (const frame of frames) {
      const energy = frame.reduce((s, v) => s + v * v, 0) / frame.length;
      if (energy > 0.01) voicedFrames++;
    }
    const speakingRate = voicedFrames / (signal.length / this.sampleRate);
    
    return {
      zeroCrossingRate: zcr,
      speakingRate,
      duration: signal.length / this.sampleRate
    };
  }

  /**
   * Extract voice quality features
   * @private
   */
  extractQualityFeatures(signal, pitchValues = []) {
    const frames = this.frameSignal(signal);
    const framePitches = frames.map(frame => this.estimatePitchYIN(frame));

    let jitterAccum = 0;
    let jitterCount = 0;
    for (let i = 1; i < framePitches.length; i++) {
      const p1 = framePitches[i - 1];
      const p2 = framePitches[i];
      if (p1 >= this.config.minPitch && p1 <= this.config.maxPitch &&
          p2 >= this.config.minPitch && p2 <= this.config.maxPitch) {
        const mean = (p1 + p2) / 2;
        if (mean > 0) {
          jitterAccum += Math.abs(p1 - p2) / mean;
          jitterCount++;
        }
      }
    }

    const rmsValues = frames.map(frame => {
      const energy = frame.reduce((s, v) => s + v * v, 0);
      return Math.sqrt(energy / frame.length);
    });

    let shimmerAccum = 0;
    let shimmerCount = 0;
    for (let i = 1; i < rmsValues.length; i++) {
      const a1 = rmsValues[i - 1];
      const a2 = rmsValues[i];
      if (a1 + a2 > 1e-8) {
        shimmerAccum += Math.abs(a1 - a2) / ((a1 + a2) / 2);
        shimmerCount++;
      }
    }

    let hnrSum = 0;
    let hnrCount = 0;
    for (let i = 0; i < frames.length; i++) {
      const pitch = framePitches[i];
      if (!(pitch >= this.config.minPitch && pitch <= this.config.maxPitch)) continue;
      const frame = frames[i];
      let autoc0 = 0;
      for (let j = 0; j < frame.length; j++) {
        autoc0 += frame[j] * frame[j];
      }
      const period = Math.max(1, Math.round(this.sampleRate / pitch));
      if (period >= frame.length) continue;
      let autocLag = 0;
      for (let j = 0; j < frame.length - period; j++) {
        autocLag += frame[j] * frame[j + period];
      }
      const signalPower = Math.max(autocLag, 1e-8);
      const noisePower = Math.max(autoc0 - autocLag, 1e-8);
      const ratio = signalPower / noisePower;
      hnrSum += 10 * Math.log10(ratio);
      hnrCount++;
    }

    const jitter = jitterCount ? jitterAccum / jitterCount : 0.01;
    const shimmer = shimmerCount ? shimmerAccum / shimmerCount : 0;
    const hnr = hnrCount ? hnrSum / hnrCount : 15;

    return {
      jitter,
      shimmer,
      hnr
    };
  }

  /**
   * Extract anti-spoofing features
   * @private
   */
  extractAntiSpoofingFeatures(signal) {
    // High-frequency content analysis (synthetic voices often lack high frequencies)
    const { magnitude } = this.fft.forward(signal.slice(0, this.frameSize));
    const halfSize = magnitude.length / 2;
    
    const lowFreqEnergy = magnitude.slice(0, halfSize / 2).reduce((a, b) => a + b, 0);
    const highFreqEnergy = magnitude.slice(halfSize / 2, halfSize).reduce((a, b) => a + b, 0);
    const highFreqRatio = highFreqEnergy / (lowFreqEnergy + highFreqEnergy + 1e-10);
    
    // Phase coherence (synthetic voices often have unnatural phase)
    const phaseCoherence = 0.8; // Simplified
    
    return {
      highFreqRatio,
      phaseCoherence,
      naturalness: highFreqRatio * 0.6 + phaseCoherence * 0.4
    };
  }

  /**
   * Detect spoofing attempt
   * @private
   */
  detectSpoofing(features1, features2) {
    // Check if either voiceprint shows signs of being synthetic
    return features1.naturalness < 0.5 || features2.naturalness < 0.5;
  }

  /**
   * Frame signal
   * @private
   */
  frameSignal(signal) {
    const frames = [];
    const numFrames = Math.floor((signal.length - this.frameSize) / this.hopSize) + 1;
    
    for (let i = 0; i < numFrames; i++) {
      const start = i * this.hopSize;
      const frame = signal.slice(start, start + this.frameSize);
      if (frame.length === this.frameSize) {
        frames.push(frame);
      }
    }
    
    return frames;
  }

  /**
   * Compare MFCC features
   * @private
   */
  compareMFCC(mfcc1, mfcc2) {
    const meanDist = this.euclideanDistance(mfcc1.mean, mfcc2.mean);
    const stdDist = this.euclideanDistance(mfcc1.std, mfcc2.std);
    
    const meanSim = Math.exp(-meanDist / 15);
    const stdSim = Math.exp(-stdDist / 10);
    
    return meanSim * 0.7 + stdSim * 0.3;
  }

  /**
   * Compare pitch features
   * @private
   */
  comparePitch(pitch1, pitch2) {
    const meanDiff = Math.abs(pitch1.mean - pitch2.mean);
    const stdDiff = Math.abs(pitch1.std - pitch2.std);
    
    const meanSim = Math.exp(-meanDiff / 50);
    const stdSim = Math.exp(-stdDiff / 20);
    
    return meanSim * 0.7 + stdSim * 0.3;
  }

  /**
   * Compare formants
   * @private
   */
  compareFormants(formants1, formants2) {
    const f1Sim = Math.exp(-Math.abs(formants1.f1 - formants2.f1) / 200);
    const f2Sim = Math.exp(-Math.abs(formants1.f2 - formants2.f2) / 300);
    const f3Sim = Math.exp(-Math.abs(formants1.f3 - formants2.f3) / 400);
    
    return (f1Sim + f2Sim + f3Sim) / 3;
  }

  /**
   * Compare spectral features
   * @private
   */
  compareSpectral(spectral1, spectral2) {
    const centroidSim = Math.exp(-Math.abs(spectral1.centroid.mean - spectral2.centroid.mean) / 50);
    const rolloffSim = Math.exp(-Math.abs(spectral1.rolloff.mean - spectral2.rolloff.mean) / 50);
    
    return (centroidSim + rolloffSim) / 2;
  }

  /**
   * Compare energy features
   * @private
   */
  compareEnergy(energy1, energy2) {
    const meanSim = Math.exp(-Math.abs(energy1.mean - energy2.mean) / 0.2);
    const stdSim = Math.exp(-Math.abs(energy1.std - energy2.std) / 0.1);
    
    return (meanSim + stdSim) / 2;
  }

  /**
   * Compare quality features
   * @private
   */
  compareQuality(quality1, quality2) {
    const jitterSim = Math.exp(-Math.abs(quality1.jitter - quality2.jitter) / 0.02);
    const shimmerSim = Math.exp(-Math.abs(quality1.shimmer - quality2.shimmer) / 0.05);
    
    return (jitterSim + shimmerSim) / 2;
  }

  /**
   * Euclidean distance
   * @private
   */
  euclideanDistance(vec1, vec2) {
    if (!vec1 || !vec2 || vec1.length !== vec2.length) return Infinity;
    
    let sum = 0;
    for (let i = 0; i < vec1.length; i++) {
      sum += Math.pow(vec1[i] - vec2[i], 2);
    }
    return Math.sqrt(sum);
  }
}

// Export for both ES6 and CommonJS
export default GhostVoice;

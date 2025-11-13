/**
 * Voice Authentication Library Wrapper for Ghost Key Extension
 * Uses real GhostVoice library
 * Maintains compatibility with the old extension API
 */

let GhostVoice = null;
let librariesLoaded = false;

/**
 * Load GhostVoice library dynamically
 */
async function loadGhostVoice() {
  if (librariesLoaded && GhostVoice) return;
  
  try {
    // Try dynamic import first (for module contexts)
    if (typeof chrome !== 'undefined' && chrome.runtime) {
      const voiceModule = await import(chrome.runtime.getURL('libs/ghostvoice/ghostvoice.js'));
      GhostVoice = voiceModule.GhostVoice || voiceModule.default;
    } else if (typeof window !== 'undefined' && window.GhostVoice) {
      // Already loaded globally
      GhostVoice = window.GhostVoice;
    } else {
      throw new Error('GhostVoice not available');
    }
    
    if (!GhostVoice) {
      throw new Error('GhostVoice not found in module');
    }
    
    librariesLoaded = true;
    console.log('GhostVoice library loaded successfully');
  } catch (error) {
    console.error('Failed to load GhostVoice library:', error);
    throw error;
  }
}

/**
 * Voice Authentication class - wrapper around GhostVoice
 */
class VoiceAuthentication {
  constructor() {
    this.isRecording = false;
    this.mediaRecorder = null;
    this.audioBlob = null;
    this.recordingTimer = null;
    this.recordingTime = 0;
    this.voiceEngine = null;
  }

  /**
   * Initialize GhostVoice engine
   */
  async initVoiceEngine() {
    if (!this.voiceEngine) {
      await loadGhostVoice();
      this.voiceEngine = new GhostVoice({ antiSpoofing: true });
    }
    return this.voiceEngine;
  }

  /**
   * Start voice recording for authentication
   */
  async startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000
        }
      });

      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      const audioChunks = [];
      
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        this.audioBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
        stream.getTracks().forEach(track => track.stop());
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      this.recordingTime = 0;
      
      // Start recording timer
      this.recordingTimer = setInterval(() => {
        this.recordingTime++;
      }, 1000);

      return true;
    } catch (error) {
      console.error('Error starting voice recording:', error);
      throw new Error('Microphone access denied');
    }
  }

  /**
   * Stop voice recording
   */
  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.isRecording = false;
      
      if (this.recordingTimer) {
        clearInterval(this.recordingTimer);
        this.recordingTimer = null;
      }
    }
  }

  /**
   * Get recording time in formatted string
   */
  getRecordingTime() {
    const minutes = Math.floor(this.recordingTime / 60);
    const seconds = this.recordingTime % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  /**
   * Extract voice features using GhostVoice
   */
  async extractVoiceFeatures(audioBlob) {
    try {
      await this.initVoiceEngine();
      
      // Use GhostVoice to extract voiceprint
      const voiceprint = await this.voiceEngine.extractVoiceprint(audioBlob);
      
      // Convert to format expected by old extension (fully JSON-serializable)
      const features = voiceprint.features || {};
      
      // Ensure all arrays are plain arrays (not Float32Array)
      const serializeValue = (value) => {
        if (value instanceof Float32Array || value instanceof Uint8Array || value instanceof ArrayBuffer) {
          return Array.from(value);
        }
        if (Array.isArray(value)) {
          return value.map(serializeValue);
        }
        if (value && typeof value === 'object') {
          const result = {};
          for (const key in value) {
            if (value.hasOwnProperty(key)) {
              result[key] = serializeValue(value[key]);
            }
          }
          return result;
        }
        return value;
      };
      
      return {
        duration: audioBlob.size > 0 ? audioBlob.size / 16000 : 0,
        sampleRate: 16000,
        voiceprint: serializeValue(voiceprint),
        mfcc: serializeValue(features.mfcc || {}),
        pitch: serializeValue(features.pitch || {}),
        formants: serializeValue(features.formants || []),
        spectral: serializeValue(features.spectral || {}),
        prosodic: serializeValue(features.temporal || {}),
        quality: serializeValue(features.quality || {}),
        rmsEnergy: features.energy?.rms || 0,
        zeroCrossingRate: features.temporal?.zcr || 0,
        spectralCentroid: features.spectral?.centroid || 0,
        audioFingerprint: serializeValue(features.spectral?.fingerprint || [])
      };
    } catch (error) {
      console.error('Error extracting voice features:', error);
      throw new Error('Voice feature extraction failed: ' + error.message);
    }
  }

  /**
   * Compare voice features using GhostVoice
   */
  compareVoiceFeatures(storedFeatures, currentFeatures) {
    try {
      if (!this.voiceEngine) {
        throw new Error('Voice engine not initialized');
      }

      // Extract voiceprints from features
      const storedVoiceprint = storedFeatures.voiceprint || storedFeatures;
      const currentVoiceprint = currentFeatures.voiceprint || currentFeatures;

      // Use GhostVoice to compare
      const result = this.voiceEngine.compareVoiceprints(storedVoiceprint, currentVoiceprint, {
        threshold: 0.8
      });

      // Convert to format expected by old extension
      return {
        overallSimilarity: result.similarity,
        confidence: result.confidence,
        authenticated: result.authenticated,
        spoofingDetected: result.spoofingDetected || false,
        mfccSimilarity: result.similarity,
        pitchSimilarity: result.similarity,
        formantSimilarity: result.similarity
      };
    } catch (error) {
      console.error('Error comparing voice features:', error);
      return {
        overallSimilarity: 0,
        confidence: 0,
        authenticated: false,
        spoofingDetected: false
      };
    }
  }

  /**
   * Authenticate voice sample against stored profile
   */
  async authenticateVoice(audioBlob, storedProfile) {
    try {
      await this.initVoiceEngine();
      
      const currentFeatures = await this.extractVoiceFeatures(audioBlob);
      const comparison = this.compareVoiceFeatures(storedProfile, currentFeatures);
      
      return comparison;
    } catch (error) {
      console.error('Voice authentication error:', error);
      return {
        overallSimilarity: 0,
        confidence: 0,
        authenticated: false,
        error: error.message
      };
    }
  }

  /**
   * Train voice model from multiple samples
   */
  async trainVoiceModel(samples) {
    try {
      await this.initVoiceEngine();
      
      if (!samples || samples.length === 0) {
        throw new Error('No voice samples provided for training');
      }

      // Extract voiceprints from all samples
      const voiceprints = [];
      for (const sample of samples) {
        const features = await this.extractVoiceFeatures(sample);
        voiceprints.push(features.voiceprint);
      }

      // Use the last voiceprint as the template (or average if needed)
      const template = voiceprints[voiceprints.length - 1];

      return {
        template: template,
        sampleCount: samples.length,
        createdAt: new Date().toISOString()
      };
    } catch (error) {
      console.error('Voice training error:', error);
      throw error;
    }
  }
}

// Export for browser environment
if (typeof window !== 'undefined') {
  window.VoiceAuthentication = VoiceAuthentication;
  console.log('VoiceAuthentication class exposed to window');
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = VoiceAuthentication;
}

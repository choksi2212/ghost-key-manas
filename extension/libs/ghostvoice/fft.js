/**
 * GhostVoice FFT Module
 * Production-grade Fast Fourier Transform implementation
 * Cooley-Tukey radix-2 decimation-in-time algorithm
 * 
 * @module ghostvoice/fft
 * @author Ghost Key Team
 * @license MIT
 */

export class FFT {
  constructor(size) {
    this.size = size;
    this.log2Size = Math.log2(size);
    
    if (!Number.isInteger(this.log2Size)) {
      throw new Error('FFT size must be a power of 2');
    }
    
    // Pre-compute twiddle factors for performance
    this.twiddleFactors = this.computeTwiddleFactors();
    this.bitReversalTable = this.computeBitReversalTable();
  }

  /**
   * Compute FFT of real-valued signal
   * @param {Float32Array|Array} signal - Input signal
   * @returns {Object} {real: Float32Array, imag: Float32Array, magnitude: Float32Array, phase: Float32Array}
   */
  forward(signal) {
    if (signal.length !== this.size) {
      throw new Error(`Signal length must be ${this.size}`);
    }

    // Convert to Float32Array if needed
    const input = signal instanceof Float32Array ? signal : new Float32Array(signal);
    
    // Initialize complex arrays
    const real = new Float32Array(this.size);
    const imag = new Float32Array(this.size);
    
    // Bit-reversal permutation
    for (let i = 0; i < this.size; i++) {
      real[this.bitReversalTable[i]] = input[i];
      imag[this.bitReversalTable[i]] = 0;
    }
    
    // Cooley-Tukey FFT algorithm
    this.cooleyTukeyFFT(real, imag);
    
    // Compute magnitude and phase
    const magnitude = new Float32Array(this.size);
    const phase = new Float32Array(this.size);
    
    for (let i = 0; i < this.size; i++) {
      magnitude[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
      phase[i] = Math.atan2(imag[i], real[i]);
    }
    
    return { real, imag, magnitude, phase };
  }

  /**
   * Compute inverse FFT
   * @param {Float32Array} real - Real part
   * @param {Float32Array} imag - Imaginary part
   * @returns {Float32Array} Reconstructed signal
   */
  inverse(real, imag) {
    const realCopy = new Float32Array(real);
    const imagCopy = new Float32Array(imag);
    
    // Conjugate
    for (let i = 0; i < this.size; i++) {
      imagCopy[i] = -imagCopy[i];
    }
    
    // Forward FFT
    this.cooleyTukeyFFT(realCopy, imagCopy);
    
    // Conjugate and scale
    const output = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      output[i] = realCopy[i] / this.size;
    }
    
    return output;
  }

  /**
   * Cooley-Tukey FFT algorithm (in-place)
   * @private
   */
  cooleyTukeyFFT(real, imag) {
    let n = this.size;
    
    // Iterative FFT
    for (let s = 1; s <= this.log2Size; s++) {
      const m = 1 << s; // 2^s
      const m2 = m >> 1; // m/2
      
      for (let k = 0; k < n; k += m) {
        for (let j = 0; j < m2; j++) {
          const t = k + j + m2;
          const u = k + j;
          
          // Twiddle factor
          const twiddleIndex = (j * n / m) | 0;
          const wr = this.twiddleFactors.real[twiddleIndex];
          const wi = this.twiddleFactors.imag[twiddleIndex];
          
          // Butterfly operation
          const tr = wr * real[t] - wi * imag[t];
          const ti = wr * imag[t] + wi * real[t];
          
          real[t] = real[u] - tr;
          imag[t] = imag[u] - ti;
          real[u] += tr;
          imag[u] += ti;
        }
      }
    }
  }

  /**
   * Pre-compute twiddle factors (complex exponentials)
   * @private
   */
  computeTwiddleFactors() {
    const real = new Float32Array(this.size);
    const imag = new Float32Array(this.size);
    
    for (let i = 0; i < this.size; i++) {
      const angle = -2 * Math.PI * i / this.size;
      real[i] = Math.cos(angle);
      imag[i] = Math.sin(angle);
    }
    
    return { real, imag };
  }

  /**
   * Compute bit-reversal permutation table
   * @private
   */
  computeBitReversalTable() {
    const table = new Uint32Array(this.size);
    
    for (let i = 0; i < this.size; i++) {
      table[i] = this.reverseBits(i, this.log2Size);
    }
    
    return table;
  }

  /**
   * Reverse bits of a number
   * @private
   */
  reverseBits(x, bits) {
    let result = 0;
    for (let i = 0; i < bits; i++) {
      result = (result << 1) | (x & 1);
      x >>= 1;
    }
    return result;
  }

  /**
   * Compute power spectrum (magnitude squared)
   * @param {Float32Array|Array} signal - Input signal
   * @returns {Float32Array} Power spectrum
   */
  powerSpectrum(signal) {
    const { magnitude } = this.forward(signal);
    const power = new Float32Array(this.size);
    
    for (let i = 0; i < this.size; i++) {
      power[i] = magnitude[i] * magnitude[i];
    }
    
    return power;
  }

  /**
   * Compute power spectral density
   * @param {Float32Array|Array} signal - Input signal
   * @param {number} sampleRate - Sample rate in Hz
   * @returns {Object} {frequencies: Float32Array, psd: Float32Array}
   */
  powerSpectralDensity(signal, sampleRate) {
    const power = this.powerSpectrum(signal);
    const frequencies = new Float32Array(this.size / 2);
    const psd = new Float32Array(this.size / 2);
    
    const df = sampleRate / this.size;
    
    for (let i = 0; i < this.size / 2; i++) {
      frequencies[i] = i * df;
      psd[i] = power[i] / (sampleRate * this.size);
    }
    
    return { frequencies, psd };
  }
}

/**
 * Utility function for quick FFT computation
 * @param {Float32Array|Array} signal - Input signal
 * @param {number} [size] - FFT size (defaults to next power of 2)
 * @returns {Object} FFT result
 */
export function fft(signal, size = null) {
  const n = size || nextPowerOf2(signal.length);
  const fftInstance = new FFT(n);
  
  // Zero-pad if necessary
  if (signal.length < n) {
    const padded = new Float32Array(n);
    padded.set(signal);
    return fftInstance.forward(padded);
  }
  
  return fftInstance.forward(signal);
}

/**
 * Find next power of 2
 * @param {number} x - Input number
 * @returns {number} Next power of 2
 */
function nextPowerOf2(x) {
  return Math.pow(2, Math.ceil(Math.log2(x)));
}

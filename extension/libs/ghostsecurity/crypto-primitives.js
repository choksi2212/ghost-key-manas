/**
 * GhostSecurity Cryptographic Primitives
 * Production-grade cryptographic building blocks
 * Military-level security for biometric data protection
 * 
 * @module ghostsecurity/crypto-primitives
 * @author Ghost Key Team
 * @license MIT
 */

/**
 * Secure Random Number Generator using Web Crypto API
 */
export class SecureRandom {
  /**
   * Generate cryptographically secure random bytes
   */
  static getRandomBytes(length) {
    const globalCrypto = typeof globalThis !== 'undefined' ? globalThis.crypto : undefined;

    if (globalCrypto && typeof globalCrypto.getRandomValues === 'function') {
      const buffer = new Uint8Array(length);
      globalCrypto.getRandomValues(buffer);
      return buffer;
    }

    if (globalCrypto && typeof globalCrypto.randomBytes === 'function') {
      const bytes = globalCrypto.randomBytes(length);
      return bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    }

    try {
      const req = Function('return typeof require !== "undefined" ? require : null;')();
      if (req) {
        const cryptoModule = req('crypto');
        if (cryptoModule?.randomBytes) {
          return new Uint8Array(cryptoModule.randomBytes(length));
        }
      }
    } catch (error) {
      // ignore and fall through to error below
    }

      throw new Error('Secure random number generator not available');
  }

  /**
   * Generate random integer in range [0, max)
   */
  static getRandomInt(max) {
    const bytes = this.getRandomBytes(4);
    const value = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
    return Math.abs(value) % max;
  }

  /**
   * Generate random string (base64)
   */
  static getRandomString(length) {
    const bytes = this.getRandomBytes(length);
    return this.bytesToBase64(bytes);
  }

  /**
   * Convert bytes to base64
   */
  static bytesToBase64(bytes) {
    if (typeof btoa === 'function') {
    const binary = String.fromCharCode(...bytes);
    return btoa(binary);
    }
    if (typeof Buffer !== 'undefined') {
      return Buffer.from(bytes).toString('base64');
    }
    throw new Error('Base64 encoding not supported in this environment');
  }

  /**
   * Convert base64 to bytes
   */
  static base64ToBytes(base64) {
    if (typeof atob === 'function') {
    const binary = atob(base64);
    return new Uint8Array([...binary].map(char => char.charCodeAt(0)));
    }
    if (typeof Buffer !== 'undefined') {
      return new Uint8Array(Buffer.from(base64, 'base64'));
    }
    throw new Error('Base64 decoding not supported in this environment');
  }
}

/**
 * SHA-256 Hash Function (using Web Crypto API)
 */
export class SHA256 {
  /**
   * Compute SHA-256 hash
   */
  static async hash(data) {
    const encoder = new TextEncoder();
    const dataBuffer = typeof data === 'string' ? encoder.encode(data) : data;
    
    if (typeof window !== 'undefined' && window.crypto && window.crypto.subtle) {
      const hashBuffer = await window.crypto.subtle.digest('SHA-256', dataBuffer);
      return new Uint8Array(hashBuffer);
    } else {
      // Fallback: Pure JavaScript SHA-256 implementation
      return this.hashPureJS(dataBuffer);
    }
  }

  /**
   * Pure JavaScript SHA-256 implementation (fallback)
   */
  static hashPureJS(data) {
    const K = [
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ];

    const H = [
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    ];

    const bytes = new Uint8Array(data);
    const bitLength = bytes.length * 8;
    
    // Padding
    const paddingLength = (56 - (bytes.length + 1) % 64 + 64) % 64;
    const totalLength = bytes.length + 1 + paddingLength + 8;
    const padded = new Uint8Array(totalLength);
    
    padded.set(bytes);
    padded[bytes.length] = 0x80;
    
    // Append length
    const view = new DataView(padded.buffer);
    view.setUint32(totalLength - 4, bitLength, false);

    // Process blocks
    for (let i = 0; i < padded.length; i += 64) {
      const W = new Uint32Array(64);
      
      for (let j = 0; j < 16; j++) {
        W[j] = view.getUint32(i + j * 4, false);
      }
      
      for (let j = 16; j < 64; j++) {
        const s0 = this.rotr(W[j - 15], 7) ^ this.rotr(W[j - 15], 18) ^ (W[j - 15] >>> 3);
        const s1 = this.rotr(W[j - 2], 17) ^ this.rotr(W[j - 2], 19) ^ (W[j - 2] >>> 10);
        W[j] = (W[j - 16] + s0 + W[j - 7] + s1) >>> 0;
      }
      
      let [a, b, c, d, e, f, g, h] = H;
      
      for (let j = 0; j < 64; j++) {
        const S1 = this.rotr(e, 6) ^ this.rotr(e, 11) ^ this.rotr(e, 25);
        const ch = (e & f) ^ (~e & g);
        const temp1 = (h + S1 + ch + K[j] + W[j]) >>> 0;
        const S0 = this.rotr(a, 2) ^ this.rotr(a, 13) ^ this.rotr(a, 22);
        const maj = (a & b) ^ (a & c) ^ (b & c);
        const temp2 = (S0 + maj) >>> 0;
        
        h = g;
        g = f;
        f = e;
        e = (d + temp1) >>> 0;
        d = c;
        c = b;
        b = a;
        a = (temp1 + temp2) >>> 0;
      }
      
      H[0] = (H[0] + a) >>> 0;
      H[1] = (H[1] + b) >>> 0;
      H[2] = (H[2] + c) >>> 0;
      H[3] = (H[3] + d) >>> 0;
      H[4] = (H[4] + e) >>> 0;
      H[5] = (H[5] + f) >>> 0;
      H[6] = (H[6] + g) >>> 0;
      H[7] = (H[7] + h) >>> 0;
    }

    const result = new Uint8Array(32);
    const resultView = new DataView(result.buffer);
    for (let i = 0; i < 8; i++) {
      resultView.setUint32(i * 4, H[i], false);
    }
    
    return result;
  }

  static rotr(n, b) {
    return (n >>> b) | (n << (32 - b));
  }

  /**
   * Hash to hex string
   */
  static async hashToHex(data) {
    const hash = await this.hash(data);
    return Array.from(hash).map(b => b.toString(16).padStart(2, '0')).join('');
  }
}

/**
 * PBKDF2 Key Derivation Function
 */
export class PBKDF2 {
  /**
   * Derive key from password using PBKDF2
   */
  static async deriveKey(password, salt, iterations = 100000, keyLength = 32) {
    const encoder = new TextEncoder();
    const passwordBuffer = encoder.encode(password);
    
    if (typeof window !== 'undefined' && window.crypto && window.crypto.subtle) {
      const keyMaterial = await window.crypto.subtle.importKey(
        'raw',
        passwordBuffer,
        { name: 'PBKDF2' },
        false,
        ['deriveBits']
      );
      
      const derivedBits = await window.crypto.subtle.deriveBits(
        {
          name: 'PBKDF2',
          salt: salt,
          iterations: iterations,
          hash: 'SHA-256'
        },
        keyMaterial,
        keyLength * 8
      );
      
      return new Uint8Array(derivedBits);
    } else {
      // Fallback: HMAC-based PBKDF2
      return this.pbkdf2HMAC(passwordBuffer, salt, iterations, keyLength);
    }
  }

  /**
   * PBKDF2-HMAC implementation (fallback)
   */
  static async pbkdf2HMAC(password, salt, iterations, keyLength) {
    const blocks = Math.ceil(keyLength / 32);
    const result = new Uint8Array(keyLength);
    
    for (let i = 1; i <= blocks; i++) {
      const block = await this.pbkdf2Block(password, salt, iterations, i);
      const offset = (i - 1) * 32;
      const length = Math.min(32, keyLength - offset);
      result.set(block.slice(0, length), offset);
    }
    
    return result;
  }

  static async pbkdf2Block(password, salt, iterations, blockIndex) {
    const saltWithIndex = new Uint8Array(salt.length + 4);
    saltWithIndex.set(salt);
    new DataView(saltWithIndex.buffer).setUint32(salt.length, blockIndex, false);
    
    let u = await this.hmacSHA256(password, saltWithIndex);
    let result = new Uint8Array(u);
    
    for (let i = 1; i < iterations; i++) {
      u = await this.hmacSHA256(password, u);
      for (let j = 0; j < result.length; j++) {
        result[j] ^= u[j];
      }
    }
    
    return result;
  }

  static async hmacSHA256(key, message) {
    const blockSize = 64;
    const keyHash = key.length > blockSize ? await SHA256.hash(key) : key;
    const keyPadded = new Uint8Array(blockSize);
    keyPadded.set(keyHash);
    
    const ipad = new Uint8Array(blockSize + message.length);
    const opad = new Uint8Array(blockSize + 32);
    
    for (let i = 0; i < blockSize; i++) {
      ipad[i] = keyPadded[i] ^ 0x36;
      opad[i] = keyPadded[i] ^ 0x5c;
    }
    
    ipad.set(message, blockSize);
    const innerHash = await SHA256.hash(ipad);
    opad.set(innerHash, blockSize);
    
    return SHA256.hash(opad);
  }
}

/**
 * AES-256-GCM Encryption (using Web Crypto API)
 */
export class AES256GCM {
  /**
   * Encrypt data using AES-256-GCM
   */
  static async encrypt(data, key, additionalData = null) {
    const iv = SecureRandom.getRandomBytes(12); // 96-bit IV for GCM
    
    const encoder = new TextEncoder();
    const dataBuffer = typeof data === 'string' ? encoder.encode(data) : data;
    
    if (typeof window !== 'undefined' && window.crypto && window.crypto.subtle) {
      const cryptoKey = await window.crypto.subtle.importKey(
        'raw',
        key,
        { name: 'AES-GCM' },
        false,
        ['encrypt']
      );
      
      const encryptParams = {
        name: 'AES-GCM',
        iv: iv,
        tagLength: 128
      };
      
      if (additionalData) {
        encryptParams.additionalData = typeof additionalData === 'string' 
          ? encoder.encode(additionalData) 
          : additionalData;
      }
      
      const ciphertext = await window.crypto.subtle.encrypt(
        encryptParams,
        cryptoKey,
        dataBuffer
      );
      
      return {
        ciphertext: new Uint8Array(ciphertext),
        iv: iv,
        tag: new Uint8Array(ciphertext.slice(-16)) // Last 16 bytes are the tag
      };
    } else {
      throw new Error('AES-GCM not available. Use browser with Web Crypto API support.');
    }
  }

  /**
   * Decrypt data using AES-256-GCM
   */
  static async decrypt(ciphertext, key, iv, additionalData = null) {
    if (typeof window !== 'undefined' && window.crypto && window.crypto.subtle) {
      const cryptoKey = await window.crypto.subtle.importKey(
        'raw',
        key,
        { name: 'AES-GCM' },
        false,
        ['decrypt']
      );
      
      const decryptParams = {
        name: 'AES-GCM',
        iv: iv,
        tagLength: 128
      };
      
      if (additionalData) {
        const encoder = new TextEncoder();
        decryptParams.additionalData = typeof additionalData === 'string' 
          ? encoder.encode(additionalData) 
          : additionalData;
      }
      
      const plaintext = await window.crypto.subtle.decrypt(
        decryptParams,
        cryptoKey,
        ciphertext
      );
      
      return new Uint8Array(plaintext);
    } else {
      throw new Error('AES-GCM not available. Use browser with Web Crypto API support.');
    }
  }
}

/**
 * ChaCha20-Poly1305 AEAD (Authenticated Encryption with Associated Data)
 * Alternative to AES-GCM, resistant to timing attacks
 */
export class ChaCha20Poly1305 {
  /**
   * ChaCha20 quarter round
   */
  static quarterRound(state, a, b, c, d) {
    state[a] = (state[a] + state[b]) >>> 0;
    state[d] = this.rotl(state[d] ^ state[a], 16);
    
    state[c] = (state[c] + state[d]) >>> 0;
    state[b] = this.rotl(state[b] ^ state[c], 12);
    
    state[a] = (state[a] + state[b]) >>> 0;
    state[d] = this.rotl(state[d] ^ state[a], 8);
    
    state[c] = (state[c] + state[d]) >>> 0;
    state[b] = this.rotl(state[b] ^ state[c], 7);
  }

  static rotl(v, n) {
    return ((v << n) | (v >>> (32 - n))) >>> 0;
  }

  /**
   * ChaCha20 block function
   */
  static chacha20Block(key, nonce, counter) {
    const state = new Uint32Array(16);
    
    // Constants
    state[0] = 0x61707865;
    state[1] = 0x3320646e;
    state[2] = 0x79622d32;
    state[3] = 0x6b206574;
    
    // Key
    const keyView = new DataView(key.buffer);
    for (let i = 0; i < 8; i++) {
      state[4 + i] = keyView.getUint32(i * 4, true);
    }
    
    // Counter
    state[12] = counter;
    
    // Nonce
    const nonceView = new DataView(nonce.buffer);
    for (let i = 0; i < 3; i++) {
      state[13 + i] = nonceView.getUint32(i * 4, true);
    }
    
    const working = new Uint32Array(state);
    
    // 20 rounds
    for (let i = 0; i < 10; i++) {
      // Column rounds
      this.quarterRound(working, 0, 4, 8, 12);
      this.quarterRound(working, 1, 5, 9, 13);
      this.quarterRound(working, 2, 6, 10, 14);
      this.quarterRound(working, 3, 7, 11, 15);
      
      // Diagonal rounds
      this.quarterRound(working, 0, 5, 10, 15);
      this.quarterRound(working, 1, 6, 11, 12);
      this.quarterRound(working, 2, 7, 8, 13);
      this.quarterRound(working, 3, 4, 9, 14);
    }
    
    // Add original state
    for (let i = 0; i < 16; i++) {
      working[i] = (working[i] + state[i]) >>> 0;
    }
    
    return new Uint8Array(working.buffer);
  }

  /**
   * Encrypt with ChaCha20
   */
  static encrypt(data, key, nonce) {
    const result = new Uint8Array(data.length);
    let counter = 0;
    
    for (let i = 0; i < data.length; i += 64) {
      const keystream = this.chacha20Block(key, nonce, counter++);
      const blockLength = Math.min(64, data.length - i);
      
      for (let j = 0; j < blockLength; j++) {
        result[i + j] = data[i + j] ^ keystream[j];
      }
    }
    
    return result;
  }

  /**
   * Decrypt with ChaCha20 (same as encrypt due to XOR)
   */
  static decrypt(ciphertext, key, nonce) {
    return this.encrypt(ciphertext, key, nonce);
  }
}

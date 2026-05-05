// src/components/LandingScene.jsx
import { useRef, useMemo, useState, useCallback } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import * as THREE from 'three'
import { uploadVideo } from '../api.js'

// ── 3D: Floating audio waveform ring ─────────────────────────────────────────
function WaveRing({ radius = 3, points = 128, color = '#00ffc8' }) {
  const meshRef = useRef()
  const timeRef = useRef(0)

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const positions = new Float32Array(points * 3)
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    return geo
  }, [points])

  useFrame((_, delta) => {
    timeRef.current += delta * 1.2
    const t = timeRef.current
    const pos = geometry.attributes.position.array

    for (let i = 0; i < points; i++) {
      const angle = (i / points) * Math.PI * 2
      const wave  = Math.sin(angle * 6 + t * 2) * 0.18
               + Math.sin(angle * 11 + t * 1.3) * 0.10
               + Math.sin(angle * 3  - t * 0.9) * 0.14
      const r = radius + wave
      pos[i * 3]     = Math.cos(angle) * r
      pos[i * 3 + 1] = Math.sin(angle) * r
      pos[i * 3 + 2] = Math.sin(angle * 4 + t) * 0.3
    }
    geometry.attributes.position.needsUpdate = true
    if (meshRef.current) meshRef.current.rotation.z += delta * 0.08
  })

  return (
    <line ref={meshRef} geometry={geometry}>
      <lineBasicMaterial color={color} transparent opacity={0.85} />
    </line>
  )
}

// ── 3D: Particle field ────────────────────────────────────────────────────────
function ParticleField({ count = 600 }) {
  const meshRef = useRef()

  const [positions, sizes] = useMemo(() => {
    const pos  = new Float32Array(count * 3)
    const size = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      pos[i * 3]     = (Math.random() - 0.5) * 18
      pos[i * 3 + 1] = (Math.random() - 0.5) * 18
      pos[i * 3 + 2] = (Math.random() - 0.5) * 18
      size[i]        = Math.random() * 0.04 + 0.01
    }
    return [pos, size]
  }, [count])

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.04
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.02) * 0.1
    }
  })

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" array={positions} count={count} itemSize={3} />
        <bufferAttribute attach="attributes-size"     array={sizes}     count={count} itemSize={1} />
      </bufferGeometry>
      <pointsMaterial color="#00ffc8" size={0.04} transparent opacity={0.4} sizeAttenuation />
    </points>
  )
}

// ── 3D: Rotating torus knot ───────────────────────────────────────────────────
function GlowKnot() {
  const ref = useRef()
  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.x = state.clock.elapsedTime * 0.3
      ref.current.rotation.y = state.clock.elapsedTime * 0.2
    }
  })
  return (
    <mesh ref={ref}>
      <torusKnotGeometry args={[1.1, 0.28, 180, 20, 2, 3]} />
      <meshStandardMaterial
        color="#00ffc8"
        emissive="#003322"
        emissiveIntensity={0.6}
        wireframe
        transparent
        opacity={0.35}
      />
    </mesh>
  )
}

// ── Upload dropzone UI ────────────────────────────────────────────────────────
function UploadZone({ onUploadSuccess }) {
  const [uploading, setUploading] = useState(false)
  const [error,     setError]     = useState(null)

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      const res = await uploadVideo(file)
      onUploadSuccess(res.job_id)
    } catch (e) {
      setError(e.message || 'Upload failed. Is the backend running?')
      setUploading(false)
    }
  }, [onUploadSuccess])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm'] },
    maxFiles: 1,
    disabled: uploading,
  })

  return (
    <div style={styles.uploadOuter}>
      <div
        {...getRootProps()}
        style={{
          ...styles.dropzone,
          borderColor: isDragActive ? 'var(--cyan)' : 'var(--border)',
          boxShadow:   isDragActive ? 'var(--glow-cyan)' : 'none',
          background:  isDragActive ? 'var(--cyan-subtle)' : 'var(--bg-card)',
          cursor: uploading ? 'not-allowed' : 'pointer',
        }}
      >
        <input {...getInputProps()} />

        <AnimatePresence mode="wait">
          {uploading ? (
            <motion.div key="uploading" style={styles.dropContent}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <UploadSpinner />
              <p style={styles.dropLabel}>Uploading...</p>
            </motion.div>
          ) : (
            <motion.div key="idle" style={styles.dropContent}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <VideoIcon active={isDragActive} />
              <p style={styles.dropLabel}>
                {isDragActive ? 'Drop to analyse' : 'Drop video here'}
              </p>
              <p style={styles.dropSub}>MP4 · MOV · AVI · MKV · WebM</p>
              <button style={styles.browseBtn}>Browse files</button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {error && (
        <motion.p initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
          style={styles.errorMsg}>
          {error}
        </motion.p>
      )}
    </div>
  )
}

function VideoIcon({ active }) {
  return (
    <svg width="48" height="48" viewBox="0 0 48 48" fill="none"
      style={{ marginBottom: 12, filter: active ? 'drop-shadow(0 0 8px #00ffc8)' : 'none',
               transition: 'filter 0.3s' }}>
      <rect x="2" y="10" width="32" height="28" rx="3" stroke="#00ffc8" strokeWidth="2" fill="none"/>
      <path d="M34 19l12-7v24l-12-7V19z" stroke="#00ffc8" strokeWidth="2" fill="none"/>
      <circle cx="16" cy="24" r="4" fill="rgba(0,255,200,0.2)" stroke="#00ffc8" strokeWidth="1.5"/>
    </svg>
  )
}

function UploadSpinner() {
  return (
    <div style={{
      width: 48, height: 48, marginBottom: 12,
      border: '2px solid rgba(0,255,200,0.2)',
      borderTop: '2px solid #00ffc8',
      borderRadius: '50%',
      animation: 'spin 0.8s linear infinite',
    }} />
  )
}

// ── Main landing page ─────────────────────────────────────────────────────────
export default function LandingScene({ onUploadSuccess }) {
  return (
    <div style={styles.root}>
      {/* Three.js canvas — full background */}
      <div style={styles.canvasWrap}>
        <Canvas camera={{ position: [0, 0, 7], fov: 55 }}>
          <ambientLight intensity={0.2} />
          <pointLight position={[4, 4, 4]}  color="#00ffc8" intensity={2} />
          <pointLight position={[-4, -4, 2]} color="#0044ff" intensity={1} />
          <GlowKnot />
          <WaveRing radius={3.8} points={160} color="#00ffc8" />
          <WaveRing radius={5.2} points={120} color="#0066ff" />
          <ParticleField count={700} />
        </Canvas>
      </div>

      {/* Overlay UI */}
      <div style={styles.overlay}>
        {/* Header */}
        <motion.div style={styles.header}
          initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}>
          <span style={styles.logo}>VISIO<span style={{ color: 'var(--cyan)' }}>VOX</span></span>
          <span style={styles.version}>v1.0 · PROTOTYPE</span>
        </motion.div>

        {/* Hero text */}
        <motion.div style={styles.hero}
          initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.4 }}>
          <h1 style={styles.heroTitle}>
            Hear Every<br />
            <span style={{ color: 'var(--cyan)', textShadow: '0 0 30px rgba(0,255,200,0.5)' }}>
              Speaker
            </span>
          </h1>
          <p style={styles.heroSub}>
            Audio-visual speech separation · Upload a multi-speaker video<br />
            and isolate any speaker with one click
          </p>
          <div style={styles.gradientLine} />
        </motion.div>

        {/* Upload zone */}
        <motion.div
          initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.7 }}>
          <UploadZone onUploadSuccess={onUploadSuccess} />
        </motion.div>

        {/* Footer badges */}
        <motion.div style={styles.badges}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}>
          {['Audio-Visual Fusion', 'Whisper ASR', 'Real-time Switching'].map(b => (
            <span key={b} style={styles.badge}>{b}</span>
          ))}
        </motion.div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = {
  root: {
    position: 'relative', width: '100%', height: '100vh',
    overflow: 'hidden', background: 'var(--bg)',
  },
  canvasWrap: {
    position: 'absolute', inset: 0,
  },
  overlay: {
    position: 'absolute', inset: 0,
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    padding: '24px',
    background: 'radial-gradient(ellipse at center, rgba(0,10,8,0.4) 0%, rgba(5,5,8,0.85) 70%)',
  },
  header: {
    position: 'absolute', top: 24, left: 32,
    display: 'flex', alignItems: 'baseline', gap: 12,
  },
  logo: {
    fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 900,
    letterSpacing: '0.12em', color: 'var(--white)',
  },
  version: {
    fontFamily: 'var(--font-body)', fontSize: 11,
    color: 'var(--muted)', letterSpacing: '0.08em',
  },
  hero: {
    textAlign: 'center', marginBottom: 40,
  },
  heroTitle: {
    fontFamily: 'var(--font-display)', fontSize: 'clamp(42px, 7vw, 80px)',
    fontWeight: 900, lineHeight: 1.1, letterSpacing: '-0.02em',
    color: 'var(--white)', marginBottom: 18,
  },
  heroSub: {
    fontFamily: 'var(--font-body)', fontSize: 14,
    color: 'var(--muted)', lineHeight: 1.8, marginBottom: 24,
  },
  gradientLine: {
    height: 1, width: 200, margin: '0 auto',
    background: 'linear-gradient(90deg, transparent, var(--cyan), transparent)',
    opacity: 0.5,
  },
  uploadOuter: {
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12,
  },
  dropzone: {
    width: 380, padding: '36px 32px',
    border: '1px dashed', borderRadius: 'var(--radius-lg)',
    transition: 'all 0.3s ease',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  dropContent: {
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', textAlign: 'center',
  },
  dropLabel: {
    fontFamily: 'var(--font-display)', fontSize: 13,
    color: 'var(--white)', letterSpacing: '0.06em', marginBottom: 4,
  },
  dropSub: {
    fontSize: 11, color: 'var(--muted)', marginBottom: 20,
    letterSpacing: '0.1em',
  },
  browseBtn: {
    background: 'transparent', border: '1px solid var(--border-glow)',
    color: 'var(--cyan)', fontFamily: 'var(--font-display)',
    fontSize: 11, letterSpacing: '0.1em',
    padding: '8px 20px', borderRadius: 4,
    transition: 'all 0.2s',
  },
  errorMsg: {
    color: 'var(--red)', fontSize: 12,
    fontFamily: 'var(--font-body)', textAlign: 'center',
  },
  badges: {
    position: 'absolute', bottom: 28,
    display: 'flex', gap: 12,
  },
  badge: {
    fontFamily: 'var(--font-display)', fontSize: 10,
    letterSpacing: '0.1em', padding: '5px 12px',
    border: '1px solid var(--border)', borderRadius: 20,
    color: 'var(--muted)', background: 'var(--bg-card)',
  },
}
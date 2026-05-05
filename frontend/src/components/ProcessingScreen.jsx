// src/components/ProcessingScreen.jsx
import { useEffect, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { motion } from 'framer-motion'
import * as THREE from 'three'
import { getStatus } from '../api.js'

const POLL_INTERVAL_MS = 2500

const STATUS_LABELS = {
  uploaded:      'Initialising pipeline...',
  preprocessing: 'Detecting faces & extracting lip crops...',
  separating:    'Isolating speaker audio...',
  transcribing:  'Generating captions with Whisper...',
  ready:         'Pre-processing complete!',
  done:          'All done!',
  error:         'Something went wrong.',
}

const STATUS_ORDER = ['uploaded', 'preprocessing', 'ready']

function statusProgress(status) {
  const map = { uploaded: 10, preprocessing: 45, ready: 100,
                separating: 60, transcribing: 85, done: 100, error: 0 }
  return map[status] ?? 10
}

// ── 3D spinner ────────────────────────────────────────────────────────────────
function DNARing({ radius, speed, color, waveAmp = 0.25, index = 0 }) {
  const ref = useRef()
  const t   = useRef(0)
  const POINTS = 80

  const geo = new THREE.BufferGeometry()
  const pos = new Float32Array(POINTS * 3)
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))

  useFrame((_, delta) => {
    t.current += delta * speed
    const arr = geo.attributes.position.array
    for (let i = 0; i < POINTS; i++) {
      const angle = (i / POINTS) * Math.PI * 2
      const wave  = Math.sin(angle * 5 + t.current + index) * waveAmp
      const r     = radius + wave
      arr[i * 3]     = Math.cos(angle) * r
      arr[i * 3 + 1] = Math.sin(angle) * r
      arr[i * 3 + 2] = Math.sin(angle * 3 + t.current * 0.7) * 0.2
    }
    geo.attributes.position.needsUpdate = true
    if (ref.current) ref.current.rotation.z += delta * 0.15
  })

  return (
    <line ref={ref} geometry={geo}>
      <lineBasicMaterial color={color} transparent opacity={0.7} />
    </line>
  )
}

function ProcessingScene() {
  return (
    <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
      <ambientLight intensity={0.1} />
      <pointLight position={[3, 3, 3]} color="#00ffc8" intensity={3} />
      <DNARing radius={1.6} speed={1.8} color="#00ffc8" waveAmp={0.3}  index={0} />
      <DNARing radius={2.3} speed={1.2} color="#0055ff" waveAmp={0.2}  index={2} />
      <DNARing radius={3.0} speed={0.8} color="#00ffc8" waveAmp={0.15} index={4} />
    </Canvas>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function ProcessingScreen({ jobId, onComplete }) {
  const [status,  setStatus]  = useState('uploaded')
  const [message, setMessage] = useState('Starting...')
  const [error,   setError]   = useState(null)
  const intervalRef = useRef(null)

  useEffect(() => {
    if (!jobId) return

    const poll = async () => {
      try {
        const data = await getStatus(jobId)
        setStatus(data.status)
        setMessage(STATUS_LABELS[data.status] || data.progress_msg)

        if (data.status === 'error') {
          setError(data.error || 'Unknown error')
          clearInterval(intervalRef.current)
          return
        }

        // Preprocessing complete — ready for player
        if (data.status === 'ready') {
          clearInterval(intervalRef.current)
          setTimeout(() => onComplete(data), 800)
        }
      } catch (e) {
        setError(e.message)
        clearInterval(intervalRef.current)
      }
    }

    poll()
    intervalRef.current = setInterval(poll, POLL_INTERVAL_MS)
    return () => clearInterval(intervalRef.current)
  }, [jobId])

  const progress = statusProgress(status)

  return (
    <div style={styles.root}>
      {/* 3D background */}
      <div style={styles.canvasWrap}>
        <ProcessingScene />
      </div>

      {/* Overlay */}
      <div style={styles.overlay}>
        <motion.div style={styles.card}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}>

          {/* Logo */}
          <div style={styles.logoRow}>
            <span style={styles.logo}>
              VISIO<span style={{ color: 'var(--cyan)' }}>VOX</span>
            </span>
          </div>

          <div style={styles.gradientLine} />

          {/* Status */}
          <motion.p style={styles.statusText} key={message}
            initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
            {message}
          </motion.p>

          {/* Progress bar */}
          <div style={styles.barTrack}>
            <motion.div style={styles.barFill}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }} />
            <div style={styles.barGlow}
              dangerouslySetInnerHTML={{ __html: '' }} />
          </div>
          <p style={styles.progressPct}>{progress}%</p>

          {/* Step indicators */}
          <div style={styles.steps}>
            {STATUS_ORDER.map((s, i) => {
              const done    = statusProgress(status) >= statusProgress(s)
              const current = status === s
              return (
                <div key={s} style={styles.stepRow}>
                  <div style={{
                    ...styles.stepDot,
                    background:   done ? 'var(--cyan)' : 'var(--bg-2)',
                    borderColor:  done ? 'var(--cyan)' : 'var(--muted)',
                    boxShadow:    done ? 'var(--glow-cyan)' : 'none',
                  }} />
                  <span style={{
                    ...styles.stepLabel,
                    color: done ? 'var(--white)' : 'var(--muted)',
                  }}>
                    {STATUS_LABELS[s]}
                  </span>
                  {current && (
                    <motion.span style={styles.pulse}
                      animate={{ opacity: [1, 0.3, 1] }}
                      transition={{ repeat: Infinity, duration: 1.2 }}>
                      ●
                    </motion.span>
                  )}
                </div>
              )
            })}
          </div>

          {error && (
            <motion.div style={styles.errorBox}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <span style={{ color: 'var(--red)' }}>ERROR:</span> {error}
            </motion.div>
          )}

          {/* Job ID */}
          <p style={styles.jobId}>JOB · {jobId?.slice(0, 8).toUpperCase()}</p>
        </motion.div>
      </div>
    </div>
  )
}

const styles = {
  root: {
    position: 'relative', width: '100%', height: '100vh',
    overflow: 'hidden', background: 'var(--bg)',
  },
  canvasWrap: { position: 'absolute', inset: 0 },
  overlay: {
    position: 'absolute', inset: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'radial-gradient(ellipse at center, rgba(0,5,4,0.3) 0%, rgba(5,5,8,0.8) 70%)',
  },
  card: {
    background: 'rgba(10,10,18,0.92)',
    border: '1px solid var(--border)', borderRadius: 20,
    padding: '40px 44px', width: 420, backdropFilter: 'blur(20px)',
    boxShadow: '0 0 60px rgba(0,255,200,0.08)',
  },
  logoRow: { marginBottom: 16 },
  logo: {
    fontFamily: 'var(--font-display)', fontSize: 20,
    fontWeight: 900, letterSpacing: '0.12em',
  },
  gradientLine: {
    height: 1, width: '100%', marginBottom: 24,
    background: 'linear-gradient(90deg, transparent, var(--cyan) 50%, transparent)',
    opacity: 0.3,
  },
  statusText: {
    fontFamily: 'var(--font-body)', fontSize: 13,
    color: 'var(--cyan)', marginBottom: 16, minHeight: 20,
  },
  barTrack: {
    height: 4, background: 'var(--bg-2)',
    borderRadius: 2, overflow: 'hidden',
    marginBottom: 6, position: 'relative',
  },
  barFill: {
    height: '100%', background: 'var(--cyan)',
    borderRadius: 2,
    boxShadow: '0 0 10px var(--cyan)',
  },
  barGlow: { position: 'absolute', inset: 0 },
  progressPct: {
    fontFamily: 'var(--font-display)', fontSize: 11,
    color: 'var(--muted)', textAlign: 'right', marginBottom: 28,
  },
  steps: { display: 'flex', flexDirection: 'column', gap: 12 },
  stepRow: { display: 'flex', alignItems: 'center', gap: 12 },
  stepDot: {
    width: 10, height: 10, borderRadius: '50%',
    border: '1px solid', flexShrink: 0, transition: 'all 0.4s',
  },
  stepLabel: {
    fontFamily: 'var(--font-body)', fontSize: 12,
    transition: 'color 0.4s', flex: 1,
  },
  pulse: { color: 'var(--cyan)', fontSize: 10 },
  errorBox: {
    marginTop: 20, padding: '10px 14px',
    background: 'rgba(255,58,92,0.08)',
    border: '1px solid rgba(255,58,92,0.3)',
    borderRadius: 6, fontSize: 12,
    fontFamily: 'var(--font-body)', color: 'var(--white)',
  },
  jobId: {
    marginTop: 28, fontSize: 10, color: 'var(--muted)',
    fontFamily: 'var(--font-display)', letterSpacing: '0.15em',
    textAlign: 'right',
  },
}
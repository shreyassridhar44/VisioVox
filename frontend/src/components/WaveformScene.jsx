// src/components/WaveformScene.jsx
// Real-time 3D waveform that reacts to the currently playing audio.
// Uses Web Audio API AnalyserNode + Three.js BufferGeometry line.

import { useRef, useEffect, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'

const BAR_COUNT  = 64
const BAR_WIDTH  = 0.12
const BAR_GAP    = 0.06
const TOTAL_W    = BAR_COUNT * (BAR_WIDTH + BAR_GAP)

// ── Bars (frequency domain) ───────────────────────────────────────────────────
function FreqBars({ analyser, color }) {
  const groupRef  = useRef()
  const barsRef   = useRef([])
  const dataArray = useMemo(
    () => analyser ? new Uint8Array(analyser.frequencyBinCount) : null,
    [analyser]
  )

  useFrame(() => {
    if (!analyser || !dataArray) {
      // Idle animation when no audio
      barsRef.current.forEach((mesh, i) => {
        if (!mesh) return
        const t   = Date.now() / 1000
        const h   = 0.05 + Math.abs(Math.sin(t * 1.2 + i * 0.3)) * 0.25
        mesh.scale.y = h
        mesh.position.y = h / 2
      })
      return
    }

    analyser.getByteFrequencyData(dataArray)

    barsRef.current.forEach((mesh, i) => {
      if (!mesh) return
      const dataIdx = Math.floor(i * dataArray.length / BAR_COUNT)
      const norm    = dataArray[dataIdx] / 255
      const h       = Math.max(0.04, norm * 2.5)
      mesh.scale.y  = h
      mesh.position.y = h / 2
    })
  })

  return (
    <group ref={groupRef} position={[-TOTAL_W / 2, -1.2, 0]}>
      {Array.from({ length: BAR_COUNT }, (_, i) => (
        <mesh
          key={i}
          ref={el => barsRef.current[i] = el}
          position={[i * (BAR_WIDTH + BAR_GAP), 0, 0]}
        >
          <boxGeometry args={[BAR_WIDTH, 1, BAR_WIDTH]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.6}
            transparent
            opacity={0.85}
          />
        </mesh>
      ))}
    </group>
  )
}

// ── Floating waveform ring (decorative) ───────────────────────────────────────
function WaveRing({ analyser, radius = 3.2, color }) {
  const ref  = useRef()
  const POINTS = 100
  const geo  = useMemo(() => {
    const g   = new THREE.BufferGeometry()
    const pos = new Float32Array(POINTS * 3)
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    return g
  }, [])
  const dataArray = useMemo(
    () => analyser ? new Uint8Array(analyser.frequencyBinCount) : null,
    [analyser]
  )
  const t = useRef(0)

  useFrame((_, delta) => {
    t.current += delta
    const pos = geo.attributes.position.array
    const audioBoost = (() => {
      if (!analyser || !dataArray) return 1
      analyser.getByteFrequencyData(dataArray)
      const avg = dataArray.slice(0, 20).reduce((a, b) => a + b, 0) / 20
      return 1 + (avg / 255) * 0.8
    })()

    for (let i = 0; i < POINTS; i++) {
      const angle = (i / POINTS) * Math.PI * 2
      const wave  = Math.sin(angle * 6 + t.current * 2)   * 0.18 * audioBoost
               + Math.sin(angle * 11 + t.current * 1.4) * 0.10
      const r = radius + wave
      pos[i * 3]     = Math.cos(angle) * r
      pos[i * 3 + 1] = Math.sin(angle) * r
      pos[i * 3 + 2] = Math.sin(angle * 4 + t.current) * 0.25
    }
    geo.attributes.position.needsUpdate = true
    if (ref.current) ref.current.rotation.z += delta * 0.06
  })

  return (
    <line ref={ref} geometry={geo}>
      <lineBasicMaterial color={color} transparent opacity={0.5} />
    </line>
  )
}

function Scene({ analyser, color }) {
  return (
    <>
      <ambientLight intensity={0.2} />
      <pointLight position={[0, 3, 3]}  color={color} intensity={3} />
      <pointLight position={[0, -3, 2]} color="#0044ff" intensity={1.5} />
      <FreqBars analyser={analyser} color={color} />
      <WaveRing analyser={analyser} radius={3.5} color={color} />
    </>
  )
}

export default function WaveformScene({ audioRef, isPlaying, speakerColor = '#00ffc8' }) {
  const analyserRef = useRef(null)
  const ctxRef      = useRef(null)
  const sourceRef   = useRef(null)

  useEffect(() => {
    const audio = audioRef?.current
    if (!audio) return

    // Create AudioContext + AnalyserNode once
    if (!ctxRef.current) {
      ctxRef.current  = new (window.AudioContext || window.webkitAudioContext)()
      analyserRef.current = ctxRef.current.createAnalyser()
      analyserRef.current.fftSize = 128
      sourceRef.current  = ctxRef.current.createMediaElementSource(audio)
      sourceRef.current.connect(analyserRef.current)
      analyserRef.current.connect(ctxRef.current.destination)
    }

    // Resume context on play (browsers require user gesture)
    if (isPlaying && ctxRef.current.state === 'suspended') {
      ctxRef.current.resume()
    }
  }, [audioRef, isPlaying])

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Canvas camera={{ position: [0, 0, 6], fov: 55 }}>
        <Scene analyser={analyserRef.current} color={speakerColor} />
      </Canvas>
    </div>
  )
}
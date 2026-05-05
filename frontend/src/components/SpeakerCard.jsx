// src/components/SpeakerCard.jsx
import { useState } from 'react'
import { motion } from 'framer-motion'
import { getThumbnailUrl } from '../api.js'

// Colors per speaker slot
const SPEAKER_COLORS = ['#00ffc8', '#ffb800', '#ff3a5c', '#7c6fff']
const SPEAKER_NAMES  = ['Speaker A', 'Speaker B', 'Speaker C', 'Speaker D']

export default function SpeakerCard({
  jobId,
  faceId,
  isSelected,
  isProcessing,
  isProcessed,
  onClick,
}) {
  const [imgError, setImgError] = useState(false)
  const color = SPEAKER_COLORS[faceId] ?? '#00ffc8'
  const label = SPEAKER_NAMES[faceId]  ?? `Speaker ${faceId}`

  return (
    <motion.div
      onClick={onClick}
      whileHover={{ scale: 1.04 }}
      whileTap={{ scale: 0.97 }}
      style={{
        ...styles.card,
        borderColor:  isSelected ? color : 'var(--border)',
        boxShadow:    isSelected
          ? `0 0 24px ${color}55, 0 0 8px ${color}33`
          : 'none',
        cursor: isProcessing ? 'wait' : 'pointer',
        background: isSelected
          ? `linear-gradient(135deg, rgba(${hexToRgb(color)}, 0.10) 0%, var(--bg-card) 100%)`
          : 'var(--bg-card)',
      }}
    >
      {/* Thumbnail */}
      <div style={styles.thumbWrap}>
        {!imgError ? (
          <img
            src={getThumbnailUrl(jobId, faceId)}
            alt={label}
            style={{
              ...styles.thumb,
              filter: isSelected
                ? `drop-shadow(0 0 6px ${color})`
                : 'grayscale(40%) brightness(0.7)',
              transition: 'filter 0.3s',
            }}
            onError={() => setImgError(true)}
          />
        ) : (
          <div style={{ ...styles.thumbFallback, borderColor: color }}>
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <circle cx="16" cy="12" r="6" stroke={color} strokeWidth="1.5" fill="none"/>
              <path d="M4 28c0-6.627 5.373-12 12-12s12 5.373 12 12"
                stroke={color} strokeWidth="1.5" fill="none" strokeLinecap="round"/>
            </svg>
          </div>
        )}

        {/* Selected indicator */}
        {isSelected && (
          <motion.div
            style={{ ...styles.selectedBadge, background: color }}
            initial={{ scale: 0 }} animate={{ scale: 1 }}>
            <svg width="10" height="10" viewBox="0 0 10 10">
              <polyline points="2,5 4,7 8,3"
                stroke="#000" strokeWidth="1.5"
                fill="none" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </motion.div>
        )}

        {/* Processing spinner */}
        {isProcessing && !isSelected && (
          <div style={styles.spinnerOverlay}>
            <div style={{ ...styles.spinner, borderTopColor: color }} />
          </div>
        )}
      </div>

      {/* Label */}
      <p style={{ ...styles.label, color: isSelected ? color : 'var(--white)' }}>
        {label}
      </p>
      <p style={styles.sub}>
        {isProcessed
          ? 'Ready'
          : isProcessing
          ? 'Processing...'
          : 'Click to isolate'}
      </p>

      {/* Bottom accent line */}
      <motion.div
        style={{ ...styles.accentLine, background: color }}
        animate={{ scaleX: isSelected ? 1 : 0 }}
        transition={{ duration: 0.3 }}
      />
    </motion.div>
  )
}

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `${r},${g},${b}`
}

const styles = {
  card: {
    position: 'relative', overflow: 'hidden',
    border: '1px solid', borderRadius: 12,
    padding: '16px 12px', width: 120,
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', gap: 8,
    transition: 'border-color 0.3s, box-shadow 0.3s, background 0.3s',
    userSelect: 'none',
  },
  thumbWrap: {
    position: 'relative', width: 72, height: 72,
  },
  thumb: {
    width: 72, height: 72, borderRadius: 8,
    objectFit: 'cover', display: 'block',
  },
  thumbFallback: {
    width: 72, height: 72, borderRadius: 8,
    border: '1px solid', display: 'flex',
    alignItems: 'center', justifyContent: 'center',
    background: 'var(--bg-2)',
  },
  selectedBadge: {
    position: 'absolute', top: -4, right: -4,
    width: 18, height: 18, borderRadius: '50%',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  spinnerOverlay: {
    position: 'absolute', inset: 0, borderRadius: 8,
    background: 'rgba(5,5,8,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  spinner: {
    width: 24, height: 24, borderRadius: '50%',
    border: '2px solid rgba(255,255,255,0.1)',
    animation: 'spin 0.8s linear infinite',
  },
  label: {
    fontFamily: 'var(--font-display)', fontSize: 11,
    letterSpacing: '0.06em', textAlign: 'center',
    transition: 'color 0.3s',
  },
  sub: {
    fontFamily: 'var(--font-body)', fontSize: 10,
    color: 'var(--muted)', textAlign: 'center',
  },
  accentLine: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    height: 2, transformOrigin: 'left',
  },
}
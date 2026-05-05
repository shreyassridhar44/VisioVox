// src/components/PlayerScreen.jsx
import { useRef, useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import WaveformScene from './WaveformScene.jsx'
import SpeakerCard   from './SpeakerCard.jsx'
import {
  processSpeaker, getStatus,
  getAudioUrl, getCaptionsUrl, getVideoUrl,
} from '../api.js'

const SPEAKER_COLORS = ['#00ffc8', '#ffb800', '#ff3a5c', '#7c6fff']
const POLL_MS        = 2000

export default function PlayerScreen({ jobId, jobData }) {
  const videoRef    = useRef(null)
  const audioRef    = useRef(null)

  const [selectedFace,    setSelectedFace]    = useState(null)
  const [processedFaces,  setProcessedFaces]  = useState({})
  // { faceId: 'idle' | 'processing' | 'done' | 'error' }
  const [faceStates,      setFaceStates]      = useState({})
  const [isPlaying,       setIsPlaying]       = useState(false)
  const [currentTime,     setCurrentTime]     = useState(0)
  const [duration,        setDuration]        = useState(0)
  const [volume,          setVolume]          = useState(1)
  const [captionsOn,      setCaptionsOn]      = useState(true)
  const [notification,    setNotification]    = useState(null)

  const faceIds     = jobData?.face_ids ?? []
  const activeColor = selectedFace !== null
    ? (SPEAKER_COLORS[selectedFace] ?? '#00ffc8')
    : '#00ffc8'

  // ── Sync video + audio playback ───────────────────────────────────────────
  const syncPlay = useCallback(() => {
    const vid = videoRef.current
    const aud = audioRef.current
    if (!vid || !aud) return
    aud.currentTime = vid.currentTime
    vid.play().catch(() => {})
    aud.play().catch(() => {})
    setIsPlaying(true)
  }, [])

  const syncPause = useCallback(() => {
    videoRef.current?.pause()
    audioRef.current?.pause()
    setIsPlaying(false)
  }, [])

  const togglePlay = useCallback(() => {
    isPlaying ? syncPause() : syncPlay()
  }, [isPlaying, syncPlay, syncPause])

  // Keep audio in sync if user seeks video
  useEffect(() => {
    const vid = videoRef.current
    if (!vid) return
    const onSeeked = () => {
      if (audioRef.current) audioRef.current.currentTime = vid.currentTime
    }
    const onTimeUpdate = () => setCurrentTime(vid.currentTime)
    const onDuration   = () => setDuration(vid.duration)
    const onEnded      = () => setIsPlaying(false)
    vid.addEventListener('seeked',      onSeeked)
    vid.addEventListener('timeupdate',  onTimeUpdate)
    vid.addEventListener('loadedmetadata', onDuration)
    vid.addEventListener('ended',       onEnded)
    return () => {
      vid.removeEventListener('seeked',         onSeeked)
      vid.removeEventListener('timeupdate',     onTimeUpdate)
      vid.removeEventListener('loadedmetadata', onDuration)
      vid.removeEventListener('ended',          onEnded)
    }
  }, [])

  // ── Speaker selection ─────────────────────────────────────────────────────
  async function handleSelectSpeaker(faceId) {
    if (faceStates[faceId] === 'processing') return

    // If already processed, just switch
    if (faceStates[faceId] === 'done') {
      switchToSpeaker(faceId)
      return
    }

    // Otherwise trigger processing
    setFaceStates(prev => ({ ...prev, [faceId]: 'processing' }))
    showNotification(`Processing speaker ${faceId}...`, 'info')

    try {
      await processSpeaker(jobId, faceId)
      pollUntilDone(faceId)
    } catch (e) {
      setFaceStates(prev => ({ ...prev, [faceId]: 'error' }))
      showNotification(`Failed: ${e.message}`, 'error')
    }
  }

  function pollUntilDone(faceId) {
    const interval = setInterval(async () => {
      try {
        const data = await getStatus(jobId)
        if (data.status === 'done') {
          clearInterval(interval)
          setFaceStates(prev => ({ ...prev, [faceId]: 'done' }))
          showNotification(`Speaker ${faceId} ready!`, 'success')
          switchToSpeaker(faceId)
        } else if (data.status === 'error') {
          clearInterval(interval)
          setFaceStates(prev => ({ ...prev, [faceId]: 'error' }))
          showNotification(`Error: ${data.error}`, 'error')
        }
      } catch {
        clearInterval(interval)
      }
    }, POLL_MS)
  }

  function switchToSpeaker(faceId) {
    const wasPlaying = isPlaying
    syncPause()
    setSelectedFace(faceId)

    // Give React time to re-render the audio src
    setTimeout(() => {
      const aud = audioRef.current
      const vid = videoRef.current
      if (aud && vid) {
        aud.load()
        aud.currentTime = vid.currentTime
        // Update caption track
        updateCaptionTrack(faceId)
        if (wasPlaying) {
          aud.play().catch(() => {})
          vid.play().catch(() => {})
          setIsPlaying(true)
        }
      }
    }, 120)
  }

  function updateCaptionTrack(faceId) {
    const vid = videoRef.current
    if (!vid) return
    // Remove old tracks
    Array.from(vid.querySelectorAll('track')).forEach(t => t.remove())
    if (!captionsOn) return
    const track     = document.createElement('track')
    track.kind      = 'subtitles'
    track.label     = `Speaker ${faceId}`
    track.srclang   = 'en'
    track.src       = getCaptionsUrl(jobId, faceId)
    track.default   = true
    vid.appendChild(track)
    // Activate
    setTimeout(() => {
      if (vid.textTracks[0]) vid.textTracks[0].mode = 'showing'
    }, 200)
  }

  useEffect(() => {
    if (selectedFace !== null) updateCaptionTrack(selectedFace)
  }, [captionsOn, selectedFace])

  // ── Volume ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume
  }, [volume])

  // ── Seek bar ──────────────────────────────────────────────────────────────
  function handleSeek(e) {
    const t = parseFloat(e.target.value)
    if (videoRef.current) videoRef.current.currentTime = t
    if (audioRef.current) audioRef.current.currentTime = t
  }

  // ── Notifications ─────────────────────────────────────────────────────────
  function showNotification(msg, type = 'info') {
    setNotification({ msg, type })
    setTimeout(() => setNotification(null), 3000)
  }

  function fmt(s) {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60).toString().padStart(2, '0')
    return `${m}:${sec}`
  }

  return (
    <div style={styles.root}>
      {/* ── Header ── */}
      <div style={styles.header}>
        <span style={styles.logo}>
          VISIO<span style={{ color: activeColor, transition: 'color 0.5s' }}>VOX</span>
        </span>
        <div style={styles.headerRight}>
          <span style={styles.jobTag}>JOB · {jobId?.slice(0, 8).toUpperCase()}</span>
        </div>
      </div>

      <div style={styles.body}>
        {/* ── Left: Video + controls ── */}
        <div style={styles.leftCol}>
          {/* Video player */}
          <div style={{
            ...styles.videoWrap,
            boxShadow: `0 0 40px ${activeColor}22`,
            borderColor: `${activeColor}44`,
            transition: 'box-shadow 0.5s, border-color 0.5s',
          }}>
            <video
              ref={videoRef}
              src={getVideoUrl(jobId)}
              muted
              playsInline
              style={styles.video}
            />
            {/* Audio element (hidden) */}
            {selectedFace !== null && (
              <audio
                ref={audioRef}
                src={getAudioUrl(jobId, selectedFace)}
                preload="auto"
              />
            )}
            {/* No speaker selected overlay */}
            {selectedFace === null && (
              <div style={styles.videoOverlay}>
                <p style={styles.videoOverlayText}>
                  Select a speaker to begin
                </p>
              </div>
            )}
          </div>

          {/* Controls bar */}
          <div style={styles.controls}>
            {/* Play/pause */}
            <button
              onClick={togglePlay}
              disabled={selectedFace === null}
              style={{
                ...styles.playBtn,
                borderColor: activeColor,
                color:       activeColor,
                opacity:     selectedFace === null ? 0.4 : 1,
              }}
            >
              {isPlaying ? '⏸' : '▶'}
            </button>

            {/* Time + seek */}
            <div style={styles.seekWrap}>
              <span style={styles.timeLabel}>{fmt(currentTime)}</span>
              <input
                type="range" min={0} max={duration || 100} step={0.1}
                value={currentTime}
                onChange={handleSeek}
                style={{ ...styles.seekBar, accentColor: activeColor }}
              />
              <span style={styles.timeLabel}>{fmt(duration)}</span>
            </div>

            {/* Volume */}
            <div style={styles.volumeWrap}>
              <span style={{ fontSize: 14 }}>🔊</span>
              <input
                type="range" min={0} max={1} step={0.05}
                value={volume}
                onChange={e => setVolume(parseFloat(e.target.value))}
                style={{ ...styles.volBar, accentColor: activeColor, width: 70 }}
              />
            </div>

            {/* Captions toggle */}
            <button
              onClick={() => setCaptionsOn(c => !c)}
              style={{
                ...styles.captionBtn,
                borderColor: captionsOn ? activeColor : 'var(--border)',
                color:       captionsOn ? activeColor : 'var(--muted)',
              }}
            >
              CC
            </button>
          </div>
        </div>

        {/* ── Right col ── */}
        <div style={styles.rightCol}>
          {/* Speaker cards */}
          <div style={styles.sectionTitle}>
            <span style={{ color: activeColor }}>//</span> SELECT SPEAKER
          </div>
          <div style={styles.speakerRow}>
            {faceIds.map(faceId => (
              <SpeakerCard
                key={faceId}
                jobId={jobId}
                faceId={faceId}
                isSelected={selectedFace === faceId}
                isProcessing={faceStates[faceId] === 'processing'}
                isProcessed={faceStates[faceId]  === 'done'}
                onClick={() => handleSelectSpeaker(faceId)}
              />
            ))}
          </div>

          <div style={styles.divider} />

          {/* Waveform */}
          <div style={styles.sectionTitle}>
            <span style={{ color: activeColor }}>//</span> AUDIO SPECTRUM
          </div>
          <div style={styles.waveformBox}>
            <WaveformScene
              audioRef={audioRef}
              isPlaying={isPlaying}
              speakerColor={activeColor}
            />
          </div>

          {/* Status panel */}
          {selectedFace !== null && (
            <motion.div style={styles.statusPanel}
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
              <div style={{ ...styles.statusDot, background: isPlaying ? activeColor : 'var(--muted)' }} />
              <span style={{ color: isPlaying ? activeColor : 'var(--muted)', fontSize: 11,
                fontFamily: 'var(--font-display)', letterSpacing: '0.1em' }}>
                {isPlaying ? 'PLAYING' : 'PAUSED'} · SPEAKER {selectedFace}
              </span>
            </motion.div>
          )}
        </div>
      </div>

      {/* ── Notification toast ── */}
      <AnimatePresence>
        {notification && (
          <motion.div
            key="toast"
            style={{
              ...styles.toast,
              borderColor: notification.type === 'error'   ? 'var(--red)'
                         : notification.type === 'success' ? 'var(--cyan)'
                         : 'var(--border)',
            }}
            initial={{ opacity: 0, y: 20, x: '-50%' }}
            animate={{ opacity: 1, y: 0,  x: '-50%' }}
            exit={{    opacity: 0, y: 10,  x: '-50%' }}
          >
            {notification.msg}
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

const styles = {
  root: {
    width: '100%', height: '100vh',
    display: 'flex', flexDirection: 'column',
    background: 'var(--bg)', overflow: 'hidden',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '16px 28px',
    borderBottom: '1px solid var(--border)',
    background: 'rgba(10,10,18,0.9)', backdropFilter: 'blur(10px)',
    flexShrink: 0,
  },
  logo: {
    fontFamily: 'var(--font-display)', fontSize: 18,
    fontWeight: 900, letterSpacing: '0.12em',
  },
  headerRight: { display: 'flex', alignItems: 'center', gap: 16 },
  jobTag: {
    fontFamily: 'var(--font-display)', fontSize: 10,
    color: 'var(--muted)', letterSpacing: '0.15em',
  },
  body: {
    flex: 1, display: 'flex', gap: 24,
    padding: '20px 28px', overflow: 'hidden',
  },
  leftCol: {
    flex: '0 0 auto', width: 520,
    display: 'flex', flexDirection: 'column', gap: 12,
  },
  videoWrap: {
    position: 'relative', borderRadius: 12,
    border: '1px solid', overflow: 'hidden',
    background: '#000', aspectRatio: '16/9',
  },
  video: {
    width: '100%', height: '100%',
    objectFit: 'contain', display: 'block',
  },
  videoOverlay: {
    position: 'absolute', inset: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'rgba(5,5,8,0.7)',
  },
  videoOverlayText: {
    fontFamily: 'var(--font-display)', fontSize: 13,
    color: 'var(--muted)', letterSpacing: '0.1em',
  },
  controls: {
    display: 'flex', alignItems: 'center', gap: 12,
    padding: '10px 14px',
    background: 'var(--bg-card)',
    borderRadius: 8, border: '1px solid var(--border)',
  },
  playBtn: {
    width: 38, height: 38, borderRadius: '50%',
    background: 'transparent', border: '1px solid',
    fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center',
    transition: 'all 0.2s', flexShrink: 0,
  },
  seekWrap: {
    flex: 1, display: 'flex', alignItems: 'center', gap: 8,
  },
  seekBar: { flex: 1, height: 4, cursor: 'pointer' },
  timeLabel: {
    fontFamily: 'var(--font-display)', fontSize: 11,
    color: 'var(--muted)', minWidth: 36, letterSpacing: '0.05em',
  },
  volumeWrap: { display: 'flex', alignItems: 'center', gap: 6 },
  volBar: { cursor: 'pointer', height: 4 },
  captionBtn: {
    padding: '4px 10px', borderRadius: 4,
    background: 'transparent', border: '1px solid',
    fontFamily: 'var(--font-display)', fontSize: 10,
    letterSpacing: '0.1em', transition: 'all 0.2s',
    flexShrink: 0,
  },
  rightCol: {
    flex: 1, display: 'flex', flexDirection: 'column',
    gap: 12, overflow: 'hidden',
  },
  sectionTitle: {
    fontFamily: 'var(--font-display)', fontSize: 11,
    letterSpacing: '0.15em', color: 'var(--muted)',
    display: 'flex', alignItems: 'center', gap: 8,
  },
  speakerRow: { display: 'flex', gap: 12, flexWrap: 'wrap' },
  divider: {
    height: 1,
    background: 'linear-gradient(90deg, var(--border) 0%, transparent 100%)',
  },
  waveformBox: {
    flex: 1, borderRadius: 12,
    border: '1px solid var(--border)',
    background: 'var(--bg-card)', overflow: 'hidden',
    minHeight: 200,
  },
  statusPanel: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '8px 14px',
    background: 'var(--bg-card)',
    border: '1px solid var(--border)', borderRadius: 6,
  },
  statusDot: {
    width: 8, height: 8, borderRadius: '50%',
    transition: 'background 0.4s',
  },
  toast: {
    position: 'fixed', bottom: 28, left: '50%',
    padding: '10px 20px', borderRadius: 8,
    background: 'var(--bg-card)', border: '1px solid',
    fontFamily: 'var(--font-body)', fontSize: 12,
    color: 'var(--white)', zIndex: 1000,
    whiteSpace: 'nowrap',
  },
}
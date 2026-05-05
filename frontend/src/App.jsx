// src/App.jsx
import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import LandingScene    from './components/LandingScene.jsx'
import ProcessingScreen from './components/ProcessingScreen.jsx'
import PlayerScreen    from './components/PlayerScreen.jsx'

// Screen names: 'landing' | 'processing' | 'player'

export default function App() {
  const [screen,    setScreen]  = useState('landing')
  const [jobId,     setJobId]   = useState(null)
  const [jobData,   setJobData] = useState(null)  // final status payload when ready

  function handleUploadSuccess(id) {
    setJobId(id)
    setScreen('processing')
  }

  function handleProcessingComplete(statusPayload) {
    setJobData(statusPayload)
    setScreen('player')
  }

  return (
    <AnimatePresence mode="wait">
      {screen === 'landing' && (
        <motion.div key="landing"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, y: -40 }}
          transition={{ duration: 0.6 }}
          style={{ width: '100%', height: '100%' }}
        >
          <LandingScene onUploadSuccess={handleUploadSuccess} />
        </motion.div>
      )}

      {screen === 'processing' && (
        <motion.div key="processing"
          initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
          style={{ width: '100%', height: '100%' }}
        >
          <ProcessingScreen jobId={jobId} onComplete={handleProcessingComplete} />
        </motion.div>
      )}

      {screen === 'player' && (
        <motion.div key="player"
          initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
          style={{ width: '100%', height: '100%' }}
        >
          <PlayerScreen jobId={jobId} jobData={jobData} />
        </motion.div>
      )}
    </AnimatePresence>
  )
}
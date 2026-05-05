// src/api.js
// All communication with the FastAPI backend lives here.
// Components import from this file — never call fetch directly in components.

const BASE = ''  // empty because Vite proxies to http://localhost:8000

export async function uploadVideo(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`)
  return res.json()   // { job_id, message, status }
}

export async function getStatus(jobId) {
  const res = await fetch(`${BASE}/status/${jobId}`)
  if (!res.ok) throw new Error(`Status check failed: ${res.statusText}`)
  return res.json()
  // { job_id, status, progress_msg, num_faces, face_ids, processed_faces? }
}

export async function processSpeaker(jobId, faceId) {
  const res = await fetch(`${BASE}/process/${jobId}?face_id=${faceId}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(`Process failed: ${res.statusText}`)
  return res.json()
}

// These return URLs — used directly as <img src>, <audio src>, <video src>
export const getThumbnailUrl  = (jobId, faceId) => `${BASE}/thumbnails/${jobId}/${faceId}`
export const getAudioUrl      = (jobId, faceId) => `${BASE}/audio/${jobId}/${faceId}`
export const getCaptionsUrl   = (jobId, faceId) => `${BASE}/captions/${jobId}/${faceId}`
export const getVideoUrl      = (jobId)         => `${BASE}/video/${jobId}`
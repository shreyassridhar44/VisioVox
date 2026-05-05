import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API calls to FastAPI so we don't need CORS during dev
    proxy: {
      '/upload':    'http://localhost:8000',
      '/status':    'http://localhost:8000',
      '/thumbnails':'http://localhost:8000',
      '/process':   'http://localhost:8000',
      '/audio':     'http://localhost:8000',
      '/captions':  'http://localhost:8000',
      '/video':     'http://localhost:8000',
      '/health':    'http://localhost:8000',
    }
  }
})
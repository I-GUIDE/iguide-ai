import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Proxy /agent/* to the deployed I-GUIDE agent so the browser calls it same-origin
// (no CORS) and SSE streams through. Override the target with AGENT_TARGET.
const AGENT_TARGET =
  process.env.AGENT_TARGET ||
  'https://iguide-agent-dev.cis220065.projects.jetstream-cloud.org';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    proxy: {
      '/agent': {
        target: AGENT_TARGET,
        changeOrigin: true,
        secure: false,
        // SSE: don't buffer
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] = 'no-cache, no-transform';
          });
        },
      },
    },
  },
});

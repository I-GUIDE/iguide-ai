import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve, sep } from 'node:path';
import { replayHandler } from './scripts/replay-server';

// Proxy /agent/* to the deployed I-GUIDE agent so the browser calls it same-origin
// (no CORS) and SSE streams through. Override the target with AGENT_TARGET.
const AGENT_TARGET =
  process.env.AGENT_TARGET ||
  'https://iguide-agent-dev.cis220065.projects.jetstream-cloud.org';

// REPLAY_ROOT=<dir> npm run dev, then /?replay=<path under it>/<task>.events.jsonl runs that
// recorded turn through the page instead of the agent (src/replay.ts). Middleware added in
// configureServer runs before the proxy below, so only /agent/* calls of a replay are answered
// from the recording; the key must name an events file inside REPLAY_ROOT.
const REPLAY_ROOT = process.env.REPLAY_ROOT ? resolve(process.env.REPLAY_ROOT) : '';
const replayPlugin = {
  name: 'agent-replay',
  configureServer(server: any) {
    const handle = replayHandler((key) => {
      const p = resolve(REPLAY_ROOT, key);
      return p.startsWith(REPLAY_ROOT + sep) && p.endsWith('.events.jsonl') ? p : null;
    });
    server.middlewares.use((req: any, res: any, next: () => void) => {
      const isReplay = /[?&]replay=/.test(req.url || '') || /^\/agent\/(replay|files)\//.test(req.url || '');
      if (!(isReplay && handle(req, res))) next();
    });
  },
};

export default defineConfig({
  plugins: [react(), ...(REPLAY_ROOT ? [replayPlugin] : [])],
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

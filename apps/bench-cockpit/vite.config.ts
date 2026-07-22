import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/echarts') || id.includes('echarts-for-react')) {
            return 'echarts';
          }
          if (id.includes('node_modules/react-dom') || id.includes('node_modules/react/')) {
            return 'react-vendor';
          }
        },
      },
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    // Keep Vite HMR on its own path so it never fights the bench WS.
    hmr: {
      protocol: 'ws',
      host: '127.0.0.1',
      port: 5173,
      path: '/@vite',
    },
    proxy: {
      // HTTP + WS under /api → Go on 8090 (use 127.0.0.1, not localhost/IPv6).
      '/api': {
        target: 'http://127.0.0.1:8090',
        changeOrigin: true,
        ws: true,
      },
      // Legacy path still proxied if something hits /ws directly.
      '/ws': {
        target: 'ws://127.0.0.1:8090',
        ws: true,
        changeOrigin: true,
      },
    },
  },
});

// frontend/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Route-level splitting would not help here: recharts is imported by
        // all four pages, so every route chunk would pull it in. Splitting by
        // dependency does — this takes the app's own code from 620 kB to 81 kB,
        // and the charting library (which react ends up inside, since recharts
        // imports it) lands in a chunk the browser keeps across app deploys
        // instead of being re-downloaded whenever a panel is edited.
        manualChunks: { charts: ['recharts'] },
      },
    },
    // The charts chunk is 540 kB minified / 153 kB over the wire, and it is a
    // vendor bundle that changes on upgrades rather than on edits. Recharts does
    // not tree-shake meaningfully, so the warning has nothing left to tell us —
    // raised rather than left firing on every build, which is how a real size
    // regression would go unnoticed.
    chunkSizeWarningLimit: 600,
  },
})

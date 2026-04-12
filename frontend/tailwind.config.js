/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          0: '#080c14',
          1: '#0c1120',
          2: '#111827',
          3: '#1a2235',
          4: '#1e293b',
        },
        accent: {
          cyan:    '#38bdf8',
          emerald: '#34d399',
          rose:    '#fb7185',
          amber:   '#fbbf24',
          violet:  '#a78bfa',
        },
        tx: {
          primary:   '#e2e8f0',
          secondary: '#94a3b8',
          muted:     '#64748b',
        },
      },
      fontFamily: {
        sans: ['Outfit', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
}

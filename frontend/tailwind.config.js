/** Design tokens for Quarter.
 *
 *  The palette is derived from the subject rather than picked: a binary has
 *  exactly two outcomes, so the two poles — `above` and `below` — are the only
 *  colours that ever carry direction, `mid` marks the 50% pole between them, and
 *  `accent` is deliberately neither, so a structural element (a link, a focus
 *  ring, a selected row) can never be misread as a directional signal. Gate
 *  semantics are a third, separate set, so "gate failed" never looks like
 *  "price down".
 *
 *  Deliberately not red/green: hostile to the ~8% of men with a red-green
 *  deficiency, and the crypto-dashboard cliché.
 *
 *  Everything resolves through CSS variables defined in index.css, so one token
 *  set serves both themes and no component redefines a colour behind a media
 *  query.
 */
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        paper: 'var(--paper)',
        surface: 'var(--surface)',
        sunken: 'var(--sunken)',
        rule: 'var(--rule)',
        'rule-firm': 'var(--rule-firm)',
        ink: 'var(--ink)',
        'ink-2': 'var(--ink-2)',
        'ink-3': 'var(--ink-3)',
        above: 'var(--above)',
        'above-wash': 'var(--above-wash)',
        below: 'var(--below)',
        'below-wash': 'var(--below-wash)',
        mid: 'var(--mid)',
        accent: 'var(--accent)',
        'accent-wash': 'var(--accent-wash)',
        pass: 'var(--pass)',
        fail: 'var(--fail)',
        warn: 'var(--warn)',
      },
      fontFamily: {
        sans: ['"Familjen Grotesk"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"Kode Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      fontSize: {
        // A real scale, on a 1.2 ratio from 13px. `text-sm`/`text-lg` as the
        // whole vocabulary is what makes a UI read as unconsidered.
        micro: ['0.6875rem', { lineHeight: '1rem', letterSpacing: '0.08em' }],
        tiny: ['0.75rem', { lineHeight: '1.125rem' }],
        base: ['0.8125rem', { lineHeight: '1.375rem' }],
        mid: ['0.9375rem', { lineHeight: '1.5rem' }],
        lg: ['1.125rem', { lineHeight: '1.625rem', letterSpacing: '-0.01em' }],
        xl: ['1.5rem', { lineHeight: '1.875rem', letterSpacing: '-0.02em' }],
        '2xl': ['2rem', { lineHeight: '2.25rem', letterSpacing: '-0.025em' }],
        '3xl': ['2.75rem', { lineHeight: '2.875rem', letterSpacing: '-0.03em' }],
      },
      borderRadius: {
        // An instrument panel, not a consumer app. 2px reads as machined; 8px
        // everywhere reads as a template.
        DEFAULT: '2px',
        sm: '1px',
        none: '0',
      },
      spacing: { rail: '13.5rem' },
      maxWidth: { shell: '82rem' },
    },
  },
  plugins: [],
};

/**
 * Refuse a colour utility whose token the Tailwind config does not declare.
 *
 * Tailwind silently drops a class it does not recognise, so `text-up` on a
 * config that declares `above` renders with no colour at all and nothing
 * says so -- not the build, not the linter, not the type checker. Shipped
 * exactly that on 2026-10-22: a new panel used `border-line`, `bg-line`,
 * `text-up` and `text-down` against a config declaring `rule`, `above` and
 * `below`, so the table arrived with no borders and no profit/loss colour.
 *
 * This only checks the COLOUR position of a handful of prefixes. It is not a
 * Tailwind reimplementation, and it stays quiet about spacing, sizing and
 * layout, where a wrong value is visible rather than invisible.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, extname } from 'node:path';

const SRC = new URL('../src', import.meta.url).pathname;
const CONFIG = new URL('../tailwind.config.js', import.meta.url).pathname;

/** Prefixes whose next segment is a colour token. */
// `accent` is deliberately absent: it is both a declared colour and a
// utility prefix, so `accent-wash` parses as accent- + wash and false-alarms.
const PREFIXES = ['text', 'bg', 'border', 'stroke', 'fill', 'ring', 'divide',
                  'outline', 'decoration', 'caret'];

/** Built-ins Tailwind ships that are not in our config's `colors`. */
const BUILTIN = new Set([
  'transparent', 'current', 'inherit', 'white', 'black', 'auto', 'none',
  // text- also carries size, alignment, weight and transform utilities
  'left', 'right', 'center', 'justify', 'start', 'end', 'wrap', 'nowrap',
  'balance', 'pretty', 'ellipsis', 'clip', 'xs', 'sm', 'base', 'lg', 'xl',
  '2xl', '3xl', '4xl', '5xl', '6xl', '7xl', '8xl', '9xl',
  // border-/divide- also carry widths and styles
  't', 'r', 'b', 'l', 'x', 'y', 's', 'e', 'solid', 'dashed', 'dotted',
  'double', 'hidden', 'collapse', 'separate', 'spacing',
  // ring-/shadow-/outline- offsets and sizes
  'offset', 'inset', 'md', 'full', 'reverse',
]);

function declaredTokens() {
  const text = readFileSync(CONFIG, 'utf8');
  const tokens = new Set();
  // `name: 'var(--x)'` and `'quoted-name': 'var(--x)'` inside the theme.
  for (const m of text.matchAll(/^\s*'?([a-z0-9-]+)'?:\s*'var\(--/gm)) {
    tokens.add(m[1]);
  }
  // fontSize / spacing style keys: `tiny: [...]`
  for (const m of text.matchAll(/^\s*'?([a-z0-9-]+)'?:\s*\[/gm)) {
    tokens.add(m[1]);
  }
  return tokens;
}

function sources(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) out.push(...sources(path));
    else if (['.ts', '.tsx'].includes(extname(path))) out.push(path);
  }
  return out;
}

const declared = declaredTokens();
const pattern = new RegExp(
  `\\b(?:[a-z-]+:)*(${PREFIXES.join('|')})-([a-z][a-z0-9-]*)`, 'g');

const bad = [];
for (const file of sources(SRC)) {
  const text = readFileSync(file, 'utf8');
  text.split('\n').forEach((line, i) => {
    for (const m of line.matchAll(pattern)) {
      const [, prefix, token] = m;
      // A compound token like `above-wash` is declared whole; a bare first
      // segment of one is not what was written, so test the full match.
      if (declared.has(token) || BUILTIN.has(token)) continue;
      const head = token.split('-')[0];
      // `border-b-0`, `divide-y-2`: a built-in first segment means this is a
      // width or a side, not a colour.
      if (declared.has(head) || BUILTIN.has(head)) continue;
      bad.push(`${file.replace(SRC, 'src')}:${i + 1}  ${prefix}-${token}`);
    }
  });
}

if (bad.length) {
  console.error(
    '\nUndeclared colour tokens. Tailwind drops these silently, so they ' +
    'render with no colour:\n');
  for (const line of bad) console.error('  ' + line);
  console.error(
    `\nDeclared: ${[...declared].sort().join(', ')}\n`);
  process.exit(1);
}
console.log(`tokens ok — ${declared.size} declared, no undeclared uses`);

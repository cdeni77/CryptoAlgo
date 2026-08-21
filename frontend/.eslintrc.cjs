/**
 * ESLint config. `npm run lint` was in package.json with every plugin installed
 * and no config file, so it had never once run — it exited with "couldn't find a
 * configuration file", which a CI step reading only the exit code would report
 * as a failure and a human running it locally would shrug at.
 *
 * ESLint 8 flat config is not used here because the installed version's plugin
 * ecosystem (`@typescript-eslint` 7, `eslint-plugin-react-hooks` 4) is still on
 * eslintrc, and mixing the two silently drops rules.
 */
module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: { ecmaVersion: 'latest', sourceType: 'module' },
  plugins: ['@typescript-eslint', 'react-hooks', 'react-refresh'],
  ignorePatterns: ['dist', 'node_modules', '*.cjs', 'vite.config.ts'],
  settings: { react: { version: '18.3' } },
  rules: {
    // The dependency-array rules are the ones that actually catch bugs in this
    // codebase: a stale closure in a polling effect shows up as a screen that
    // silently stops updating.
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn',
    'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],

    // An unused variable after a refactor is usually a leftover, but an
    // underscore prefix is a deliberate "I know".
    '@typescript-eslint/no-unused-vars': [
      'error',
      { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
    ],

    // `any` defeats the point of having types on API responses, where a wrong
    // shape is exactly the bug worth catching.
    '@typescript-eslint/no-explicit-any': 'error',

    // Empty catch blocks are how this frontend came to hide every backend
    // failure behind stale data. A deliberate ignore needs a comment in it.
    'no-empty': ['error', { allowEmptyCatch: false }],

    eqeqeq: ['error', 'always', { null: 'ignore' }],
    'no-console': ['warn', { allow: ['warn', 'error'] }],
  },
  overrides: [
    {
      // JSX brings its own globals and needs the TS resolver for .tsx.
      files: ['**/*.tsx'],
      parserOptions: { ecmaFeatures: { jsx: true } },
      globals: { JSX: 'readonly' },
    },
  ],
};

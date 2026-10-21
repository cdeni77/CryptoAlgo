/** Move between pages from inside a page component.
 *
 * **Not a prop, because `PAGES` is a `Record<RoutePath, ComponentType>`.** That
 * map is what makes a route without a component a `tsc` error rather than a
 * blank screen, and widening it to carry a `navigate` prop would give that up
 * for the sake of two links. `App` already listens for `popstate` to handle the
 * back button, so pushing state and announcing it uses the path that is
 * guaranteed to work — the same one the browser itself uses.
 */
export function go(path: string): void {
  // Compared against pathname AND search, because a link may differ from where
  // you already are only by its query — `/decisions?traded=1` from
  // `/decisions` is a real move, and comparing pathnames alone would swallow
  // it and leave the filter off.
  if (window.location.pathname + window.location.search === path) return;
  window.history.pushState({}, '', path);
  window.dispatchEvent(new PopStateEvent('popstate'));
}

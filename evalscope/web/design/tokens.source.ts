/**
 * Design token single source of truth (SSOT).
 *
 * This module is the authoritative definition of every design token that exists
 * in BOTH the runtime CSS (`src/index.css`) and the human-facing design document
 * (`DESIGN.md` frontmatter). Each token records:
 *
 *   - its canonical value (per theme where relevant), and
 *   - where that value is expected to appear in each surface (the CSS custom
 *     property name and the DESIGN.md frontmatter key).
 *
 * The contract (Req 17.2) is that the Markdown-displayed token value and the CSS
 * token definition are both generated from — or validated against — this single
 * source, so the same token reads byte-for-byte identically in both places.
 *
 * This file intentionally only holds the SSOT DATA and TYPES plus small
 * serialization helpers. The actual drift comparison (parsing CSS/Markdown and
 * diffing against this source) lives in `scripts/drift/tokenDrift.ts` (task 17.3),
 * which consumes the `TokenSource` type exported here. Canonical values mirror the
 * runtime CSS in `src/index.css`, because that is what users actually see; any
 * divergence in `DESIGN.md` is surfaced by the drift check rather than silently
 * reconciled here.
 */

/** Theme a token value belongs to. */
export type ThemeName = 'dark' | 'light'

/** All themes a themed token is defined for, in a stable order. */
export const THEMES: readonly ThemeName[] = ['dark', 'light']

/** Logical grouping of a token, used for reporting and organization. */
export type TokenCategory = 'radius' | 'transition' | 'color' | 'shadow'

/**
 * Canonical value of a token.
 *
 * A `shared` value is theme-agnostic (the same string applies to every theme,
 * e.g. a border radius). A `themed` value carries a distinct string per theme
 * (e.g. an accent color that shifts between dark and light).
 */
export type TokenValue =
  | { readonly kind: 'shared'; readonly value: string }
  | { readonly kind: 'themed'; readonly dark: string; readonly light: string }

/** Location of a token's value inside the CSS source (`src/index.css`). */
export interface CssRef {
  /**
   * CSS custom property name, e.g. `--accent`. For themed tokens the same
   * property name is redeclared in each theme block with a different value.
   */
  readonly var: string
}

/** Location of a token's value inside the DESIGN.md frontmatter. */
export interface MarkdownRef {
  /**
   * Dotted key path for the shared or dark-theme value, e.g. `accent` or
   * `rounded.sm`.
   */
  readonly key: string
  /**
   * Dotted key path for the light-theme value when the token is themed and the
   * light value is displayed separately, e.g. `accent-light`. Omitted for shared
   * tokens or when Markdown does not display a distinct light value.
   */
  readonly lightKey?: string
}

/** A single design token and where its canonical value must appear. */
export interface TokenDefinition {
  /** Stable, canonical token identifier used by the SSOT and drift reports. */
  readonly name: string
  /** Logical category of the token. */
  readonly category: TokenCategory
  /** Optional human-readable note about the token's intent. */
  readonly description?: string
  /** Canonical value(s) of the token. */
  readonly value: TokenValue
  /** Where the value is expected in the CSS source. */
  readonly css: CssRef
  /** Where the value is expected in the DESIGN.md frontmatter. */
  readonly markdown: MarkdownRef
}

/** The complete, ordered set of design tokens forming the single source of truth. */
export type TokenSource = readonly TokenDefinition[]

/** Convenience constructor for a theme-agnostic (shared) token value. */
function shared(value: string): TokenValue {
  return { kind: 'shared', value }
}

/** Convenience constructor for a per-theme (themed) token value. */
function themed(dark: string, light: string): TokenValue {
  return { kind: 'themed', dark, light }
}

/**
 * The design token single source of truth.
 *
 * Only tokens present in both `src/index.css` and `DESIGN.md` are included, since
 * the byte-for-byte contract is defined across those two surfaces. Markdown-only
 * scales (spacing, typography, breakpoints, container) have no CSS custom-property
 * counterpart and are therefore intentionally out of scope for cross-surface drift.
 */
export const TOKEN_SOURCE: TokenSource = [
  // ── Radius (theme-agnostic; declared once in :root) ────────────────────────
  {
    name: 'radius-xs',
    category: 'radius',
    value: shared('4px'),
    css: { var: '--radius-xs' },
    markdown: { key: 'rounded.xs' },
  },
  {
    name: 'radius-sm',
    category: 'radius',
    value: shared('8px'),
    css: { var: '--radius-sm' },
    markdown: { key: 'rounded.sm' },
  },
  {
    name: 'radius-md',
    category: 'radius',
    description: 'Default card radius; CSS exposes it as the base --radius token.',
    value: shared('12px'),
    css: { var: '--radius' },
    markdown: { key: 'rounded.md' },
  },
  {
    name: 'radius-lg',
    category: 'radius',
    value: shared('16px'),
    css: { var: '--radius-lg' },
    markdown: { key: 'rounded.lg' },
  },
  {
    name: 'radius-xl',
    category: 'radius',
    value: shared('20px'),
    css: { var: '--radius-xl' },
    markdown: { key: 'rounded.xl' },
  },

  // ── Transition (theme-agnostic) ────────────────────────────────────────────
  {
    name: 'transition-fast',
    category: 'transition',
    value: shared('150ms cubic-bezier(0.4, 0, 0.2, 1)'),
    css: { var: '--transition-fast' },
    markdown: { key: 'transition.fast' },
  },
  {
    name: 'transition-base',
    category: 'transition',
    value: shared('250ms cubic-bezier(0.4, 0, 0.2, 1)'),
    css: { var: '--transition-base' },
    markdown: { key: 'transition.base' },
  },
  {
    name: 'transition-slow',
    category: 'transition',
    value: shared('400ms cubic-bezier(0.4, 0, 0.2, 1)'),
    css: { var: '--transition-slow' },
    markdown: { key: 'transition.slow' },
  },

  // ── Color — brand & accent (themed) ────────────────────────────────────────
  {
    name: 'accent',
    category: 'color',
    description: 'Single brand violet; the conversion target across both themes.',
    value: themed('#816DF8', '#6c57e8'),
    css: { var: '--accent' },
    markdown: { key: 'accent', lightKey: 'accent-light' },
  },
  {
    name: 'accent-dim',
    category: 'color',
    value: themed('rgba(129, 109, 248, 0.12)', 'rgba(108, 87, 232, 0.14)'),
    css: { var: '--accent-dim' },
    markdown: { key: 'accent-dim', lightKey: 'accent-dim-light' },
  },

  // ── Color — surface ladder (themed) ────────────────────────────────────────
  {
    name: 'bg',
    category: 'color',
    description: 'Page body surface.',
    value: themed('#0c0c1a', '#faf9f5'),
    css: { var: '--bg' },
    markdown: { key: 'bg', lightKey: 'bg-light' },
  },
  {
    name: 'bg-deep',
    category: 'color',
    description: 'One step below the page; input wells and pill-tab containers.',
    value: themed('#09091a', '#f0ebe1'),
    css: { var: '--bg-deep' },
    markdown: { key: 'bg-deep', lightKey: 'bg-deep-light' },
  },
  {
    name: 'bg-card',
    category: 'color',
    description: 'Default card / dialog / table surface.',
    value: themed('#12122b', '#ffffff'),
    css: { var: '--bg-card' },
    markdown: { key: 'bg-card', lightKey: 'bg-card-light' },
  },
  {
    name: 'bg-card2',
    category: 'color',
    description: 'Elevated / hover surface.',
    value: themed('#16163a', '#f5f0e7'),
    css: { var: '--bg-card2' },
    markdown: { key: 'bg-card2', lightKey: 'bg-card2-light' },
  },
  {
    name: 'surface-glass',
    category: 'color',
    description: 'Glassmorphic sticky top-nav backdrop.',
    value: themed('rgba(18, 18, 43, 0.7)', 'rgba(250, 249, 245, 0.80)'),
    css: { var: '--surface-glass' },
    markdown: { key: 'surface-glass', lightKey: 'surface-glass-light' },
  },

  // ── Color — text ladder (themed) ───────────────────────────────────────────
  {
    name: 'text',
    category: 'color',
    description: 'Primary body text.',
    value: themed('#e2e8f0', '#141413'),
    css: { var: '--text' },
    markdown: { key: 'text', lightKey: 'text-light' },
  },
  {
    name: 'text-muted',
    category: 'color',
    value: themed('#8896aa', '#6c6a64'),
    css: { var: '--text-muted' },
    markdown: { key: 'text-muted', lightKey: 'text-muted-light' },
  },
  {
    name: 'text-dim',
    category: 'color',
    value: themed('#7a8195', '#8e8b82'),
    css: { var: '--text-dim' },
    markdown: { key: 'text-dim', lightKey: 'text-dim-light' },
  },

  // ── Color — hairline borders (themed) ──────────────────────────────────────
  {
    name: 'border',
    category: 'color',
    description: 'Standard hairline border.',
    value: themed('rgba(129, 109, 248, 0.10)', '#e6dfd8'),
    css: { var: '--border' },
    markdown: { key: 'border', lightKey: 'border-light' },
  },
  {
    name: 'border-md',
    category: 'color',
    description: 'Emphasized hairline border.',
    value: themed('rgba(129, 109, 248, 0.18)', '#d6cdbe'),
    css: { var: '--border-md' },
    markdown: { key: 'border-md', lightKey: 'border-md-light' },
  },
  {
    name: 'border-strong',
    category: 'color',
    description: 'Hover / focus boundary border.',
    value: themed('rgba(129, 109, 248, 0.28)', '#c1b6a3'),
    css: { var: '--border-strong' },
    markdown: { key: 'border-strong', lightKey: 'border-strong-light' },
  },

  // ── Shadows (themed) ───────────────────────────────────────────────────────
  {
    name: 'shadow-sm',
    category: 'shadow',
    value: themed(
      '0 2px 8px rgba(0, 0, 0, 0.4)',
      '0 1px 2px rgba(20, 20, 19, 0.04), 0 4px 12px rgba(20, 20, 19, 0.06)',
    ),
    css: { var: '--shadow-sm' },
    markdown: { key: 'shadows.sm', lightKey: 'shadows.sm-light' },
  },
  {
    name: 'shadow-md',
    category: 'shadow',
    description: 'Default elevation; CSS exposes it as the base --shadow token.',
    value: themed(
      '0 4px 20px rgba(0, 0, 0, 0.55)',
      '0 4px 16px rgba(20, 20, 19, 0.07), 0 12px 32px rgba(20, 20, 19, 0.05)',
    ),
    css: { var: '--shadow' },
    markdown: { key: 'shadows.md', lightKey: 'shadows.md-light' },
  },
  {
    name: 'shadow-lg',
    category: 'shadow',
    value: themed(
      '0 8px 40px rgba(0, 0, 0, 0.6)',
      '0 12px 24px rgba(20, 20, 19, 0.09), 0 24px 48px rgba(20, 20, 19, 0.07)',
    ),
    css: { var: '--shadow-lg' },
    markdown: { key: 'shadows.lg', lightKey: 'shadows.lg-light' },
  },
  {
    name: 'shadow-glow',
    category: 'shadow',
    value: themed('0 0 20px rgba(129, 109, 248, 0.25)', '0 0 20px rgba(108, 87, 232, 0.22)'),
    css: { var: '--shadow-glow' },
    markdown: { key: 'shadows.glow', lightKey: 'shadows.glow-light' },
  },
  {
    name: 'shadow-glow-soft',
    category: 'shadow',
    value: themed('0 0 12px rgba(129, 109, 248, 0.20)', '0 0 12px rgba(108, 87, 232, 0.18)'),
    css: { var: '--shadow-glow-soft' },
    markdown: { key: 'shadows.glow-soft', lightKey: 'shadows.glow-soft-light' },
  },
]

/**
 * Resolve the canonical value of a token for a given theme.
 *
 * Shared tokens return the same value for every theme; themed tokens return the
 * theme-specific value.
 *
 * @param token Token definition to resolve.
 * @param theme Theme to resolve the value for.
 * @returns The canonical value string.
 */
export function getTokenValue(token: TokenDefinition, theme: ThemeName): string {
  if (token.value.kind === 'shared') {
    return token.value.value
  }
  return theme === 'dark' ? token.value.dark : token.value.light
}

/**
 * Resolve the DESIGN.md frontmatter key that should carry a token's value for a
 * given theme.
 *
 * Shared tokens use their single key for every theme. Themed tokens use the base
 * key for the dark value and `lightKey` for the light value. Returns `null` when
 * the token has no Markdown representation for the requested theme (e.g. a themed
 * token whose light value is not separately displayed).
 *
 * @param token Token definition to inspect.
 * @param theme Theme to resolve the Markdown key for.
 * @returns The dotted Markdown key path, or `null` when not represented.
 */
export function getMarkdownKey(token: TokenDefinition, theme: ThemeName): string | null {
  if (token.value.kind === 'shared') {
    return token.markdown.key
  }
  if (theme === 'dark') {
    return token.markdown.key
  }
  return token.markdown.lightKey ?? null
}

/** A single flattened, comparable (token, theme) expectation. */
export interface FlatTokenExpectation {
  /** Canonical token name. */
  readonly name: string
  /** Theme this expectation applies to. */
  readonly theme: ThemeName
  /** CSS custom property name expected to carry the value. */
  readonly cssVar: string
  /** DESIGN.md frontmatter key expected to carry the value. */
  readonly markdownKey: string
  /** Canonical value both surfaces must match byte-for-byte. */
  readonly value: string
}

/**
 * Flatten the token source into one comparable expectation per (token, theme).
 *
 * This serialization helper is the primary input for the drift check (task 17.3):
 * each returned entry pairs a canonical value with the exact CSS custom property
 * and Markdown key expected to carry it, so the check can diff both surfaces
 * byte-for-byte without re-deriving any naming conventions. Themed tokens whose
 * light value has no Markdown key are skipped for that theme.
 *
 * @param source Token source to flatten. Defaults to {@link TOKEN_SOURCE}.
 * @returns One expectation per (token, theme) that is present in both surfaces.
 */
export function flattenTokens(source: TokenSource = TOKEN_SOURCE): FlatTokenExpectation[] {
  const expectations: FlatTokenExpectation[] = []
  for (const token of source) {
    for (const theme of THEMES) {
      const markdownKey = getMarkdownKey(token, theme)
      if (markdownKey === null) {
        // No Markdown representation for this theme; nothing to compare here.
        continue
      }
      // Shared tokens only need to be emitted once (they are theme-agnostic).
      if (token.value.kind === 'shared' && theme !== 'dark') {
        continue
      }
      expectations.push({
        name: token.name,
        theme,
        cssVar: token.css.var,
        markdownKey,
        value: getTokenValue(token, theme),
      })
    }
  }
  return expectations
}

/**
 * Serialize a token as a CSS custom-property declaration for a given theme.
 *
 * Used to generate the CSS token definition from the SSOT (Req 17.2), e.g.
 * `--accent: #816DF8;`.
 *
 * @param token Token definition to serialize.
 * @param theme Theme whose value should be emitted.
 * @returns A single CSS declaration string terminated with a semicolon.
 */
export function serializeCssDeclaration(token: TokenDefinition, theme: ThemeName): string {
  return `${token.css.var}: ${getTokenValue(token, theme)};`
}

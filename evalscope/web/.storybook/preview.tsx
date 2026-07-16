import type { Preview, Decorator } from '@storybook/react-vite'
import { ThemeProvider } from '../src/contexts/ThemeContext'
import { LocaleProvider } from '../src/contexts/LocaleContext'
import '../src/index.css'

type StoryTheme = 'light' | 'dark'

const THEME_STORAGE_KEY = 'evalscope-theme'

/**
 * Mounts the app's Theme and Locale providers around every story and syncs
 * the toolbar theme global with the provider state.
 *
 * ThemeProvider drives the visual theme through the `data-theme` attribute on
 * the document element and initialises itself from localStorage. The decorator
 * writes the selected theme to storage and the attribute before rendering, then
 * keys ThemeProvider on the theme so it re-initialises when the toolbar toggle
 * changes — giving a working light/dark global switch.
 */
const withProviders: Decorator = (Story, context) => {
  const theme = (context.globals.theme as StoryTheme) ?? 'dark'

  window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  document.documentElement.setAttribute('data-theme', theme)

  return (
    <ThemeProvider key={theme}>
      <LocaleProvider>
        <Story />
      </LocaleProvider>
    </ThemeProvider>
  )
}

const preview: Preview = {
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
  },
  globalTypes: {
    theme: {
      description: 'Global light/dark theme for components',
      toolbar: {
        title: 'Theme',
        icon: 'circlehollow',
        items: [
          { value: 'light', title: 'Light', icon: 'sun' },
          { value: 'dark', title: 'Dark', icon: 'moon' },
        ],
        dynamicTitle: true,
      },
    },
  },
  initialGlobals: {
    theme: 'dark',
  },
  decorators: [withProviders],
}

export default preview

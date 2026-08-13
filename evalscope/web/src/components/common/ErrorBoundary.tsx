import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertCircle } from 'lucide-react'
import Button from '@/components/ui/Button'

/** Text shown by the fallback, so a boundary inside the providers can translate it. */
export interface ErrorBoundaryLabels {
  title: string
  /** Shown when the error carries no message of its own. */
  body: string
  action: string
}

/**
 * English defaults for the outermost boundary.
 *
 * The root boundary sits outside `LocaleProvider` — it has to, in order to catch a
 * failure in the providers themselves — so it cannot translate. Boundaries mounted
 * inside the providers pass their own translated labels.
 */
const DEFAULT_LABELS: ErrorBoundaryLabels = {
  title: 'Something went wrong',
  body: 'An unexpected error occurred.',
  action: 'Reload',
}

interface Props {
  children: ReactNode
  fallback?: ReactNode
  labels?: ErrorBoundaryLabels
  /**
   * Recovery action. Defaults to a full page reload, which is all the root
   * boundary can offer. A route-level boundary passes something cheaper, because
   * the rest of the app is still mounted and working.
   */
  onRecover?: () => void
}

interface State {
  hasError: boolean
  error: Error | null
}

/**
 * Catches a render failure in its subtree and offers a way out.
 *
 * Two of these are mounted. The root one in `App` is the last resort for a
 * failure in the providers themselves. The one in `MainLayout` is keyed by route,
 * so a page that throws is contained: the navigation, theme and locale stay
 * usable, navigating elsewhere clears the error by remount, and the scan state is
 * not discarded.
 */
export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack)
  }

  handleRecover = () => {
    this.setState({ hasError: false, error: null })
    if (this.props.onRecover) {
      this.props.onRecover()
      return
    }
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback
      const labels = this.props.labels ?? DEFAULT_LABELS

      return (
        <div className="flex min-h-[60vh] items-center justify-center">
          <div className="w-full max-w-[420px] rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-10 text-center shadow-[var(--shadow)]">
            <div className="mb-4 inline-flex h-14 w-14 items-center justify-center rounded-[var(--radius)] border border-[var(--danger-border)] bg-[var(--danger-bg)]">
              <AlertCircle size={24} className="text-[var(--danger)]" />
            </div>
            <h2 className="type-title-md mt-0 mb-2 text-[var(--text)]">{labels.title}</h2>
            <p className="type-body-sm mt-0 mb-5 leading-normal text-[var(--text-muted)]">
              {this.state.error?.message || labels.body}
            </p>
            <Button onClick={this.handleRecover}>{labels.action}</Button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

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

  handleReload = () => {
    this.setState({ hasError: false, error: null })
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback

      return (
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-[var(--radius)] p-10 max-w-[420px] w-full text-center shadow-[var(--shadow)]">
            {/* Error icon */}
            <div className="w-14 h-14 rounded-2xl bg-[var(--danger-bg)] border border-[var(--danger-border)] inline-flex items-center justify-center mb-4">
              <svg
                width={24}
                height={24}
                viewBox="0 0 24 24"
                fill="none"
                stroke="var(--danger)"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </div>
            <h2 className="text-[var(--text)] text-lg font-semibold mb-2 mt-0">
              Something went wrong
            </h2>
            <p className="text-[var(--text-muted)] text-sm mb-5 mt-0 leading-normal">
              {this.state.error?.message || 'An unexpected error occurred.'}
            </p>
            <button
              onClick={this.handleReload}
              className="bg-[var(--accent)] text-[var(--bg)] border-0 rounded-[var(--radius-sm)] px-6 py-2 text-sm font-medium cursor-pointer transition-opacity duration-150 hover:opacity-85"
            >
              Reload
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

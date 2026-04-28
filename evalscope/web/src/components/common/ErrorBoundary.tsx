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
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '60vh',
          }}
        >
          <div
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              padding: '2.5rem',
              maxWidth: 420,
              width: '100%',
              textAlign: 'center',
              boxShadow: 'var(--shadow)',
            }}
          >
            {/* Error icon */}
            <div
              style={{
                width: 56,
                height: 56,
                borderRadius: 16,
                background: 'var(--danger-bg)',
                border: '1px solid var(--danger-border)',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: 16,
              }}
            >
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
            <h2
              style={{
                color: 'var(--text)',
                fontSize: '1.125rem',
                fontWeight: 600,
                margin: '0 0 8px',
              }}
            >
              Something went wrong
            </h2>
            <p
              style={{
                color: 'var(--text-muted)',
                fontSize: '0.875rem',
                margin: '0 0 20px',
                lineHeight: 1.5,
              }}
            >
              {this.state.error?.message || 'An unexpected error occurred.'}
            </p>
            <button
              onClick={this.handleReload}
              style={{
                background: 'var(--accent)',
                color: 'var(--bg)',
                border: 'none',
                borderRadius: 'var(--radius-sm)',
                padding: '0.5rem 1.5rem',
                fontSize: '0.875rem',
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'opacity 0.15s',
              }}
              onMouseOver={(e) => (e.currentTarget.style.opacity = '0.85')}
              onMouseOut={(e) => (e.currentTarget.style.opacity = '1')}
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

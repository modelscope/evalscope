import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

type ButtonVariant = 'primary' | 'ghost' | 'outline'
type ButtonSize = 'sm' | 'md' | 'lg'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode
  variant?: ButtonVariant
  size?: ButtonSize
}

const variantStyles: Record<ButtonVariant, string> = {
  primary:
    'bg-[var(--accent)] text-white hover:bg-[var(--accent-dark)] hover:shadow-[0_0_20px_rgba(129,109,248,0.25)] active:scale-[0.98]',
  ghost:
    'bg-transparent text-[var(--text)] hover:bg-[var(--bg-card)] active:bg-[var(--bg-card2)]',
  outline:
    'bg-transparent border border-[var(--border-md)] text-[var(--text)] hover:border-[var(--accent)] hover:text-[var(--accent)]',
}

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-xs rounded-[var(--radius-sm)]',
  md: 'px-4 py-2 text-sm rounded-[var(--radius-sm)]',
  lg: 'px-6 py-2.5 text-base rounded-[var(--radius)]',
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ children, variant = 'primary', size = 'md', className, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center gap-2 font-medium transition-all duration-[var(--transition)] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed',
          variantStyles[variant],
          sizeStyles[size],
          className,
        )}
        {...props}
      >
        {children}
      </button>
    )
  },
)

Button.displayName = 'Button'
export default Button

import type { HTMLAttributes } from 'react'
import Callout from './Callout'

/**
 * Danger-variant {@link Callout} with no icon.
 *
 * Kept as its own name because "a failed read on this surface" is by far the most
 * common notice in the app, and the shorter call reads better at those sites.
 */
export default function ErrorAlert({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <Callout variant="danger" className={className} {...props} />
}

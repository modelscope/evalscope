/** Keep evidence navigation inside an embedded reader when one is present. */
export function scrollToContent(element: HTMLElement | null | undefined): void {
  if (!element) return
  const scroller = element.closest<HTMLElement>('[data-sample-scroll]')
  if (scroller) {
    const toolbar = scroller.querySelector<HTMLElement>('[data-reader-toolbar]')
    scroller.scrollTo?.({ top: scroller.scrollTop + element.getBoundingClientRect().top - scroller.getBoundingClientRect().top - (toolbar?.getBoundingClientRect().height ?? 0) - 12 })
    return
  }
  element.scrollIntoView?.({ block: 'center' })
}

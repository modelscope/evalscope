/** Render exact quotations as text nodes; evidence never introduces HTML. */
export default function EvidenceText({ text, quotes }: { text: string; quotes: string[] }) {
  const quote = quotes.find(q => q && text.includes(q))
  if (!quote) return <>{text}</>
  const at = text.indexOf(quote)
  return <>{text.slice(0, at)}<mark className="bg-[var(--warning-bg)] text-[var(--text)]">{quote}</mark>
    <EvidenceText text={text.slice(at + quote.length)} quotes={quotes} /></>
}

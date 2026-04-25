import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

interface Props {
  content: string
}

export default function MarkdownRenderer({ content }: Props) {
  if (!content) return null
  return (
    <div className="prose prose-sm prose-invert max-w-none break-words [&_table]:text-xs [&_pre]:bg-[var(--color-surface)] [&_code]:bg-[var(--color-surface)] [&_code]:px-1 [&_code]:rounded [&_img]:max-w-full [&_img]:rounded">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
        {content}
      </ReactMarkdown>
    </div>
  )
}

interface Props {
  src: string
  height?: number
  className?: string
}

export default function ChartEmbed({ src, height = 400, className }: Props) {
  return (
    <iframe
      src={src}
      className={className}
      style={{
        width: '100%',
        height,
        border: 'none',
        borderRadius: 8,
        background: 'transparent',
      }}
      sandbox="allow-scripts allow-same-origin"
      loading="lazy"
    />
  )
}

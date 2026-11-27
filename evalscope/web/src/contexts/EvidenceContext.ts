import { createContext } from 'react'

export const EvidenceByMessage = createContext<Record<string, string[]>>({})
export const EvidenceQuotes = createContext<string[]>([])

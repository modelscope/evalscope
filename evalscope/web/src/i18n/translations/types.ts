export type Locale = 'en' | 'zh'

export interface Dict {
  [key: string]: string | Dict
}

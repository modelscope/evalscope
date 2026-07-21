import type { Dict } from './types'

export const en: Dict = {
  validation: {
    required: 'This field is required.',
    datasetArgs: {
      invalidJson: 'Dataset Args must be valid JSON.',
      invalidStructure: 'Dataset Args must be a JSON object.',
    },
    numeric: {
      belowMin: 'Value is below the minimum allowed.',
      aboveMax: 'Value is above the maximum allowed.',
      stepMismatch: 'Value does not match the required step.',
      notFinite: 'Value must be a valid number.',
    },
  },
}

export const zh: Dict = {
  validation: {
    required: '此字段为必填项。',
    datasetArgs: {
      invalidJson: 'Dataset Args 必须是有效的 JSON。',
      invalidStructure: 'Dataset Args 必须是一个 JSON 对象。',
    },
    numeric: {
      belowMin: '数值低于允许的最小值。',
      aboveMax: '数值超过允许的最大值。',
      stepMismatch: '数值不符合步长要求。',
      notFinite: '数值必须是有效的数字。',
    },
  },
}

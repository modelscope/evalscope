import type { Dict } from './types'

export const en: Dict = {
  validation: {
    required: 'This field is required.',
    datasetArgs: {
      invalidJson: 'Dataset Args must be valid JSON.',
      invalidStructure: 'Dataset Args must be a JSON object.',
    },
    extraConfig: {
      invalidJson: 'Extra Config must be valid JSON.',
      invalidStructure: 'Extra Config contains a value with an unsupported type.',
      evalConfigObject: 'Use an inline JSON object or null for eval_config.',
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
    extraConfig: {
      invalidJson: '额外配置必须是有效的 JSON。',
      invalidStructure: '额外配置包含类型不受支持的值。',
      evalConfigObject: '请将 eval_config 填写为内联 JSON 对象或 null。',
    },
    numeric: {
      belowMin: '数值低于允许的最小值。',
      aboveMax: '数值超过允许的最大值。',
      stepMismatch: '数值不符合步长要求。',
      notFinite: '数值必须是有效的数字。',
    },
  },
}

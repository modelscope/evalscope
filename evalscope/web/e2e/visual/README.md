# Visual regression baselines

light/dark 双主题视觉回归基线基础设施。作为解耦/重构前后 DOM 结构与可见文本一致性的比对基线（Requirements 14.6、15.6）。

## 目录约定

```
e2e/visual/
  capture.ts          # 可复用的 light/dark 双主题基线捕获工具
  __screenshots__/    # 视觉基线快照集中存放目录（由 19.3 通过 snapshotPathTemplate 接入）
  README.md           # 本说明
```

- **快照集中存放**：所有基线图片统一放在 `e2e/visual/__screenshots__/`，而非默认散落在各测试文件旁的 `*-snapshots/` 目录。集中路径由任务 19.3 在 `playwright.config.ts` 中通过 `snapshotPathTemplate` 配置。
- **命名**：每次捕获对每个主题产出 `${name}-${theme}.png`（如 `reports-page-light.png`）。Playwright 会再自动追加当前 project（`mobile-390` / `desktop-1024`）与平台后缀，因此单次调用即可覆盖 light/dark × 视口 的基线矩阵。

## 双主题机制

主题切换复用应用自身的 theme 机制，与 `.storybook/preview.tsx` 保持一致：

- 写入 localStorage 键 `evalscope-theme`（值 `light` / `dark`）。
- `ThemeProvider`（`src/contexts/ThemeContext.tsx`）据此初始化并在 document 根元素上设置 `data-theme` 属性驱动 CSS。

`capture.ts` 提供的辅助函数据此在 light 与 dark 两主题下截图，避免临时性的样式覆盖，使快照反映真实渲染主题。

## 确定性

- **无网络**：复用 `e2e/fixtures.ts` 的离线契约（拦截所有非 localhost 请求）。API 响应通过脱敏 fixtures + `page.route` 提供。
- **固定视口**：由 `playwright.config.ts` 的 `mobile-390` / `desktop-1024` 两个 project 固定。
- **固定时区/locale/种子**：见 `playwright.config.ts` 的确定性上下文与 `PW_SEED`。
- 捕获时禁用动画（`animations: 'disabled'`）以稳定像素。

## 工具函数（`capture.ts`）

| 函数 | 用途 |
| --- | --- |
| `seedThemeBeforeLoad(page, theme)` | 导航前注入 theme，使首屏即为目标主题（适合 E2E 全新导航流程）。 |
| `applyTheme(page, theme)` | 将已加载页面切换到指定主题（写 localStorage + reload + 等待 `data-theme` 生效）。 |
| `captureThemeBaselines(page, name, options?)` | 对当前路由捕获整页 light/dark 基线。 |
| `captureElementThemeBaselines(page, target, name, options?)` | 对单个元素（组件）捕获 light/dark 基线（适合 Storybook 组件基线）。 |

## 使用示例

```ts
import { test } from '../fixtures'
import { captureThemeBaselines } from './capture'

test('reports page visual baseline', async ({ page }) => {
  await page.goto('/reports')
  // 捕获 reports-page-light.png 与 reports-page-dark.png
  await captureThemeBaselines(page, 'reports-page')
})
```

## 相关任务

- **本任务（2.2）**：仅建立目录约定、双主题捕获工具与本说明（基础设施骨架）。
- **19.1**：提供高风险组件的 Storybook stories，作为组件级基线来源。
- **19.2**：提供关键流程的 E2E，作为流程级基线来源。
- **19.3**：配置差异阈值（差异像素占比 >0.1% 判失败并标注差异区域）、`snapshotPathTemplate` 集中路径与「仅在修改 `evalscope/web` 路径的 PR 上运行」的 path filter。

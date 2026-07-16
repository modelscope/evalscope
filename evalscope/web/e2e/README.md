# E2E tests (Playwright)

端到端测试目录，配置见仓库根的 `playwright.config.ts`。

## 约定

- **本地静态预览**：测试运行在 `vite preview`（默认端口 `4173`）产出的静态构建之上，`baseURL` 为 `http://localhost:4173`。运行前需先 `npm run build`。
- **禁用外网**：所有非 localhost 请求会被 `e2e/fixtures.ts` 中的共享 `test` fixture 拦截并中断。API 响应必须通过脱敏 fixtures + `page.route` mock 提供，测试不得访问真实后端或任何第三方源。
- **确定性**：固定时区（UTC）、locale（en-US）、配色（light）与随机种子（`PW_SEED`，见 `playwright.config.ts`），保证多次运行结果一致。
- **视口预设（projects）**：
  - `mobile-390` — 390×844（移动端响应式契约）。
  - `desktop-1024` — 1024×768（桌面）。

## 结构

```
e2e/
  fixtures.ts        # 共享 test/expect，强制离线（拦截外网请求）
  example.spec.ts    # 骨架占位示例
  README.md          # 本说明
```

## 运行

```bash
npm run build      # 先构建静态产物
npm run e2e        # 运行关键流程 E2E（不生成开发机视觉基线）
npm run e2e:visual # 运行视觉回归；CI 使用已提交的 Linux goldens
```

新增流程测试时，从 `./fixtures` 导入 `test` / `expect`，并用 `page.route` 挂载脱敏 fixtures。

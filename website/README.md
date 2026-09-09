# EvalScope website

Standalone GitHub Pages source for `https://modelscope.github.io/evalscope/`. It intentionally lives only on the long-lived `website` branch and is never merged into `main`.

## Local checks

```bash
cd website
npm ci
npm run verify
npm run dev
```

The site is static. It does not run models, request model APIs, download datasets, or require a GPU during build or deploy.

## Updating from EvalScope

```bash
git fetch origin
git switch website
git merge origin/main
cd website && npm run verify
git push origin website
```

Resolve only website-adjacent conflicts. The benchmark catalog is regenerated from the current branch's benchmark metadata at build time, making each sync auditable.

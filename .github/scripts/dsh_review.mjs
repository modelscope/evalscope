import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const action = resolve('.dsh-action');
const entrypoint = join(action, 'dist/index.js');
const upstreamDigest = '149636796b47bd9749cbb18ae0f10c3c71f25a4b9ba38d8f2d75ff9636027d81';
const review = join(process.env.RUNNER_TEMP, 'evalscope-review');
const { GH_TOKEN, PR_REVIEW_API_KEY, PR_REVIEW_MODEL, PR_REVIEW_REASONING_EFFORT, ...environment } = process.env;
if (!GH_TOKEN || !PR_REVIEW_API_KEY || !PR_REVIEW_MODEL) {
  throw new Error('AI review requires the configured model and credentials.');
}

function replaceOnce(source, original, replacement) {
  if (source.split(original).length !== 2) {
    throw new Error('The pinned DSH-action integration point changed.');
  }
  return source.replace(original, replacement);
}

let source = readFileSync(entrypoint, 'utf8');
if (createHash('sha256').update(source).digest('hex') !== upstreamDigest) {
  throw new Error('Unexpected DSH-action distribution. Revalidate the integration before upgrading it.');
}
// Use the upstream proxy's 32 MiB default for streamed thinking. Keep its 2 MiB worker log limit.
source = replaceOnce(source, ',maxResponseBytes:d.maxOutputBytes', '');
// Recheck readiness before DSH publishes inline, including reviews that started before conversion to draft.
source = replaceOnce(
  source,
  'if(d.data.state!=="open")throw new Error("Pull request is no longer open")',
  'if(d.data.state!=="open"||d.data.draft)throw new Error("Pull request is no longer open and ready")',
);
// Keep DSH's inline publisher and give its rendered summary to EvalScope's existing sticky comment.
source = replaceOnce(
  source,
  'await(0,Wi.kv)(a,{owner:c.owner,repo:c.repo,issueNumber:c.pullNumber},c.expectedAuthorId,"summary",k.body)',
  'await (await import("node:fs/promises")).writeFile(process.env.EVALSCOPE_DSH_SUMMARY_PATH,k.body,"utf8")',
);
writeFileSync(entrypoint, source);

const inputs = {
  'allow-write': 'false',
  'base-url': 'https://dashscope.aliyuncs.com/apps/anthropic',
  command: 'review',
  'deepseek-api-key': PR_REVIEW_API_KEY,
  diagnostics: 'off',
  'exclude-comments-by-actor': 'github-actions[bot]',
  'github-token': GH_TOKEN,
  'max-findings': '100',
  model: PR_REVIEW_MODEL,
  'progress-comment': 'false',
  prompt: readFileSync(join(review, 'prompt.txt'), 'utf8'),
  'reasoning-effort': PR_REVIEW_REASONING_EFFORT || 'high',
  'timeout-minutes': '15',
  'worker-network': 'false',
};
const actionInputs = Object.fromEntries(Object.entries(inputs).map(([key, value]) => [`INPUT_${key.toUpperCase()}`, value]));
const run = spawn(process.execPath, [entrypoint], {
  env: {
    ...environment,
    ...actionInputs,
    EVALSCOPE_DSH_SUMMARY_PATH: join(review, 'dsh-summary.md'),
    GITHUB_ACTION_PATH: action,
    GITHUB_EVENT_NAME: 'pull_request_target',
    GITHUB_EVENT_PATH: join(review, 'pull-request.json'),
    // The workflow already authorized the request. Use its bot identity to keep
    // the same read-only, non-executing DSH policy for forks and internal branches.
    GITHUB_ACTOR: 'github-actions[bot]',
  },
  stdio: 'inherit',
});
run.on('error', (error) => { throw error; });
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => run.kill(signal));
}
run.on('exit', (code) => { process.exitCode = code ?? 1; });

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { accessSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const action = resolve('.dsh-action');
const entrypoint = join(action, 'dist/index.js');
const upstreamDigest = '149636796b47bd9749cbb18ae0f10c3c71f25a4b9ba38d8f2d75ff9636027d81';
const review = join(process.env.RUNNER_TEMP, 'evalscope-review');
const workspace = join(review, 'workspace');
const { GH_TOKEN, PR_REVIEW_API_KEY, PR_REVIEW_MODEL, PR_REVIEW_REASONING_EFFORT, ...environment } = process.env;
if (!GH_TOKEN || !PR_REVIEW_API_KEY || !PR_REVIEW_MODEL) {
  throw new Error('AI review requires the configured model and credentials.');
}
const eventPath = join(review, 'pull-request.json');
const headSha = JSON.parse(readFileSync(eventPath, 'utf8')).pull_request.head.sha;
const evidence = join(workspace, '.evalscope-review');
accessSync(join(evidence, 'changes.diff'));
if (JSON.parse(readFileSync(join(evidence, 'ci.json'), 'utf8')).head_sha !== headSha) {
  throw new Error('The CI evidence is bound to a different commit.');
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
// Opt this operation into the controlled Docker read profile. PR trust cannot
// grant execution, extensions, network access, or repository writes.
source = replaceOnce(
  source,
  'if(!g){return{trust:h&&A?"trusted-read":"untrusted"',
  'if(!g&&l==="review"&&process.env.EVALSCOPE_DSH_WORKSPACE){' +
    'return{trust:"trusted-read",allowed:true,reason:"Read-only immutable EvalScope review workspace",' +
    'capabilities:{...b,loadExtensions:false,accessNetwork:false}}}' +
    'if(!g){return{trust:h&&A?"trusted-read":"untrusted"',
);
// Mount Git blobs already staged by the workflow, separate from its checkout.
source = replaceOnce(
  source,
  'b=await(0,Be.mkdtemp)((0,p.join)((0,Ye.tmpdir)(),"dsh-action-workspace-"));const a=(0,p.join)(b,"source");',
  'b=await(0,Be.mkdtemp)((0,p.join)((0,Ye.tmpdir)(),"dsh-action-workspace-"));' +
    'if(process.env.EVALSCOPE_DSH_WORKSPACE){' +
    'if(A.trust!=="trusted-read"||g.isolation!=="docker"||d?.kind!=="pull_request"||' +
    'd.headSha!==process.env.EVALSCOPE_DSH_SHA)throw new Error("Invalid EvalScope review workspace binding");' +
    'return{tempRoot:b,agentWorkspace:process.env.EVALSCOPE_DSH_WORKSPACE}}' +
    'const a=(0,p.join)(b,"source");',
);
// Refuse publication if the agent skipped repository evidence collection.
source = replaceOnce(
  source,
  'return publishPullRequestReview(a,c,l,{operation:"review",summary:u.output.summary,findings:u.output.findings,',
  'const receipts=(u.toolReceipts??[]).filter(r=>r.completed&&r.counted&&r.ok);' +
    'if(!["workspace.read","workspace.search"].every(id=>receipts.some(r=>r.id===id)))' +
    'throw new Error("The review did not complete repository reads and searches");' +
    'const text=u.output.summary.trim().split(/\\r?\\n/,1)[0];' +
    'const reviewStatus=text.startsWith("Limited: ")?text.slice(0,200):text.startsWith("Checked: ")?text:' +
    '(text==="Static review complete."||u.output.findings.length>0?' +
    '"Static review complete.":"Limited: The model did not return a concise review status.");' +
    'return publishPullRequestReview(a,c,l,{operation:"review",summary:reviewStatus,findings:u.output.findings,',
);
source = replaceOnce(
  source,
  'const g=selectPublicationFindings(h,u.findings,d);let A=0',
  'const g=selectPublicationFindings(h,u.findings,d);' +
    'const eligible=new Set(filterHighPrecisionFindings(u.findings));' +
    'const rejected=u.findings.filter(f=>!eligible.has(f));' +
    'const filtered=rejected.map(({severity,category,confidence,evidence})=>' +
    '({severity,category,confidence,hasEvidence:evidence!==undefined}));' +
    'const notes=rejected.filter(f=>f.evidence!==undefined).map(f=>({' +
    'title:safeBody(f.title),path:(0,ae.oR)(f.path),line:f.line,confidence:f.confidence,evidence:safeBody(f.evidence)}));' +
    'const selection={candidates:u.findings.length,severities:g.map(({finding})=>finding.severity)};' +
    'console.log("EvalScope finding selection",JSON.stringify({...selection,filtered,duplicatesOrLimit:eligible.size-g.length}));' +
    'await (await import("node:fs/promises")).writeFile(process.env.EVALSCOPE_DSH_FINDINGS_PATH,' +
    'JSON.stringify({...selection,notes}));let A=0',
);
// Use the same explicit severity names in inline and summary-only headings.
source = replaceOnce(
  source,
  'function severityIcon(a){switch(a){case"critical":return"🔴";case"high":return"🟠";case"medium":return"🟡";case"low":return"🔵"}}',
  'function severityIcon(a){switch(a){case"critical":return"🔴 Critical ·";case"high":return"🟠 High ·";' +
    'case"medium":return"🟡 Medium ·";case"low":return"🔵 Low ·"}}',
);
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
  'allowed-tools': '["workspace.read","workspace.search"]',
  'base-url': 'https://dashscope.aliyuncs.com/apps/anthropic',
  command: 'review',
  'deepseek-api-key': PR_REVIEW_API_KEY,
  diagnostics: 'off',
  'exclude-comments-by-actor': 'github-actions[bot]',
  'github-token': GH_TOKEN,
  'max-findings': '100',
  model: PR_REVIEW_MODEL,
  'permission-profile': 'custom',
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
    EVALSCOPE_DSH_FINDINGS_PATH: join(review, 'dsh-findings.json'),
    EVALSCOPE_DSH_WORKSPACE: workspace,
    EVALSCOPE_DSH_SHA: headSha,
    GITHUB_ACTION_PATH: action,
    GITHUB_EVENT_NAME: 'pull_request_target',
    GITHUB_EVENT_PATH: eventPath,
    // The workflow authorized the request. The same read-only profile applies
    // to forks and internal branches.
    GITHUB_ACTOR: 'github-actions[bot]',
  },
  stdio: 'inherit',
});
run.on('error', (error) => { throw error; });
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => run.kill(signal));
}
run.on('exit', (code) => { process.exitCode = code ?? 1; });

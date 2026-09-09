import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const metaDir = path.resolve(root, '..', 'evalscope', 'benchmarks', '_meta');
const files = fs.readdirSync(metaDir).filter((file) => file.endsWith('.json'));
if (files.length === 0) throw new Error('No benchmark metadata files found.');
for (const file of files) {
  const data = JSON.parse(fs.readFileSync(path.join(metaDir, file), 'utf8'));
  if (!data.meta || (!data.meta.name && !data.meta.pretty_name))
    throw new Error(`Missing benchmark display name in ${file}`);
}
const source = fs
  .readdirSync(path.join(root, 'src'), { recursive: true })
  .filter((file) => String(file).endsWith('.astro'))
  .map((file) => fs.readFileSync(path.join(root, 'src', String(file)), 'utf8'))
  .join('\n');
for (const forbidden of ['localhost', 'AstroWind', 'sk-'])
  if (source.includes(forbidden)) throw new Error(`Forbidden website content: ${forbidden}`);
for (const label of ['Available Today', 'Exploring', 'External Research'])
  if (!source.includes(label)) throw new Error(`Research label missing: ${label}`);
console.log(`Content checks passed: ${files.length} benchmark metadata records.`);

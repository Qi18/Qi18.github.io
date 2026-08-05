import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd(), 'src/content/posts');
const patterns = [
  ['private key', /-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----/],
  ['AWS access key', /AKIA[0-9A-Z]{16}/],
  ['Alibaba Cloud access key', /LTAI[0-9A-Za-z]{12,}/],
  ['GitHub token', /gh[pousr]_[A-Za-z0-9_]{20,}/],
];

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...await walk(full));
    else if (/\.(?:md|mdx)$/i.test(entry.name)) files.push(full);
  }
  return files;
}

const violations = [];
for (const file of await walk(root)) {
  const text = await readFile(file, 'utf8');
  for (const [name, pattern] of patterns) {
    if (pattern.test(text)) violations.push(`${path.relative(process.cwd(), file)}: ${name}`);
  }
}

if (violations.length) {
  console.error('Public content safety check failed:');
  for (const item of violations) console.error(`- ${item}`);
  process.exit(1);
}

console.log('Public content safety check passed.');

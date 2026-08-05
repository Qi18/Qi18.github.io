import { createHash } from 'node:crypto';
import {
  cp,
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const blogRoot = path.resolve(process.env.BLOG_ROOT || process.cwd());
const vaultRoot = path.resolve(
  process.env.OBSIDIAN_VAULT || '/Users/rich/Documents/Obsidian Vault/knowledge',
);
const postsDir = path.resolve(
  process.env.OBSIDIAN_OUTPUT_DIR || path.join(blogRoot, 'src/content/posts'),
);
const assetsDir = path.resolve(
  process.env.OBSIDIAN_ASSET_DIR || path.join(blogRoot, 'public/obsidian-assets'),
);
const imageExtensions = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.avif']);
const secretPatterns = [
  ['private key', /-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----/],
  ['AWS access key', /AKIA[0-9A-Z]{16}/],
  ['Alibaba Cloud access key', /LTAI[0-9A-Za-z]{12,}/],
  ['GitHub token', /gh[pousr]_[A-Za-z0-9_]{20,}/],
];

function parseScalar(raw) {
  const value = raw.trim();
  if (!value) return '';
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    return value.slice(1, -1);
  }
  if (value === 'true') return true;
  if (value === 'false') return false;
  if (/^-?\d+$/.test(value)) return Number(value);
  if (value.startsWith('[') && value.endsWith(']')) {
    return value.slice(1, -1).split(',').map((item) => parseScalar(item)).filter(Boolean);
  }
  return value;
}

function parseFrontmatter(text, file) {
  const normalized = text.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n');
  if (!normalized.startsWith('---\n')) return { data: {}, body: normalized };
  const end = normalized.indexOf('\n---\n', 4);
  if (end < 0) throw new Error(`${file}: frontmatter is not closed`);
  const block = normalized.slice(4, end);
  const body = normalized.slice(end + 5);
  const data = {};
  let listKey = null;
  for (const line of block.split('\n')) {
    const listItem = line.match(/^\s+-\s+(.+)$/);
    if (listItem && listKey) {
      if (!Array.isArray(data[listKey])) data[listKey] = [];
      data[listKey].push(parseScalar(listItem[1]));
      continue;
    }
    const match = line.match(/^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$/);
    if (!match) {
      if (line.trim() && !line.trim().startsWith('#')) {
        throw new Error(`${file}: unsupported frontmatter line: ${line.trim()}`);
      }
      continue;
    }
    const [, key, raw] = match;
    data[key] = raw.trim() ? parseScalar(raw) : [];
    listKey = raw.trim() ? null : key;
  }
  return { data, body };
}

async function walkMarkdown(dir) {
  const files = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...await walkMarkdown(full));
    else if (entry.isFile() && entry.name.toLowerCase().endsWith('.md')) files.push(full);
  }
  return files;
}

async function walkFiles(dir) {
  const files = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.name === '.git') continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...await walkFiles(full));
    else if (entry.isFile()) files.push(full);
  }
  return files;
}

function ensureSafePublicNote(note) {
  const required = ['slug', 'title', 'description', 'publishedAt'];
  for (const key of required) {
    if (note.data[key] === undefined || note.data[key] === '') {
      throw new Error(`${note.relative}: publish:true requires ${key}`);
    }
  }
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(String(note.data.slug))) {
    throw new Error(`${note.relative}: slug must use lowercase letters, numbers, and hyphens`);
  }
  for (const [name, pattern] of secretPatterns) {
    if (pattern.test(note.source)) throw new Error(`${note.relative}: contains a possible ${name}`);
  }
}

function yamlString(value) {
  return JSON.stringify(String(value));
}

function renderFrontmatter(data) {
  const lines = [
    '---',
    `title: ${yamlString(data.title)}`,
    `description: ${yamlString(data.description)}`,
    `publishedAt: ${yamlString(data.publishedAt)}`,
  ];
  if (data.updatedAt) lines.push(`updatedAt: ${yamlString(data.updatedAt)}`);
  const tags = Array.isArray(data.tags) ? data.tags : data.tags ? [data.tags] : [];
  lines.push('tags:');
  for (const tag of tags) lines.push(`  - ${yamlString(tag)}`);
  if (Array.isArray(data.knowledgePath) && data.knowledgePath.length) {
    lines.push('knowledgePath:');
    for (const segment of data.knowledgePath) lines.push(`  - ${yamlString(segment)}`);
  }
  if (data.knowledgeOrder) lines.push(`knowledgeOrder: ${yamlString(data.knowledgeOrder)}`);
  if (data.series) lines.push(`series: ${yamlString(data.series)}`);
  if (data.seriesOrder !== undefined && data.seriesOrder !== '') {
    lines.push(`seriesOrder: ${Number(data.seriesOrder)}`);
  }
  lines.push(`draft: ${data.draft === true ? 'true' : 'false'}`);
  lines.push(`featured: ${data.featured === true ? 'true' : 'false'}`);
  lines.push('---', '', '<!-- Generated from Obsidian. Do not edit directly. -->', '', '');
  return lines.join('\n');
}

function addLookup(map, key, note) {
  if (!key) return;
  const normalized = key.replace(/\\/g, '/').replace(/\.md$/i, '').trim().toLowerCase();
  if (!normalized) return;
  if (!map.has(normalized)) map.set(normalized, note);
  else if (map.get(normalized) !== note) map.set(normalized, null);
}

function normalizeWikiTarget(target) {
  return target.split('#', 1)[0].replace(/\\/g, '/').replace(/\.md$/i, '').trim().toLowerCase();
}

function sanitizeAssetName(name) {
  return name.normalize('NFKC').replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || 'asset';
}

function isInside(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

async function moveDirectory(source, target) {
  try {
    await rename(source, target);
  } catch (error) {
    if (error.code !== 'EXDEV') throw error;
    await cp(source, target, { recursive: true });
    await rm(source, { recursive: true, force: true });
  }
}

async function replaceGeneratedPosts(source, target) {
  await mkdir(target, { recursive: true });
  const generatedEntries = (await readdir(source, { withFileTypes: true })).filter((entry) => entry.isFile());
  for (const entry of generatedEntries) {
    const existing = await readFile(path.join(target, entry.name), 'utf8').catch(() => null);
    if (existing !== null && !existing.includes('<!-- Generated from Obsidian. Do not edit directly. -->')) {
      throw new Error(`Public slug conflicts with a hand-written post: ${entry.name}`);
    }
  }

  for (const entry of await readdir(target, { withFileTypes: true })) {
    if (!entry.isFile() || !/\.(?:md|mdx)$/i.test(entry.name)) continue;
    const file = path.join(target, entry.name);
    const text = await readFile(file, 'utf8');
    if (text.includes('<!-- Generated from Obsidian. Do not edit directly. -->')) {
      await rm(file, { force: true });
    }
  }

  await rm(path.join(target, 'obsidian'), { recursive: true, force: true });
  for (const entry of generatedEntries) {
    const from = path.join(source, entry.name);
    const to = path.join(target, entry.name);
    await cp(from, to);
  }
  await rm(source, { recursive: true, force: true });
}

async function main() {
  const vaultInfo = await stat(vaultRoot).catch(() => null);
  if (!vaultInfo?.isDirectory()) throw new Error(`Vault does not exist: ${vaultRoot}`);

  const markdownFiles = await walkMarkdown(vaultRoot);
  const notes = [];
  for (const file of markdownFiles) {
    const source = await readFile(file, 'utf8');
    const relative = path.relative(vaultRoot, file);
    const { data, body } = parseFrontmatter(source, relative);
    if (data.publish !== true) continue;
    const directory = path.dirname(relative);
    data.knowledgePath = directory === '.' ? [] : directory.split(path.sep);
    data.knowledgeOrder = relative.replace(/\.md$/i, '').split(path.sep).join('/');
    const note = { file, relative, source, data, body };
    ensureSafePublicNote(note);
    notes.push(note);
  }

  const slugSet = new Set();
  const noteLookup = new Map();
  for (const note of notes) {
    const slug = String(note.data.slug);
    if (slugSet.has(slug)) throw new Error(`Duplicate public slug: ${slug}`);
    slugSet.add(slug);
    addLookup(noteLookup, note.relative, note);
    addLookup(noteLookup, path.basename(note.relative), note);
    addLookup(noteLookup, note.data.title, note);
    addLookup(noteLookup, slug, note);
  }

  const allVaultFiles = await walkFiles(vaultRoot);
  const assetLookup = new Map();
  for (const file of allVaultFiles) {
    const key = path.basename(file).toLowerCase();
    if (!assetLookup.has(key)) assetLookup.set(key, [file]);
    else assetLookup.get(key).push(file);
  }

  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'obsidian-blog-sync-'));
  const tempPosts = path.join(tempRoot, 'posts');
  const tempAssets = path.join(tempRoot, 'assets');
  await mkdir(tempPosts, { recursive: true });
  await mkdir(tempAssets, { recursive: true });

  async function copyAsset(rawTarget, note) {
    const cleanTarget = decodeURIComponent(rawTarget.split('#', 1)[0].trim());
    const candidates = [];
    const sourceRelative = path.resolve(path.dirname(note.file), cleanTarget);
    const vaultRelative = path.resolve(vaultRoot, cleanTarget);
    for (const candidate of [sourceRelative, vaultRelative]) {
      if (isInside(vaultRoot, candidate)) {
        const info = await stat(candidate).catch(() => null);
        if (info?.isFile()) candidates.push(candidate);
      }
    }
    if (!candidates.length) candidates.push(...(assetLookup.get(path.basename(cleanTarget).toLowerCase()) || []));
    const unique = [...new Set(candidates)];
    if (unique.length !== 1) {
      throw new Error(`${note.relative}: attachment must resolve exactly once: ${cleanTarget}`);
    }
    const source = unique[0];
    const extension = path.extname(source).toLowerCase();
    if (!imageExtensions.has(extension)) {
      throw new Error(`${note.relative}: only image embeds can be public: ${cleanTarget}`);
    }
    const bytes = await readFile(source);
    const digest = createHash('sha256').update(bytes).digest('hex').slice(0, 8);
    const base = sanitizeAssetName(path.basename(source, extension));
    const outputName = `${base}-${digest}${extension}`;
    const outputDir = path.join(tempAssets, String(note.data.slug));
    await mkdir(outputDir, { recursive: true });
    await writeFile(path.join(outputDir, outputName), bytes);
    return `/obsidian-assets/${note.data.slug}/${outputName}`;
  }

  for (const note of notes) {
    let body = note.body;
    const embeds = [...body.matchAll(/!\[\[([^\]]+)\]\]/g)];
    for (const match of embeds) {
      const target = match[1].split('|', 1)[0];
      const publicPath = await copyAsset(target, note);
      body = body.replace(match[0], `![](${publicPath})`);
    }

    const markdownImages = [...body.matchAll(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g)];
    for (const match of markdownImages) {
      const target = match[2];
      if (/^(?:https?:|data:|\/)/i.test(target)) continue;
      const publicPath = await copyAsset(target, note);
      body = body.replace(match[0], `![${match[1]}](${publicPath})`);
    }

    body = body.replace(/\[\[([^\]]+)\]\]/g, (_, raw) => {
      const [targetWithHeading, alias] = raw.split('|', 2);
      const display = alias || targetWithHeading.split('#', 1)[0];
      const target = noteLookup.get(normalizeWikiTarget(targetWithHeading));
      return target ? `[${display}](/posts/${target.data.slug}/)` : display;
    });

    const generated = `${renderFrontmatter(note.data)}${body.replace(/^\n*#\s+[^\n]+\n+/, '')}`;
    await writeFile(path.join(tempPosts, `${note.data.slug}.md`), generated, 'utf8');
  }

  await rm(assetsDir, { recursive: true, force: true });
  await mkdir(path.dirname(assetsDir), { recursive: true });
  await replaceGeneratedPosts(tempPosts, postsDir);
  await moveDirectory(tempAssets, assetsDir);
  await rm(tempRoot, { recursive: true, force: true });

  console.log(`Obsidian sync complete: ${notes.length} public note(s).`);
}

main().catch((error) => {
  console.error(`Obsidian sync failed: ${error.message}`);
  process.exit(1);
});

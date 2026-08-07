import type { CollectionEntry } from 'astro:content';

export type PostEntry = CollectionEntry<'posts'>;

export interface SeriesTreeNode {
  name: string;
  path: string[];
  posts: PostEntry[];
  directPosts: PostEntry[];
  children: SeriesTreeNode[];
}

export const sortPostsByDate = (posts: PostEntry[]) =>
  [...posts].sort((a, b) => b.data.publishedAt.valueOf() - a.data.publishedAt.valueOf());

export const sortPostsBySeries = (posts: PostEntry[]) =>
  [...posts].sort((a, b) => {
    const order = (a.data.seriesOrder ?? Number.MAX_SAFE_INTEGER) - (b.data.seriesOrder ?? Number.MAX_SAFE_INTEGER);
    return order || a.data.publishedAt.valueOf() - b.data.publishedAt.valueOf();
  });

export function getSeriesPath(post: PostEntry) {
  const path = post.data.seriesPath?.map((segment) => segment.trim()).filter(Boolean) ?? [];
  if (path.length) return path;
  return post.data.series ? [post.data.series] : [];
}

export function getSeriesUrl(path: string[]) {
  return `/series/${path.map(encodeURIComponent).join('/')}/`;
}

export function isSameSeriesPath(left: string[], right: string[]) {
  return left.length === right.length && left.every((segment, index) => segment === right[index]);
}

export function isSeriesPathPrefix(prefix: string[], path: string[]) {
  return prefix.length <= path.length && prefix.every((segment, index) => segment === path[index]);
}

export function buildSeriesTree(posts: PostEntry[]): SeriesTreeNode[] {
  const roots: SeriesTreeNode[] = [];

  for (const post of posts) {
    const path = getSeriesPath(post);
    if (!path.length) continue;

    let siblings = roots;
    path.forEach((name, index) => {
      let node = siblings.find((candidate) => candidate.name === name);
      if (!node) {
        node = { name, path: path.slice(0, index + 1), posts: [], directPosts: [], children: [] };
        siblings.push(node);
      }
      node.posts.push(post);
      if (index === path.length - 1) node.directPosts.push(post);
      siblings = node.children;
    });
  }

  const sortNodes = (nodes: SeriesTreeNode[]) => {
    nodes.sort((a, b) => {
      const newestA = Math.max(...a.posts.map((post) => post.data.publishedAt.valueOf()));
      const newestB = Math.max(...b.posts.map((post) => post.data.publishedAt.valueOf()));
      return newestB - newestA || a.name.localeCompare(b.name, 'zh-CN');
    });
    nodes.forEach((node) => {
      node.posts = sortPostsByDate(node.posts);
      node.directPosts = sortPostsBySeries(node.directPosts);
      sortNodes(node.children);
    });
  };

  sortNodes(roots);
  return roots;
}

export function flattenSeriesTree(nodes: SeriesTreeNode[]): SeriesTreeNode[] {
  return nodes.flatMap((node) => [node, ...flattenSeriesTree(node.children)]);
}

export function getReadingMinutes(body = '') {
  const chineseCharacters = body.match(/[\u3400-\u9fff]/g)?.length ?? 0;
  const latinWords = body
    .replace(/[\u3400-\u9fff]/g, ' ')
    .match(/[A-Za-z0-9]+(?:[-_.'’][A-Za-z0-9]+)*/g)?.length ?? 0;

  return Math.max(1, Math.ceil(chineseCharacters / 300 + latinWords / 200));
}

export function toSearchText(body = '') {
  return body
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`]*`/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/[#>*_~|\-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

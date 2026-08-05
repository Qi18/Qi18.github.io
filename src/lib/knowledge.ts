import type { PostEntry } from './posts';

export interface KnowledgeNode {
  name: string;
  path: string[];
  posts: PostEntry[];
  children: KnowledgeNode[];
  totalPosts: number;
}

interface MutableKnowledgeNode extends Omit<KnowledgeNode, 'children'> {
  children: MutableKnowledgeNode[];
  childMap: Map<string, MutableKnowledgeNode>;
}

const collator = new Intl.Collator('zh-CN', { numeric: true, sensitivity: 'base' });

export function getKnowledgeAnchor(path: string[]) {
  const normalized = path.join('-').normalize('NFKC').replace(/[^\p{Letter}\p{Number}-]+/gu, '-');
  return `knowledge-${normalized.replace(/^-+|-+$/g, '').toLocaleLowerCase('zh-CN')}`;
}

export function getKnowledgeNumber(post: PostEntry) {
  const fileName = post.data.knowledgeOrder?.split('/').at(-1) ?? '';
  return fileName.match(/^(\d+(?:\.\d+)*)/)?.[1] ?? '';
}

function countPosts(node: MutableKnowledgeNode): number {
  node.totalPosts = node.posts.length + node.children.reduce((total, child) => total + countPosts(child), 0);
  return node.totalPosts;
}

function sortNode(node: MutableKnowledgeNode) {
  node.children.sort((a, b) => collator.compare(a.name, b.name));
  node.posts.sort((a, b) => collator.compare(
    a.data.knowledgeOrder ?? a.data.title,
    b.data.knowledgeOrder ?? b.data.title,
  ));
  node.children.forEach(sortNode);
}

export function buildKnowledgeTree(posts: PostEntry[]): KnowledgeNode[] {
  const root: MutableKnowledgeNode = {
    name: '',
    path: [],
    posts: [],
    children: [],
    childMap: new Map(),
    totalPosts: 0,
  };

  for (const post of posts) {
    let parent = root;
    for (const segment of post.data.knowledgePath ?? []) {
      let child = parent.childMap.get(segment);
      if (!child) {
        child = {
          name: segment,
          path: [...parent.path, segment],
          posts: [],
          children: [],
          childMap: new Map(),
          totalPosts: 0,
        };
        parent.childMap.set(segment, child);
        parent.children.push(child);
      }
      parent = child;
    }
    parent.posts.push(post);
  }

  sortNode(root);
  root.children.forEach(countPosts);
  return root.children;
}

import type { CollectionEntry } from 'astro:content';

export type PostEntry = CollectionEntry<'posts'>;

export const sortPostsByDate = (posts: PostEntry[]) =>
  [...posts].sort((a, b) => b.data.publishedAt.valueOf() - a.data.publishedAt.valueOf());

export const sortPostsBySeries = (posts: PostEntry[]) =>
  [...posts].sort((a, b) => {
    const order = (a.data.seriesOrder ?? Number.MAX_SAFE_INTEGER) - (b.data.seriesOrder ?? Number.MAX_SAFE_INTEGER);
    return order || a.data.publishedAt.valueOf() - b.data.publishedAt.valueOf();
  });

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

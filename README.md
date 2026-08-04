# Rich's Notebook

一个使用 Astro 构建的个人博客，内容以 Markdown / MDX 保存。

## 本地运行

```bash
pnpm install
pnpm dev
```

访问 `http://localhost:4321`。

## 发布文章

在 `src/content/posts/` 新建 Markdown 文件，Frontmatter 字段示例：

```yaml
---
title: "文章标题"
description: "文章摘要"
publishedAt: 2026-08-04
updatedAt: 2026-08-04
tags:
  - LLM
series: "系列名称"
draft: false
featured: false
---
```

## 上线前需要替换

1. 修改 `src/site.config.ts` 中的站名、邮箱和 GitHub 地址。
2. 修改 `astro.config.mjs` 中的正式域名。
3. 按需要替换首页与关于页中的个人介绍。
4. 删除或改写三篇示例文章。

## 构建

```bash
pnpm build
pnpm preview
```

`dist/` 目录可以直接部署到 Cloudflare Pages、Vercel 或其他静态托管平台。

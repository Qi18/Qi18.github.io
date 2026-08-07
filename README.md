# Rich's Notebook

一个使用 Astro 构建的个人博客，内容以 Markdown / MDX 保存。

## V7 功能

- 用 `seriesPath` 把系列组织为可展开的文件夹树。
- 父目录聚合子系列文章，叶子目录保留系列阅读顺序。
- 文章页展示完整系列面包屑，同时保留旧的单层系列 URL。

## V6 功能

- 用“系列”作为唯一的内容组织入口。
- 移除重复的知识库树、目录面包屑和一级文章导航。
- 保留全文搜索、标签、文章上下篇和 Obsidian 发布白名单。

## V5 功能

- 公开文章自动继承 Obsidian 的目录层级和数字编号。
- 新增知识库树形入口，只展示包含公开文章的目录分支。
- 文章页显示知识库面包屑，搜索支持目录关键词。
- 手写博客文章与 Obsidian 知识笔记保持独立展示。

## V4 功能

- 完整 Obsidian Vault 保存到私有仓库 `Qi18/knowledge-vault-private`。
- 博客只同步明确标记 `publish: true` 的文章，默认不公开。
- 自动转换公开文章间的 Wiki Link，并只复制公开文章实际引用的图片。
- 构建前执行公开内容敏感信息检查。

详细规则见 [`docs/obsidian-publishing.md`](docs/obsidian-publishing.md)。

## V3 功能

- 更完整的个人主页、当前关注和知识阅读地图。
- 独立更新日志，记录每一版解决的问题。
- 文章原生分享、复制链接和返回顶部。
- 保留 V2 的系列、全文搜索和长文阅读能力。

## V2 功能

- 按系列组织文章并提供顺序阅读入口。
- 静态全文搜索，无需后端服务。
- 文章阅读时间、阅读进度、移动端目录和代码复制。
- 上一篇 / 下一篇文章导航。
- GitHub Actions 自动部署到 GitHub Pages。

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
seriesPath:
  - "一级目录"
  - "系列名称"
seriesOrder: 1
draft: false
featured: false
---
```

也可以在 Obsidian 笔记中加入 `publish: true`，然后执行：

```bash
OBSIDIAN_VAULT="/Users/rich/Documents/Obsidian Vault/knowledge" pnpm sync:obsidian
pnpm build
```

## 内容配置

1. 在 `src/site.config.ts` 中维护站名和 GitHub 地址。
2. 在 `astro.config.mjs` 中维护正式域名。
3. 按需要修改首页与关于页中的个人介绍。
4. 在 Frontmatter 中使用 `seriesPath` 组织目录层级，`series` 保留为叶子系列名，`seriesOrder` 控制系列内阅读顺序。

## 构建

```bash
pnpm build
pnpm preview
```

`main` 分支推送到 GitHub 后，由 `.github/workflows/deploy-pages.yml` 自动构建并发布到 GitHub Pages。

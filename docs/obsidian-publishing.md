# Obsidian 与博客同步规则

## 仓库边界

- `Qi18/knowledge-vault-private`：完整 Vault 的私有备份，包含耐久笔记、附件和稳定的 Obsidian 配置。
- `Qi18/Qi18.github.io`：公开博客，只包含手写博客文章和明确允许公开的 Obsidian 文章。
- 没有 `publish: true` 的笔记一律不生成到公开博客。

## 公开一篇笔记

在 Obsidian 笔记顶部加入完整 Frontmatter：

```yaml
---
publish: true
slug: rag-system-design
title: "从 Demo 到系统：RAG 工程设计"
description: "从检索、重排到评估，整理一套可复用的 RAG 工程方法。"
publishedAt: 2026-08-05
updatedAt: 2026-08-05
tags:
  - RAG
  - LLM
series: "LLM 工程"
seriesOrder: 1
featured: false
---
```

其中 `slug`、`title`、`description`、`publishedAt` 是必填项。`slug` 只能使用小写英文字母、数字和连字符。

## 执行同步

在博客目录运行：

```bash
OBSIDIAN_VAULT="/Users/rich/Documents/Obsidian Vault/knowledge" pnpm sync:obsidian
pnpm build
```

同步器会：

1. 只读取 `publish: true` 的 Markdown。
2. 把公开笔记间的 `[[Wiki Link]]` 转成博客链接。
3. 把指向未公开笔记的 Wiki Link 转成纯文本，不泄露正文。
4. 只复制公开笔记实际引用的图片到 `public/obsidian-assets/`。
5. 遇到重复 slug、缺失附件、非图片嵌入或高置信度密钥特征时立即失败。

生成文件位于 `src/content/posts/obsidian/`，不要直接编辑。

## 私有快照排除项

完整快照不会包含旧 `.git` 历史、`.DS_Store`、`._*`、`.qoder/`、`.trash/`、Obsidian 工作区状态、缓存、`node_modules/` 和临时文件。

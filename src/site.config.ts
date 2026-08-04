export const siteConfig = {
  title: "Rich's Notebook",
  shortTitle: 'RN',
  description: '记录 LLM、工程实践与持续学习。',
  author: 'Rich',
  email: 'hello@example.com',
  github: 'https://github.com/',
  nav: [
    { href: '/', label: '首页' },
    { href: '/posts', label: '文章' },
    { href: '/tags', label: '标签' },
    { href: '/about', label: '关于' },
  ],
} as const;

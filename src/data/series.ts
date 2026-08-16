export interface SeriesLink {
  label: string;
  href: string;
}

export interface SeriesOverview {
  title: string;
  description: string;
  links: SeriesLink[];
}

export interface SeriesDefinition {
  path: string[];
  overview?: SeriesOverview;
}

export const seriesDefinitions: SeriesDefinition[] = [
  { path: ['LLM', 'Train'] },
  { path: ['LLM', 'Agent'] },
  { path: ['LLM', 'Agent RL'] },
  { path: ['LLM', '基础'] },
  { path: ['LLM', '评测'] },
  {
    path: ['LLM', 'Train', 'NanoChat'],
    overview: {
      title: 'NanoChat 项目说明',
      description: 'NanoChat 是 Andrej Karpathy 开源的简洁 LLM 实验框架，面向单机 GPU 节点，覆盖分词、预训练、微调、评测和推理等主要阶段。这个系列沿着真实执行链路记录源码理解与训练实验。',
      links: [
        { label: 'GitHub 项目地址', href: 'https://github.com/karpathy/nanochat' },
        { label: '我的实际实验', href: 'https://github.com/Qi18/nanochat/tree/experiment/l20-d24-swanlab-20260806' },
      ],
    },
  },
  { path: ['LLM', 'Train', 'NanoChat', '实验过程'] },
  { path: ['LLM', 'Train', 'NanoChat', '源码阅读'] },
  {
    path: ['LLM', 'Agent RL', 'Search-R1'],
    overview: {
      title: 'Search-R1 项目说明',
      description: 'Search-R1 基于 veRL 训练推理与搜索交错的语言模型。这个子系列记录官方代码结构、检索服务、PPO/GRPO 训练链路以及我们的实际实验。',
      links: [
        { label: '官方项目', href: 'https://github.com/PeterGriffinJin/Search-R1' },
        { label: '我的实际实验', href: 'https://github.com/Qi18/search-r1-lab/tree/experiment/l20-search-r1-20260816' },
      ],
    },
  },
  { path: ['LLM', 'Agent RL', 'Search-R1', '实验过程'] },
  { path: ['LLM', 'Agent RL', 'Search-R1', '源码阅读'] },
];

export function getSeriesDefinition(path: string[]) {
  return seriesDefinitions.find((definition) => (
    definition.path.length === path.length
      && definition.path.every((segment, index) => segment === path[index])
  ));
}

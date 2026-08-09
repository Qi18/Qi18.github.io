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
  {
    path: ['LLM', '实验', 'NanoChat'],
    overview: {
      title: 'NanoChat 项目说明',
      description: 'NanoChat 是 Andrej Karpathy 开源的简洁 LLM 实验框架，面向单机 GPU 节点，覆盖分词、预训练、微调、评测和推理等主要阶段。这个系列沿着真实执行链路记录源码理解与训练实验。',
      links: [
        { label: 'GitHub 项目地址', href: 'https://github.com/karpathy/nanochat' },
        { label: '我的实际实验', href: 'https://github.com/Qi18/nanochat/tree/experiment/l20-d24-swanlab-20260806' },
      ],
    },
  },
  { path: ['LLM', '实验', 'NanoChat', '实验过程'] },
  { path: ['LLM', '实验', 'NanoChat', '源码阅读'] },
];

export function getSeriesDefinition(path: string[]) {
  return seriesDefinitions.find((definition) => (
    definition.path.length === path.length
      && definition.path.every((segment, index) => segment === path[index])
  ));
}

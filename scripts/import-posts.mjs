import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceRoot = process.env.CAREER_SOURCE ?? String.raw`C:\Users\xjiaj\Documents\research\career`;
const outputRoot = path.join(projectRoot, "src", "content", "blog");

const posts = [
  {
    source: "week_01/decoder-only-transformer-dataflow.md",
    slug: "decoder-only-transformer-dataflow",
    title: "Decoder-only Transformer、长上下文与 Agent 后训练",
    description: "沿着一次 token 生成，拆解 Decoder-only Transformer 的前向计算、训练推理、长上下文机制与 Agent 能力形成路径。",
    publishedAt: "2026-09-13",
    tags: ["Transformer", "LLM", "长上下文"],
    accent: "blue",
  },
  {
    source: "week_02/rag-retrieval-systems.md",
    slug: "rag-retrieval-systems",
    title: "RAG 检索系统：索引、融合、重排与 Agentic Retrieval",
    description: "从企业知识库场景出发，梳理解析、索引、混合召回、融合、重排、引用与评测构成的完整证据链。",
    publishedAt: "2026-09-17",
    tags: ["RAG", "检索", "向量数据库", "Agent"],
    accent: "orange",
  },
  {
    source: "week_03/cheat-sheets.md",
    slug: "math-and-machine-learning-cheat-sheets",
    title: "数学与机器学习 Cheat Sheets",
    description: "汇总优化不等式、矩阵分析、概率论、Metropolis–Hastings 与最优传输中的常用定义、公式和推导。",
    publishedAt: "2026-10-04",
    tags: ["数学", "机器学习", "概率论", "优化"],
    accent: "green",
  },
  {
    source: "week_03/llm-fine-tuning-and-preference-optimization.md",
    slug: "llm-fine-tuning-and-preference-optimization",
    title: "大模型微调：从 SFT、RLHF 到 DPO 与 GRPO",
    description: "比较 SFT、RLHF、PPO、DPO 与 GRPO 的数据、目标和工程边界，并建立是否应该微调的决策路径。",
    publishedAt: "2026-10-08",
    tags: ["微调", "RLHF", "DPO", "GRPO"],
    accent: "violet",
  },
  {
    source: "week_04/agent-systems-harness-tools-memory-security.md",
    slug: "agent-architecture-tools-memory-and-reliable-execution",
    title: "Agent 智能体系统：从执行闭环到 Harness 设计",
    description: "从任务执行闭环出发，梳理 Harness 的执行控制、上下文与记忆、工具与协议、安全隔离、验证和恢复。",
    publishedAt: "2026-10-15",
    tags: ["Agent", "Harness", "工具调用", "记忆"],
    accent: "blue",
  },
];

function quote(value) {
  return JSON.stringify(value);
}

function estimateReadingMinutes(markdown) {
  const withoutCode = markdown.replace(/```[\s\S]*?```/g, " ");
  const han = (withoutCode.match(/[\u3400-\u9fff]/g) ?? []).length;
  const latinWords = (withoutCode.match(/[A-Za-z0-9_]+/g) ?? []).length;
  return Math.max(1, Math.ceil(han / 550 + latinWords / 220));
}

await mkdir(outputRoot, { recursive: true });

for (const post of posts) {
  const sourcePath = path.join(sourceRoot, ...post.source.split("/"));
  let body = await readFile(sourcePath, "utf8");
  body = body.replace(/^#\s+.*\r?\n+/, "");
  body = body.replace(
    "(../week_02/rag-retrieval-systems.md)",
    "(/articles/rag-retrieval-systems/)",
  );

  const readingMinutes = estimateReadingMinutes(body);
  const frontmatter = [
    "---",
    `title: ${quote(post.title)}`,
    `description: ${quote(post.description)}`,
    `publishedAt: ${post.publishedAt}`,
    `tags: [${post.tags.map(quote).join(", ")}]`,
    `readingMinutes: ${readingMinutes}`,
    `accent: ${post.accent}`,
    "draft: false",
    "---",
    "",
  ].join("\n");

  await writeFile(path.join(outputRoot, `${post.slug}.md`), `${frontmatter}${body.trim()}\n`, "utf8");
  console.log(`Imported ${post.source} -> ${post.slug}.md (${readingMinutes} min)`);
}

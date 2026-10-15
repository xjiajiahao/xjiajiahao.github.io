import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const SITE = "https://xjiajiahao.github.io";
const BASE = "/xiebro-blog";

/**
 * Astro 只会给自身生成的 URL 加上 base，不会改写 Markdown 正文里的站内绝对链接。
 * 这里补上前缀，避免文章之间的互链在项目页下 404。
 */
function rehypeBaseLinks() {
  const prefix = BASE.replace(/\/+$/, "");
  const visit = (node) => {
    if (
      node.type === "element" &&
      node.tagName === "a" &&
      typeof node.properties?.href === "string"
    ) {
      const href = node.properties.href;
      if (href.startsWith("/") && !href.startsWith("//")) {
        node.properties.href = `${prefix}${href}`;
      }
    }
    node.children?.forEach(visit);
  };
  return (tree) => visit(tree);
}

export default defineConfig({
  site: SITE,
  base: BASE,
  output: "static",
  trailingSlash: "always",
  integrations: [sitemap()],
  markdown: {
    processor: unified({
      gfm: true,
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex, rehypeBaseLinks],
    }),
    syntaxHighlight: "shiki",
    shikiConfig: {
      theme: "github-dark-default",
      wrap: true,
    },
  },
});

import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const SITE = "https://xjiajiahao.github.io";

export default defineConfig({
  site: SITE,
  output: "static",
  trailingSlash: "always",
  integrations: [sitemap()],
  markdown: {
    processor: unified({
      gfm: true,
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
    syntaxHighlight: "shiki",
    shikiConfig: {
      theme: "github-dark-default",
      wrap: true,
    },
  },
});

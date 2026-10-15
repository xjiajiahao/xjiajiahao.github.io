import rss from "@astrojs/rss";
import { getPublishedPosts } from "../lib/posts";
import { withBase } from "../lib/url";

export async function GET(context: { site: URL }) {
  const posts = await getPublishedPosts();
  return rss({
    title: "谢bro的研究笔记",
    description: "记录谢bro的研究随想和见闻。",
    site: new URL(withBase("/"), context.site),
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.publishedAt,
      link: withBase(`/articles/${post.id}/`),
      categories: post.data.tags,
    })),
    customData: "<language>zh-CN</language>",
  });
}

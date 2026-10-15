import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const blog = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/blog" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    publishedAt: z.coerce.date(),
    updatedAt: z.coerce.date().optional(),
    tags: z.array(z.string()),
    readingMinutes: z.number().int().positive(),
    accent: z.enum(["blue", "orange", "green", "violet"]).default("blue"),
    draft: z.boolean().default(false),
  }),
});

export const collections = { blog };

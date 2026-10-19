# 谢bro的研究笔记

基于 Astro 的中文技术博客，发布大模型、RAG、Agent、数学与机器学习研究笔记。

## 本地开发

```bash
npm install
npm run import:posts
npm run dev
```

## 构建

```bash
npm run build
npm run preview
```

## 同步研究笔记

导入脚本默认从 `C:\Users\xjiaj\Documents\research\career` 读取文章。也可以通过环境变量指定其他位置：

```powershell
$env:CAREER_SOURCE = "D:\path\to\career"
npm run import:posts
```

## 部署

推送到 `main` 分支后，GitHub Actions（`.github/workflows/deploy.yml`）会自动构建并部署到 GitHub Pages。

- 线上地址：https://xjiajiahao.github.io/
- 仓库：https://github.com/xjiajiahao/xjiajiahao.github.io

站点部署在 GitHub Pages 根域名，`astro.config.mjs` 中只配置 `site`，不设置子路径 `base`。组件里的站内链接统一使用 `src/lib/url.ts` 的 `withBase()`。

仓库 Settings → Pages 中的 Source 需要设为 **GitHub Actions**。

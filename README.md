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

推送到 `main` 分支后，GitHub Actions 会自动构建并部署到 GitHub Pages。仓库 Settings → Pages 中的 Source 需要设为 **GitHub Actions**。

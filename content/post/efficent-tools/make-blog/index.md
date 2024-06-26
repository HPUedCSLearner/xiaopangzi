---
title: "搭建博客"
description: 
date: 2024-06-26T23:06:50+08:00
image: 
math: 
license: 
hidden: false
comments: true
draft: false
---


### 博客预览



https://hpuedcslearner.github.io/blog/

https://hpuedcslearner.github.io/

https://hpuedcslearner.github.io/xiaopangzi





# 1、hugo



[create a site](https://gohugo.io/getting-started/quick-start/)

```bash
hugo new site quickstart
cd quickstart
git init
git submodule add https://github.com/theNewDynamic/gohugo-theme-ananke.git themes/ananke
echo "theme = 'ananke'" >> hugo.toml
hugo server
```



## Add content 

```bash
hugo new content content/posts/my-first-post.md
```





```bash
hugo server --buildDrafts
hugo server -D
```



[自动构建](https://gohugo.io/hosting-and-deployment/hosting-on-github/)

`.github/workflows/hugo.yaml`



```yaml
# Sample workflow for building and deploying a Hugo site to GitHub Pages
name: Deploy Hugo site to Pages

on:
  # Runs on pushes targeting the default branch
  push:
    branches:
      - main

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# Sets permissions of the GITHUB_TOKEN to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write

# Allow only one concurrent deployment, skipping runs queued between the run in-progress and latest queued.
# However, do NOT cancel in-progress runs as we want to allow these production deployments to complete.
concurrency:
  group: "pages"
  cancel-in-progress: false

# Default to bash
defaults:
  run:
    shell: bash

jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    env:
      HUGO_VERSION: 0.128.0
    steps:
      - name: Install Hugo CLI
        run: |
          wget -O ${{ runner.temp }}/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.deb \
          && sudo dpkg -i ${{ runner.temp }}/hugo.deb          
      - name: Install Dart Sass
        run: sudo snap install dart-sass
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: recursive
          fetch-depth: 0
      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v4
      - name: Install Node.js dependencies
        run: "[[ -f package-lock.json || -f npm-shrinkwrap.json ]] && npm ci || true"
      - name: Build with Hugo
        env:
          # For maximum backward compatibility with Hugo modules
          HUGO_ENVIRONMENT: production
          HUGO_ENV: production
          TZ: America/Los_Angeles
        run: |
          hugo \
            --gc \
            --minify \
            --baseURL "${{ steps.pages.outputs.base_url }}/"          
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./public

  # Deployment job
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```



注意：如果构建失败，这里的Source需要选择`GitHub Actions`



<!-- ![image-20240626225338598](/images/image-20240626225338598.png) -->
![](/image-20240626225338598.png)




### Theme

https://themes.gohugo.io/

https://stack.jimmycai.com/guide/getting-started

https://github.com/CaiJimmy/hugo-theme-stack



```bash
git clone https://github.com/CaiJimmy/hugo-theme-stack/ themes/hugo-theme-stack
git submodule add https://github.com/CaiJimmy/hugo-theme-stack/ themes/hugo-theme-stack
```





#### 一个新的仓库

### …or create a new repository on the command line



```bash
echo "# xiaopangzi" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:HPUedCSLearner/xiaopangzi.git
git push -u origin main
```

### …or push an existing repository from the command line



```bash
git remote add origin git@github.com:HPUedCSLearner/xiaopangzi.git
git branch -M main
git push -u origin main
```



参考链接：

1、 https://gohugo.io/getting-started/quick-start/

2、deploy —— https://gohugo.io/hosting-and-deployment/hosting-on-github/

3、使用 Hugo 和 GitHub Pages 搭建并部署一个静态博客网站   —— https://jaredyam.github.io/posts/build-and-deploy-a-static-blog-website-with-hugo-and-github-pages/

4、如何用 GitHub Pages + Hugo 搭建个人博客 —— https://cuttontail.blog/blog/create-a-wesite-using-github-pages-and-hugo/

5、指定安装版本 —— https://github.com/gohugoio/hugo/releases/tag/v0.127.0











# 2、hexo
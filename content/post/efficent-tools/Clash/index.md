---
title: "Clash"
description: 
date: 2024-06-27T00:01:20+08:00
image: 
math: 
license: 
hidden: false
comments: true
draft: false
---


## 日常使用

```bash
alias clash_start="screen -S clash /home/xxx/clash/clash -d /home/xxx/clash/"
alias clash_stop="pkill clash"

alias proxy_on="export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890"
alias proxy_off="unset http_proxy https_proxy"
```



## github加速

```
$ git config --global http.proxy 'http://127.0.0.1:7890'
$ git config --global https.proxy 'https://127.0.0.1:7890'
```



```
git config  http.proxy 'http://127.0.0.1:7890'
git config  https.proxy 'https://127.0.0.1:7890'
```





```bash
alias clash_start='screen -S clash /home/feng/software/ikuuu/clash -d /home/feng/software/ikuuu/'
alias clash_stop="pkill clash"

alias proxy_on="export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890"
alias proxy_off="unset http_proxy https_proxy"

alias gitproxy="git config  http.proxy 'http://127.0.0.1:7890' && git config  https.proxy 'https://127.0.0.1:7890'"
```



### systemcd clash

`/etc/systemd/syste`







参考链接：



1、**Ubuntu纯命令行走Clash终端代理(Linux同理)** —— https://www.hengy1.top/article/3dadfa74.html

2、clash官方网址 —— https://clashforwindows.org/clash-for-windows-official/

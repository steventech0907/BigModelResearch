#!/bin/bash

# 动态生成密码并设置
read -s -p "请输入密码: " password
echo

# 获取当前用户名
current_user=$(id -un)
# 使用 sudo 命令执行 chsh 命令
echo "$password" | sudo -S chsh -s /usr/bin/fish "$current_user"

# 切换到指定目录
cd /mnt/data1/miniconda3/bin

# 初始化 Conda
./conda init fish

# 启动 Fish Shell
fish
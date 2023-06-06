#!/bin/bash

# 动态生成用户名
read -p "请输入用户名: " username

# 检查用户是否已经存在
if id "$username" >/dev/null 2>&1; then
    echo "用户 $username 已经存在！"
    exit 1
fi

# 动态生成密码并设置
read -s -p "请输入密码: " password
echo

# 动态生成公钥并写入authorized_keys
read -p "请输入公钥: " pubkey

# 创建用户
useradd "$username" || { echo "创建用户 $username 失败！"; exit 1; }

# 添加到"trade"组
if ! getent group trade | grep -q "\b$username\b"; then
    usermod -aG trade "$username" || { echo "添加用户 $username 到 trade 组失败！"; exit 1; }
fi

# 添加到"wheel"组
if ! getent group wheel | grep -q "\b$username\b"; then
    usermod -aG wheel "$username" || { echo "添加用户 $username 到 wheel 组失败！"; exit 1; }
fi

# 设置密码
echo "$username:$password" | chpasswd || { echo "设置密码失败！"; exit 1; }

# 创建.ssh目录并设置权限
mkdir -p "/home/$username/.ssh" && chmod 755 "/home/$username/.ssh" || { echo "创建 .ssh 目录失败！"; exit 1; }

# 写入公钥到authorized_keys
echo "$pubkey" > "/home/$username/.ssh/authorized_keys" && chmod 644 "/home/$username/.ssh/authorized_keys" || { echo "写入公钥到 authorized_keys 失败！"; exit 1; }

echo "用户 $username 创建成功！"
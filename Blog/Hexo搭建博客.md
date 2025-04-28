**lizengsheng用于存放个人简历**

**lzs用于存放个人博客**

# 1.本地环境

```
cd /Users/lzs/Desktop/workspace/Blog/lizengsheng

hexo init

hexo clean 
hexo g -d

hexo s
```

# 2.SSH Key

**将Github改名为lizengsheng**

```
建立仓库
lizengsheng.github.io
lzs.github.io
```

**生成SSH Key**

```
cd /Users/lzs/.ssh/ 
# ssh-keygen -t rsa -C "zengshengli775@gmail.com"
ssh-keygen -t rsa -f /Users/lzs/.ssh/新密钥名称(rsa_blog) -C "zengshengli775@gmail.com"
```

**配置config文件**

```
Host github.com # 前缀名可以任意设置
HostName ssh.github.com
User git
IdentityFile /Users/lzs/.ssh/id_rsa # 自己对应的密钥路径

Host blog # 前缀名可以任意设置
HostName ssh.github.com
User git
IdentityFile /Users/lzs/.ssh/rsa_blog # 自己对应的密钥路径
```

**添加新的 SSH 密钥 到 SSH agent**

```
ssh-add id_rsa # 第一个密钥名称
ssh-add rsa_blog # 第二个密钥名称
```

如果执行以上命令出现错误：`Could not open a connection to your authentication agent.`，那么就需要先执行`ssh-agent bash`，再执行以上命令。

**验证配置**

```
ssh -T git@github.com // 与上面配置的Host 名字对应
ssh -T git@blog // 与上面配置的Host 名字对应
```

**设置local配置**

```

git config user.name "lizengsheng"
git config user.email "zengshengli775@gmail.com"

查看用户名和邮箱地址
git config user.name
git config user.email


git config --list

添加远程仓库的时候，就不能直接使用https的方式了，只能使用ssh方式
git remote add origin git@github.com: home/example.git # home account
git remote add origin git@blog: work/example.git # work account

git remote -v 确认是否连接上

git push origin master
```

# 3.部署到Github

```
cd /Users/lzs/Desktop/workspace/Blog/lizengsheng

安装Git部署插件
npm install hexo-deployer-git --save

Hexo本地文件夹，找到_config.yml文件修改
deploy:
  type: git
  repository: git@blog:lizengsheng/lizengsheng.github.io.git
  branch: master
```

# 4.安装主题
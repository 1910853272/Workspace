## 1.清除global config配置

```shell
git config --global --unset user.name 
git config --global --unset user.email
```

## 2.生成多个SSH密钥

```shell
#bash
# 将email修改为你的第一账户的email，建议为常用的账号,默认生成id_rsa文件
ssh-keygen -q -t rsa -C "1910853272@qq.com" 
# 将email修改为你的第二账户的email
ssh-keygen -q -t rsa -C "zengshengli775@gmail.com" -f ~/.ssh/id_rsa_blog
```

## 3.添加SSH密钥到GitHub账号

分别将id_rsa和id_rsa_blog文件的内容添加到对应的GitHub账号的SSH Keys设置页面

```shell
#bash
pbcopy < id_rsa_.pub
pbcopy < id_rsa_blog.pub
```

## 4.配置SSH config文件

在`~/.ssh/`目录下创建`config`文件

```shell
#bash
touch ~/.ssh/config
```

然后在文件中输入以下内容并保存

```shell
Host github.com
HostName github.com
IdentityFile ~/.ssh/id_rsa
PreferredAuthentications publickey

Host github_b.com 
HostName github.com
IdentityFile ~/.ssh/id_rsa_blog
PreferredAuthentications publickey
```

## 5.把专用密钥添加到高速缓存中

```shell
#bash
ssh-agent bash
ssh-add ~/.ssh/id_rsa
ssh-add ~/.ssh/id_rsa_blog
```

## 6.测试SSH连接

```shell
#bash
ssh -T git@github.com
ssh -T git@github_b.com
```

正常情况下，你会得到如下消息

```shell
Hi xxx! You've successfully authenticated, but GitHub does not provide shell access.
```

## 7.clone and push

```shell
#bash 
git clone git@github.com:username/repo.git
git clone git@github_b.com:username/repo.git
```
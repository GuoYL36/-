
# Linux

- [目录](#)
	- [点(.)的各种意思和用法](#点的各种意思和用法)
	- [查看端口占用情况](#查看端口占用情况)
	- [查看线程数](#查看线程数)
	- [查看内存使用情况](#查看内存使用情况)
	- [进程间数据传输机制](#进程间数据传输机制)
	- [一些变量及符号](#一些变量及符号)
	- [Vim](#Vim)
    - [命令](#命令)

## 点的各种意思和用法
+ **单个点的含义：**
	+ 位置：表示“当前目录”
	+ 作为命令：等同于source

+ **两个点的含义：**
	+ 用于构建序列：
		+ echo {1..10}，打印序列1到10；
		+ echo {1..10..2}，打印序列1到10中的奇数；
		+ mkdir file_{1..10}，创建10个以数字结尾的文件；
		+ echo {a..z}{a..z}，打印从aa到zz的所有字母组合。

----

## 查看端口占用情况
+ lsof –i: 端口号
+ netstat –tunlp | grep 端口号
----

## 查看线程数
+ top –H （不加H就是查看进程）
+ top –Xh
----

## 查看内存使用情况
+ free：快速查看内存使用情况
+ ps：显示各个进程的内存使用情况以及更详细的物理内存使用情况
+ top：显示每个进程的实时使用率
----

## 进程间数据传输机制
+ **间接通信**
	+ 管道：
		+ 子进程从父进程继承文件描述符。一个程序的输出定向为另一个程序的输入（称作管道，管道是父进程为子进程建立的数据传输通道）。管道里的数据是字节流，且是非结构化数据。管道里有一定的buffer。
	+ 消息队列：
		+ 不涉及父子进程。消息队列可以实现多个不相干的进程之间的数据传输。消息队列里的数据可以是结构化数据；消息队列中也有一定的buffer大小。

+ **直接通信**
	+ 共享内存：
		+ 每个地址空间（进程）明确地设置了共享内存段，需要考虑同步互斥的问题。
----

## 一些变量及符号
+ ${}：告诉shell展开花括号里的内容
+ %：告诉shell需要在展开字符串之后从字符串的末尾去掉某些内容

```bash
	a=”Too longgg”   
	echo ${a%gg}  # too long
```
+ \#：告诉shell需要在展开字符串之后从字符串的开头去掉某些内容

```bash
	a=”Hello World!”
	echo Goodbye${a#Hello}   #Goodbye World!
```

## Vim
[Linux学习笔记](https://blog.csdn.net/NOT_GUY/article/details/86726743)

+ **Windows中默认的文件格式是GBK(gb2312)**，而Linux一般是UTF-8

+ 普通模式下，vim编辑器提供了一些命令来编辑缓冲区中的数据
	+ 删除
		+ 当前光标所在位置的n个字符：(n)x
		+ 删除当前光标所在位置的n行：(n)dd
		+ 删除当前光标所在位置的n个单词：(n)dw
		+ 删除当前光标所在位置至行尾的内容：d$ 
		+ 删除当前光标所在行行尾的换行符（拼接行）：J
	+ 撤销
		+ 撤销前一编辑命令：u 
		+ 在当前光标后追加数据：a
		+ 在当前光标所在行行尾追加数据：A
	+ 复制
		+ 复制当前光标所在位置的n行：(n)yy
		+ 复制当前光标所在位置的n个单词：(n)yw
		+ 复制当前光标所在位置至行尾的内容：y$
		+ 复制第26行至50行：命令模式下，:26,50y
	+ 粘贴
		+ 粘贴：p
	+ 查找
		+ 查找：/查找的文本
		+ 向下查找：n
		+ 向上查找：N
	+ 替换
		+ 替换当前行的old为new：:s/old/new
		+ 替换当前行所有old为new：:s/old/new/g
		+ 替换第n行开始到最后一行中每一行的第一个old为new：:n,$s/old/new/
		+ 替换第n行开始到最后一行中每一行所有的old为new：:n,$s/old/new/g
		+ 替换整个文件中的所有old：:%s/old/new/g
		+ 替换整个文件中的所有old，但在每次出现时提示：:%s/old/new/gc
		+ 用char替换当前光标所在位置的单个字符：r  char
		+ 用text覆盖当前光标所在位置的数据，直到按下ESC键：R  text

+ 可视模式: 按ctrl+v进入可视模式
	+ 复制
		+ 移动光标选中需要复制的行，使用y复制选中块到缓冲区
	+ 剪切
		+ 移动光标选中需要复制的行，使用d剪切选中块到缓冲区
	+ 粘贴
		+ 将光标移动到粘贴的位置，并使用p进行粘贴
    + 多行注释
        + 选中需要注释的行
        + 按大写字母I，然后插入注释符
        + 按2次ESC键即可全部注释
    + 取消多行注释
        + 选中多个注释符
        + 按d键即可全部取消
    

+ 在vim中查看文件编码：:set fileencoding		

+ 高级技巧
	+ 比较多个文件：vim -d file1 file2
	+ 打开多个文件：vim file1 file2 file3
		+ 启动vim后只有一个窗口显示的是file1，可以在末行模式中输入ls查看到打开的三个文件，也可以在末行模式中输入b<num>来显示另一个文件，例如可以用:b 2将file2显示出来，可以用:b3将file3显示出来。
	+ 拆分和切换窗口
		+ 可以在末行模式中输入sp或vs来实现对窗口的水平或垂直拆分，这样我们就可以同时打开多个编辑窗口，通过按两次Ctrl+w就可以实现编辑窗口的切换，在一个窗口中执行退出操作只会关闭对应的窗口，其他的窗口继续保留。
		
## 下载及传送文件
+ 从给定的url地址下载相应地文件
    + wget url
    
+ linux之间互相传递文件
    + scp -r 文件名.后缀 目标服务器账号@目标服务器地址: 目标路径


## 命令
+ 删除指定日期之前的文件
    + find /dir0 -mtime +36 -type f -name "filename[12]" -exec rm -rf {} \;
        + /dir0: 设置查找的目录
        + -mtime +36：设置时间为36天前
        + -type f：设置查找的类型为文件
        + -name "filename[12]"：文件名称为filename1或filename2
        + -exec rm -rf：强制删除文件，包括目录.(-exec必须由一个;结束)
        + {} \;：固定写法，或'{}' \;
        + **注意: 当文件数量较多时，可能会报错：参数太多(exec导致的)**
    + find /dir0 -ctime +36 -type f | xargs rm -rf
        + xargs命令会分批次的处理结果

+ sudo和su
    + su: 切换了root身份，但shell环境仍然是普通用户的Shell；此时pwd，工作目录还是普通用户的工作目录.
    + su -：连用户和Shell环境一起切换成root身份，只有切换了Shell环境才不会出现PATH环境变量错误，此时pwd，工作目录变成root的工作目录.
    
+ 让程序在后台运行和一直运行
    + 让程序在后台自动运行
        + 在程序结尾加&：/usr/local/mysql/bin/mysqld_safe --user=mysql & 
        + 终端关闭，进程结束

    + 退出账户/关闭终端后继续运行相应的进程(守护进程)
        + nohup /usr/local/mysql/bin/mysqld_safe --user=mysql &
        + 带&：即使终端关闭或者电脑死机程序依然运行，前提是程序递交到服务器上。
        + 把标准输出(STDOUT)和标准错误(STDERR)结果输出到nohup.txt文件。

## 退出xshell后linux程序后台运行
+ nohup python a.py &
+ 创建tmux会话


## tmux使用
+ 安装tmux：yum install tmux
+ 进入tmux窗口：tmux
    + 接入指定会话：tmux attach -t 会话编号/会话名
+ 退出tmux窗口：ctrl+d 或 exit
+ 唤醒快捷键的前缀建：ctrl+b
+ 新建会话：tmux new -s <session-name>
+ 分离会话：ctrl+b+d 或 tmux detach
+ 杀死会话：tmux kill-session -t 会话编号/会话名
+ 切换会话：tmux switch -t 会话编号/会话名
+ 重命名会话： tmux rename-session -t 0 <new-name> 或 Ctrl+b $
+ 列出所有会话：Ctrl+b s
+ 划分窗格：tmux split-window 或 tmux split-window -h
+ 切换窗格
    + 光标切换到上方窗格：tmux select-pane -U
    + 光标切换到下方窗格：tmux select-pane -D
    + 光标切换到左边窗格：tmux select-pane -L
    + 光标切换到右边窗格：tmux select-pane -R

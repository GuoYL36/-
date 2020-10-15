# docker三元素：镜像image、容器container、仓库repo
+ 1、容器是镜像的实例化，有镜像才有容器；
+ 2、镜像类似于洋葱，一层套一层的文件构成；


# docker镜像是什么？
## 定义：轻量级、可执行、独立的软件包，用来打包软件运行环境和基于运行环境开发的软件，包含代码、运行时、库、环境变量和配置文件。
+ 1、UnionFS：联合文件系统，docker镜像的基础，支持对文件系统的修改作为一次提交来一层层的叠加。一次同时加载多个文件系统，但从外面看只能看到一个文件系统，联合加载会把各层文件系统叠加起来，这样最终的文件系统会包含所有底层的文件和目录。
+ 2、加载原理：docker镜像最底层是bootfs,包含boot加载器和内核，当boot加载完成，整个内核都在内存中后，内存的使用权由bootfs转交给内核，此时系统会卸载bootfs。在bootfs之上还有rootfs，包含linux系统中标准目录和文件。（这也是docker镜像文件小于虚拟机文件的原因）
+ 3、分层的镜像：docker pull的时候能看到很多镜像文件的ID
+ 4、为啥采用分层结构：共享资源——下载一次之后会有缓存，下次下载就会很快。
+ 5、特点：只读的；当容器启动时，一个新的可写层被加载到镜像的顶部，这一层叫作“容器层”，“容器层”之下的叫“镜像层”。


# 容器数据卷
+ 1、why：
	+ 1)对数据的要求希望是持久化的；
	+ 2)容器之间希望有可能共享数据；
	+ 3)docker容器产生的数据如果不通过docker commit生成新的镜像，当容器删除后，数据就没有了。为了在docker中保存数据，使用卷。
+ 2、what：类似Redis中rdb、aof文件
+ 3、do：容器的持久化+容器间继承+共享数据
	+ 特点：数据卷可在容器之间共享或重用数据；卷中的更改可直接生效；数据卷中的更改不会包含在镜像的更新中；数据卷的生命周期一直持续到没有容器使用它为止。
+ 4、数据卷：容器内添加。
	+ 1）直接命令添加，两个目录中的数据可以共享：
		+ docker run -it -v /宿主机绝对目录路径:/容器目录路径 镜像名
		+ docker run -it -v /宿主机绝对目录路径:/容器目录路径:ro 镜像名      //ro表示容器内目录只读
		+ 注意：docker挂载主机目录docker访问出现cannot open directory.:Permission denied，解决方法为：在挂载目录后(即镜像名前)多加一个--privileged=true参数

	+ 2）dockerFile中添加容器数据卷（和上面命令的目的一样，不过这是在文件中添加数据卷并生成新的容器）：
		+ a.宿主机中体创建一个文件夹；
		+ b.在dockerfile中用volume命令给镜像添加一个或多个数据卷：VOLUME ["/宿主机绝对目录路径","/容器目录路径"]
		+ c.file脚本构建，例如：
			+ FROM centos
			+ VOLUME ["/宿主机绝对目录路径","/容器目录路径"]			
			+ CMD echo "finished======suceessed"
			+ CMD /bin/bash		
		+ e.生成新镜像文件：docker build -f file脚本 -t 新的镜像名:version .    // .表示当前目录下,如果file脚本名字为DockerFile，则“-f file脚本”可省略。	
+ 5、数据卷容器
	+ 1）定义：命名的容器挂载数据卷，其它容器通过挂载这个父容器实现数据共享，挂载数据卷的容器称之为数据卷容器。即：活动硬盘上挂活动硬盘，实现容器间数据传递共享。
	+ 2）dc02容器数据卷挂载到dc01容器上：docker run -it --name dc02 --volumes-from dc01 centos 
	+ 3）容器之间配置信息的传递，数据卷的生命周期一直持续到没有容器使用它为止。

+ 6、dockerfile解析
	+ 1）构建步骤：手动编写一个dockerfile文件，必须符合file的规范；根据这个文件后，直接docker build命令执行获得一个自定义的镜像；run执行镜像文件。
	+ 2）dockerfile是用来构建docker镜像的构建文件，是由一系列参数和命令构成的脚本。
	+ 3）基础知识：
		+ a.scratch：所有镜像文件的祖先类；
		+ b.每条保留字指令都必须为大写字母且后面要跟随至少一个参数；
		+ c.指令按照从上到下顺序执行；
		+ d.#表示注释；
		+ e.每条指令都会创建一个新的镜像层，并对镜像进行提交。
	+ 4）保留字指令
		+ a.FROM：基础镜像，当前新镜像是基于哪个镜像
		+ b.MAINTAINER：镜像维护者的姓名和邮箱
		+ c.RUN：构建容器运行的命令，比如安装一些基础镜像没有的软件包
		+ d.EXPOSE：该镜像实例化成容器后，对外暴露的服务端口
		+ e.WORKDIR：创建容器后，终端默认进入的工作目录
		+ f.ENV：设置环境变量
		+ g.ADD：拷贝加解压
		+ h.COPY：拷贝
		+ i.VOLUME：容器数据卷，用于保存数据和数据持久化
		+ j.CMD：指定一个容器运行时要运行的命令，dockerfile中可以有多个命令，但只有最后一个生效，CMD会被docker run之后的参数替换。
		+ k.ENTRYPOINT：指定一个容器运行时要运行的命令，但是多个命令都会生效。
		+ l.ONBUILD：当构建一个被继承的dockerfile时运行命令，父镜像在被子继承后父镜像的onbuild被触发。

# tomcat完整构建dockerfile
+ 1）新建一个目录：mkdir -p /mydocker/tomcat9
+ 2）在上述目录下touch c.txt
+ 3）将jdk和tomcat安装的压缩包(.tar.gz)拷贝进目录/mydocker/tomcat9
+ 4）在/mydocker/tomcat9目录下新建dockerfile文件
	'''shell
    FROM centos
	MAINTAINER  XXX<XXX@126.com>
	#把宿主机当前上下文的c.txt拷贝到容器/usr/local路径下
	COPY c.txt /usr/local/cincontainer.txt
	#把java与tomcat添加到容器中
	ADD jdk-8u171-linux-x64.tar.gz /usr/local
	ADD apache-tomcat-9.0.8.tar.gz /usr/local
	#安装vim编辑器
	RUN yum -y install vim
	#设置工作访问时候的WORKDIR路径，进入默认的工作目录
	ENV MYPATH /usr/local
	WORKDIR $MYPATH
	#配置java与tomcat环境变量
	ENV JAVA_HOME /usr/local/jdk1.8.0_171
	ENV CLASSPATH $JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
	ENV CATALINA_HOME /usr/local/apache-tomcat-9.0.8
	ENV CATALINA_BASE /usr/local/apache-tomcat-9.0.8
	ENV PATH $PATH:$JAVA_HOME/bin:$CATALINA_HOME/lib:$CATALINA_HOME/bin
	#容器运行时监听的端口
	EXPOSE 8080
	#启动时运行tomcat
	CMD /usr/local/apache-tomcat-9.0.8/bin/startup.sh && tail -F /usr/local/apache-tomcat-9.0.8/bin/log/catalina.out
    '''
+ 5）构建
+ 6）run


# 7、Docker常用安装
+ 1）安装步骤
	+ a.docker hub上查找
+ 2）安装tomcat
	

# 帮助命令
+ docker version：查看版本
+ docker info：信息描述
+ docker --help：帮助信息

# 镜像命令
+ service docker start：启动docker
+ service docker stop：停止docker
+ service docker restart：重启docker
+ docker container --help：列出docker cli命令
+ docker images ls：列出本地主机上的镜像
	+ -a：all，列出所有镜像(包含中间层)
	+ -q：列出镜像的id
	+ --digests：显示摘要的备注信息
	+ --no-trunc：显示完整的镜像信息
+ docker container ls：列出本地主机上的容器
+ docker search 某个镜像的名字 ：搜索某个镜像
	+ -s 数字：STARS，列出收藏数不小于设定值的镜像
	+ --no-trunc：显示完整描述
+ docker pull 某个镜像的名字:version ：拉取某个版本的镜像,缺省为latest
+ docker rmi 某个镜像的名字ID：删除某个镜像
	+ -f：force，强制删除，例子：docker rmi -f $(docker images -qa)，批量删除。
+ docker run 镜像文件：执行镜像
+ docker commit -m="镜像描述" -a=作者 容器ID 要创建的目标镜像名:[标签名]
+ docker push  //参考推送到阿里云上的代码


# 容器命令
+ 新建并启动容器，进入以镜像实例化的容器中：docker run [options] IMAGE [command][arguments]
	+ options:
		+ --name="容器新名字"：为容器指定一个新名字
		+ -d：后台运行容器，并返回容器ID，也即启动守护式容器，但ps不显示，说明“docker容器后台运行，必须有前台进程，如果没有前台进程，则后台进程直接会被杀死。”		
		+ -i：以交互模式运行容器，通常与-t同时使用
		+ -t：为容器重新分配一个伪输入终端，通常与-i同时使用
		+ -P：随机端口映射
		+ -p：指定端口映射，4种格式：
			+ ip:hostPort:containerPort
			+ ip::containerPort
			+ hostPort:containerPort(主机端口:容器端口)
			+ containerPort
    + command：所有linux的命令，默认/bin/bash
+ 查看容器id和镜像id对应的方法：在本地shell中(非容器)运行docker ps
+ 列出docker中当前所有正在运行的容器：docker ps [options]
	+ options：
		+ -a：列出当前所有正在运行的容器+历史运行过的
		+ -l：显示最近创建的容器
		+ -n 数字：显示最近n个创建的容器
		+ -q：静默模式，只显示容器编号
		+ --no-trunc：不截断输出
+ 退出容器：exit(停止并退出)、ctrl+P+Q(停止但不退出)
+ 启动容器：docker start 容器名或容器ID
+ 重启容器：docker restart 容器名或容器ID
+ 停止容器：docker stop 容器名或容器ID
+ 强制停止容器：docker kill 容器名或容器ID
+ 删除已停止容器：docker rm 容器名或容器ID
	+ 批量删除：1）docker rm -f $(docker ps -a -q)；2）docker ps -a -q | xargs docker rm
+ 查看容器日志：docker logs -f -t --tail 容器ID
	+ -t：加入时间戳
	+ -f：跟随最新的日志打印
	+ --tail 数字：显示最后多少条
+ 查看容器内运行的进程：docker top 容器名或ID
+ 查看容器内部细节：docker inspect 容器ID
+ 进入正在运行的容器并以命令行交互：docker attach 容器ID
+ 不进入正在运行的容器但以命令行交互：docker exec -t 容器ID shell命令 
+ 从容器内拷贝文件到主机上(在宿主机上操作)：docker cp 容器ID:文件路径 宿主机路径
 


mysql端口：3306
tomcat端口：8080

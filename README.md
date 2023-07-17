# openapi-testcase

## 用户场景
### 创建单机任务（可以查看监控，任务状态，查看tensorboard，了解任务运行过程中的指标）
a) 编写一个任务，enable tensorboard开启训练过程监控，提交后可以看到任务运行、查看任务的资源利用情况   
 目前使用项目来展示：
 ```
  git clone https://github.com/microsoft/pai.git 
  cd pai 
  git reset --hard dd08930431d05ed490cf7ceeecd262e473c187cd 
  cd docs/user/samples/ 
  python minist_tensorboard.py --data_dir ./data --log_dir /mnt/tensorboard
 ```     
b) enable tensorboard后，可查看训练过程的关键指标   
### 创建分布式任务
a) 编写一个分布式任务，并行运行多个训练任务，分别创建单机多卡任务，多机多卡任务      
 目前使用https://github.com/microsoft/pai/blob/pai-for-edu/contrib/edu-examples/pytorch_cifar10/yaml/Resnet18_4gpu.yaml    
 进行单机多卡训练，训练轮数改小后可以完成，需要将调整后代码和训练数据放到镜像中进行优化      
 **注**：nvidia-smi 可以查看当前节点上面的gpu和进程使用情况   
b) 编写一个分布式任务，任务包含多个角色，支持同时运行不同类型的训练任务，可以查看任务的资源利用情况，以及通过tensorboard查看训练过程的关键指标   
c) 编写一个分布式任务，任务包含多个角色，通过配置指定交互的ip和端口，可以支持角色训练过程的交互   
  https://github.com/pkunight/openpai-testcase/blob/main/models/MNIST/MNIST-DDP-nccl.yaml   
  基于PAI平台的多机多卡任务在测试环境可以执行了.    
  周望做了一个pytorch+cuda+nccl的环境, 基于DDP分布式的MNIST任务, 2机器+3GPU卡.   
镜像和配置我都放到github了: https://github.com/pkunight/openpai-testcase   
当前存在一个问题, 我们的PAI平台没有按照文档写的那样把任务分配的IP和端口写入到环境变量, 导致测试的时候只能人工写入, 机器多了就没法实现了. 这个还需要看看我们部署的PAI平台的源码是为啥.   
### 通过挂载存储，使用本地数据进行预训练
a) 搭建本地共享存储，如nfs服务   
b) 安装nfs storageclass与provisioner等k8s套件   
c) 以用户组为单位，分配共享存储   
d) 创建任务，选择所分配的共享存储，可以使用该共享存储读取预训练数据   
### 通过使用vscode，管理和使用训练平台（创建多种训练任务）
a) 安装vscode的插件    
   https://github.com/microsoft/openpaivscode/releases/download/0.3.2/pai-vscode-0.3.2.vsix   
b) 配置平台链接   
c) 查看当前运行任务   
d) 通过提交训练任务文件来创建训练任务   
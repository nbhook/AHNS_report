
# 项目介绍
本项目完成论文 ***Adaptive Hardness Negative Sampling for Collaborative Filtering*** 的复现。基于ReChorus推荐系统框架(https://github.com/THUwangcy/ReChorus)完成。  
 本项目在ReChorus框架中增加了使用AHNS采样方法的LightGCN和NeuMF模型用以完成对比试验。

# 环境依赖
环境依赖均与ReChorus框架的环境依赖相同。

    python==3.10.4  
    torch==1.12.1  
    cudatoolkit==10.2.89  
    numpy==1.22.3  
    ipython==8.10.0  
    jupyter==1.0.0  
    tqdm==4.66.1  
    pandas==1.4.4  
    scikit-learn==1.1.3  
    scipy==1.7.3  
    pickle  
    yaml  


# 目录结构描述
    ├── ReadMe.md           // 帮助文档
    
    ├── ReChorus-master             // 基于ReChorus框架的代码
    
    │   ├── data     // 包含实验所用数据集
    
    │       ├── Grocery_and_Gourment_Food

    │       ├── MIND_Large

    │       ├── MovieLens_1M
    
    │   ├── docs             // 包含ReChorus框架使用细节说明
    
    │   ├── src     // 包含基于ReChorus框架的TOP-K推荐实验代码
    
    │       ├── helpers //推荐测试实现代码

    │       ├── models  //推荐模型，包括基于AHNS方法修改的模型

    │       ├── utils   //工具包

    │   └── README.md   //针对ReChorus框架的帮助文档

    │   └── requirements.txt    //环境配置文件
    
    └── readme.md                // 帮助文档


 
# 使用说明
执行命令与ReChorus框架已经设计的命令行相同。  
如运行使用了AHNS负采样的LightGCN模型：  

    cd ./ReChorus-master/src
    python main.py --model_name AHNSLightGCN --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-8 --dataset Grocery_and_Gourmet_Food --path /data --dropout 0.3 --p -2 --alpha 1 --beta 0.1  
其他执行信息可参考./ReChorus-master目录下ReChorus框架的文档。

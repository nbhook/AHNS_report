# 项目介绍  
这是一个在ReChorus推荐系统框架*https://github.com/THUwangcy/ReChorus.*复现了论文*Adaptive Hardness Negative Sampling for Collaborative Filtering*的创新方法。  
在其中增加了使用了AHNS采样方法的LightGCN和NeuMF模型
# 环境依赖  
环境依赖均与ReChorus框架的环境依赖相同
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
# 执行命令  
执行命令与ReChorus框架已经设计的命令行相同。  
如运行使用了AHNS负采样的LightGCN模型：  
python main.py --model_name AHNSLightGCN --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-8 --dataset Grocery_and_Gourmet_Food --path /data --dropout 0.3 --p -2 --alpha 1 --beta 0.1  

其他详细信息参考ReChorus框架的文档。
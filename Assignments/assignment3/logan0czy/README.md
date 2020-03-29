# Assignment3 question answer  
## 1. Machine Learning and Neural Networks  
a) Adam Optimizer    
momentum部分使得每一次的梯度更新主要是在上一时刻之前的累计梯度的基础上偏向当前梯度一点，这样避免了到达一个局部的陡峭点时计算出的梯度过大而使得更新步幅太大而造成震荡。

adagrad部分beta2越小每一次计算的梯度越大，分母上的累计梯度平方和其实是估计函数对各变量的二阶导数，表示的是函数在各变量维度上平均的变化快慢，变化快的分母值大，计算出的更新幅度小，反之易得。用这一方法同样能避免局部梯度值陡增陡降带来的震荡。  

b) Dropout   
gamma = 1 / (1-p_drop)  

在训练时使用dropout可以打破隐藏层各特征之间的依赖关系，使得它们独立的抽取特征，同时dropout使每一次训练都是用不同的子模型进行预测，最后在evaluation阶段保留所有节点相当于是将各个模型进行了融合，综合了每一个子模型的判断。  
## 2. Neural Transition-based Dependency Parsing  
b) 2*n+1 steps  
each word corresponds to "shift" and "left/right arc" transitions, plus "left/right arc" to 'root' finally.  

f)
i.  
**Error type**: Verb Phrase Attachment Error  
**Incorrect Dependency**: wedding-->fearing  
**Correct Dependency**: heading-->fearing  

ii.   
**Error type**: Coordination Attachment Error  
**Incorrect Dependency**: makes-->rescue  
**Correct Dependency**: rush-->rescue  

iii.  
**Error type**: Prepositional Phrase Attachment Error  
**Incorrect Dependency**: named-->Midland  
**Correct Dependency**: Joe-->Midland  

iv.  
**Error type**: Modifier Attachment Error  
**Incorrect Dependency**: most<--elements  
**Correct Dependency**: most<--crucial

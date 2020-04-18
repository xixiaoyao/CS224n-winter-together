# NMT Assignment  
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository  
(作业都是个人的理解，不一定对。。。)  
## 1.Neural Machine Translation with RNNs  
(g)  
mask的作用是为了标记每一条padding后的句子padding的地方，便于将句子传递给网络进行计算时让网络知道每一条句子应该计算的合适大小；mask的重要性在于处理时避免将`<pad>`符号作为句中字符进行处理而原句的本来表征产生不必要的影响。  
(j)  
*dot attention to multiplicative attention*:  
advantage:计算简单，不需要训练额外的参数  
disadvantage:标称能力差，可能提取不出有效的信息  
*multiplicative attention to additative attention*:  
advantage:计算更简单，相同的权重维度下可以达到和additative attention接近的性能，并且乘性变换使得反向传播过程更容易  
disadvantage:表达能力更弱，而且性能没有additative attention稳定，当维度变高时，可能效果反而变差。  
## 2.Analyzing NMT systems  
(a)  
i.possible reason:原句第一个词的本意直译过来了，而不是根据文本意思改变对第一个词的解释 possible way:增加翻译上下文，或者增加隐藏层的大小来加强信息的存储。  
ii.possible reason:原文西班牙语的语序和英语的惯用语序不同，西班牙语把修饰词放在了后面 possible way:attention或许用additative attention机制能增强一点性能。  
iii.possible reason:出现了词典以外的词汇 possible ways:扩充词典，或者有的论文中介绍的把phrase-level MT和character-level MT相结合。  
iv.possible reason:把原句的词直译为了have possible ways:用表达能力更好的embedding vector，并在训练中对它进行更新。  
v.possible reason:词语中含有bias，把teacher默认为woman possible ways:预训练embedding，语料中尽量包含多种风格或来源来消除其中隐含的bias。  
vi.possible reason:可能两个语言的计量方式不相同  possible way:增加语料  
(c)  
i.BLEU score of c1: 0.548  BLEU score of c2:0.447  
虽然第一个得分高，但是第二个的翻译更好  
ii.score of c1: 0.548  score of c2:0.316  
iii.如果使用单一的翻译参照的话，由于每一条参照各自的翻译风格不同，用的词汇也不相同，这样会给BLEU的评判带来比较大的variance  
iv.  
advantages:计算简便快捷，便于算法快速迭代；评价标准单一客观，不受语言类型、风格等的影响  
disadvantages:与真实的翻译水平的评价标准还是有一定距离；BLEU得分低的有可能是翻译的还不错的。
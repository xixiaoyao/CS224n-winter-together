# Assignment 5 written questions  
## Character-based convolutional encoder for NMT  
### Model description and written questions  
(a) 对于卷积神经网络也是一样的。卷积核是作用于输入序列的固定长度，所以无论输入序列长度有多长区别只在于进行卷积的次数  
(b) 如果需要至少一个输出的话，左右两侧padding各自的大小为：(kernel_size - (m_model+2))/2；如果是要kernel的每一个列向量权重都可以作用于输入序列的每一个字符上的话，左右两侧padding的大小为kernel_size-1。  
(c) 除了能减缓梯度消失的问题，同时也使得网络自适应地选择向后传递的信息；b_gate的初始化应该尽量使sigmoid函数初始输出为1，即尽量保留x_proj的值，所以初始化应该为positive的。  
(d) 优点1：更好的并行化；优点2：每一个step的representation都能够接触到全局的信息。  
(f)自己编写了sanity_check以后发现在模型构建以及其他各个部分的过程中，使用简单的程序来进行检查是十分必要的，比如排查输入输出维度是否和预期的匹配，网络中各个部分有哪些参数，对网络的计算过程有更清晰的认识。  
## Analyzing NMT Systems  
(a) 找到的字典中的词——'traducir':4603; 'traduzco':40991; 'traduce':7931; '不存在的——'traduces', 'traduzca', 'traduzcas'  
why bad:很多没有出现在词典中，但是和词典中一些词形式相近的词在翻译时作为`<UNK>`处理会极大影响对原句的语义表征。  
this model's solution:使用word-character based model的好处是不受提供的词典大小的限制，在出现词典外的单词时通过character model能够捕获到与词典内形式相近的词的意思  
(b)  
i.`word2vec all`--nearest words for each item  
* `financial`: economic, business, markets, market, money  
* `neuron`: neurons, dendrites, cerebellum, nerve, excitatory  
* `Francisco`: san, jose, diego, california, los  
* `naturally`: occurring, easily, natural, humans, therefore  
* `expectation`: operator, assumption, consequence, otherwise, implies  

ii. `character-base`--nearest words for same items  
* `financial`: vertical, informal, physical, cultural, electrical  
* `neuron`: Newton, George, NBA, Delhi, golden  
* `Francisco`: France, platform, tissue, Foundation, microphone  
* `naturally`: practically, typically, significantly, mentally, gradually  
* `expectation`: exception, indication, integration, separation, expected  

iii.  
Word2Vec是语义相似性，CharCNN是形式上的相似。从各自的模型来看，Word2Vec是基于上下文训练得出word embedding，所以能够很好的反应各词之间的语义信息；而CharCNN对字符向量运用卷积，因为单一的字符是没有确切的意义的，所以卷积得到的结果很可能就是根据词的组成字符的相似性得出的。  

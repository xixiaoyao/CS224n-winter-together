### 1 Machine Learning & Neural Networks (8 points)
#### (a) Adam Optimizer
1. Each update will be mostly the same as the previous one, only a $1-\beta_1$ propotion of $m$ receives an update at each step, so the updates won't vary as much. This will help us maintain a smaller variance and potentially make it faster to reach a local optimum. One intuition is that $m$ will help prevent the model parameters from bouncing 
around as much when moving towards alocal optimum. Another is that doing the rolling average is a bit like computing 
the gradients over a larger minibatch, so each update will be close to the true gradient over the whole dataset
(i.e., lower variance means each gradient eastimates is closer to the mean).

2. The parameters with the smallest gradients will get the larger updates. This means that are at a place where the
loss with respect to them is pretty flat will get larger updates, helping them move off the flat areas.


#### (b) Dropout
1. What must $\gamma$ equal in terms of $p_{\text{drop}}$? Briefly justify your answer. <br>
   $$
   E_{p_{drop}}[h_{drop}]_i=p_{drop}*0+(1-d_{drop})*\gamma*h_i\\
   \gamma=\frac{1}{1-p_{drop}}
   $$
   <br>

   这个 $\gamma$ 应该是为了保持在 $dropout$ 之后激活值总和不变，减少信息损失
   
2. Dropout is used for prevent overfitting. If we apply dropout during evaluation, neurons will be disabled randomly thus network will have different output every activation. This undermines consistency.

### 2 Neural Transition-Based Dependency Parsing (42 points)

1. parsing the sentence *"I parsed this sentence correctly"*

   |             Stack              |        Buffer         |   New dependency    | Transtition |
   | :----------------------------: | :-------------------: | :-----------------: | :---------: |
   |             .....              |         .....         |        .....        |    .....    |
   |      [ROOT, parsed, this]      | [sentence, correctly] |                     |    SHIFT    |
   | [ROOT, parsed, this, sentence] |      [correctly]      |                     |    SHIFT    |
   |    [ROOT, parsed, sentence]    |      [correctly]      |  sentence -> this   |  LEFT-ARC   |
   |         [ROOT, parsed]         |      [correctly]      | parsed -> sentence  |  RIGHT-ARC  |
   |   [ROOT, parsed, correctly]    |          []           |                     |    SHIFT    |
   |         [ROOT, parsed]         |          []           | parsed -> correctly |  RIGHT-ARC  |
   |             [ROOT]             |          []           |   ROOT -> parsed    |  RIGHT-ARC  |

2. *2\*n* steps，一个词进去出来共两个操作 

3. f 题

   1. 
      * **Error Type**: Verb Phrase Attachment Error
      * **Incorrect dependency**: wedding -> fearing
      * **Correct dependency**: fearing -> I
   2. 
      * **Error Type**: Coordination Attachment Error
      * **Incorrect dependency**: makes-> rescue
      * **Correct dependency**: rush -> rescue
   3. 
      - **Error Type**: Prepositional Phrase Attachment Error 
      - **Incorrect dependency**: named -> Midland
      - **Correct dependency**: guy -> Midland
   4. 
      * **Error Type**: Modifier Attachment Error   
      * **Incorrect dependency**: elements -> most  
      * **Correct dependency**: crucial -> most



参考：https://github.com/coderfirefly/cs224n-works/blob/master/a3/Assignment3.ipynb
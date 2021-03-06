# Persian Sentiment Analysis 
A trained model to predict sentiment class of a given Persian text.

# Installation

```bash
pip3 install persian_sa

````

# Read More:
To find about preprocessing and feature engineering, and how the model predicts visit [arXiv](https://arxiv.org/abs/2101.08087).

  
# Usage:

## Running the source code:
- To run the program, use python3 persian_sa.py
- Next you will be prompted to give a Persian text as input.
- To exit the program write ```exit``` on terminal.

 ```bash
 
MOHAMMADs:persian_sa mohammadkasra$ python3 persian_sa.py 


This app uses ML to predict setntiment (e.g., Positive or Negative)
of a given Persian text. Toexit  the app write  'exit' in terminal.


Input: زیاد در خاطرات دیگران ورود نکنید، چرا که در خاطرات هر شخص رازهایی وجود دارد که حتی می ترسد آن ها را برای خودش آشکار کند!
... Negative!


Input: زندگی همچون یک آینه است زمانی که در آن لبخند بزنیم شگفت انگیزترین نتایج را به دست خواهیم آورد
... Positive!


Input: exit
...  exit: 0
MOHAMMADs:persian_sa mohammadkasra$ 
        
```

## Running after Pip install

```python

>>> from persian_sa import persian_sa
>>> 
>>> persian_sa.predict_sentiment('می تواند به همین دلیل از آن متنفر باشد')
'Negative!'
>>> # Or you can predict the class number; if you set "return_class_label = True"
>>> persian_sa.predict_sentiment('می تواند به همین دلیل از آن متنفر باشد', return_class_label = True)
0
>>> persian_sa.predict_sentiment('اجرای آنها شادی مطلق است')
'Positive!'
>>> persian_sa.predict_sentiment('اجرای آنها شادی مطلق است', return_class_label = True)
1
>>>

```

# Wikipedia Word Embedding
Based on https://github.com/liorshk/wordembedding-hebrew

#### Note: This code works for Hebrew, but it should work on any other language as well.

1. Install dependencies using `make install`.

2. Download Hebrew dataset from wikipedia:
   1. Go to [wikimedia dumps](https://dumps.wikimedia.org/hewiki/latest/).
   2. Download `hewiki-latest-pages-articles.xml.bz2`.
  
   In linux this can be easily done using
   ```
   wget https://dumps.wikimedia.org/hewiki/latest/hewiki-latest-pages-articles.xml.bz2
   ```

3. Create your text corpus.
   
   `python create_corpus.py`.<br>
   TODO add CLI interface, arg parse, add usage here.

4. Train the model:
   
   TODO add CLI interface, arg parse, add usage here.


5. Play with your trained model using `playground.ipynb`.


### TODO
* לבדוק את השוני בין המודלים - לקחת מילה, לבדוק מרחק ל-1000 מילים רנדומליות. איך נראית ההתפלגות? מה ההבדל בין המודלים השונים?
* לחלק מילים למחלקות שקילות (האוטו, באוטו, מהאוטו => אוטו).
  * נאיבי: לקחת כל מילה, לבדוק אם מורידים את הפריקס האם התוצאה קיימת במאגר. אם כן - אותה מחלקה, להחליף במילה המייצגת.
  * יותר טוב: לבדוק אם המילה שבודקים קרובה למילה בלי הפרפיקס. אם קרוב (מאוד) - אותה מחלקה, להחליף במילה המייצגת.
  * יותר טוב: לאמן באיטרציות, בכל שלב להסתכל על המודל שאומן לפני כן עד שלא מחליפים (הרבה) מילים חדשות.
  * אפשר להכניס גם סיומות (הטיות ריבוי, מינים, זמנים).
  * חשוב: לשמור בתהליך את מילון מחלקות השקילות (מילה מקורית -> נציגת מחלקה), מכיוון שבמודל הסופי יהיו חסרות הרבה מילים, עדיין נצטרך לדעת לקבל מילה עם הטיה ולהגיד מה המחלקה שלה.
  * לבדוק כמה התגשויות יש (כמה פעמים מילה אחת אפשר לפרש בשני הקשרים שונים).
  * עוד אפשרות: לא נסנן מילה שקרובה לשתי מחלקות.
  * שאלה: איך בוחרים נציג מחלקת שקילות? המילה הקצרה ביותר? המילה שבאמצע? אולי לא משנה בכלל אם אנחנו יודעים לתרגם.

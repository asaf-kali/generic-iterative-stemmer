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

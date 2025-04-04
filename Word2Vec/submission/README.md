# Word2Vec 


This contains word2vec algorithm to create vector embeddings for any dataset and an FFNN classifier to classify hate, humor and sarcasm. 


<!-- TREEVIEW START -->
```bash
├── submission/
│   ├── data/
│   │   ├── hate/
│   │   ├── humor/
│   │   └── sarcasm/
│   ├── hate/
│   │   ├── embeddings/
│   │   └── model/
│   ├── humor/
│   │   ├── embeddings/
│   │   └── model/
│   └── sarcasm/
│       ├── embeddings/
│       ├── model/
```

<!-- TREEVIEW END -->



Files present:


- `word2vec_actual.py` - The first implementation of word2vec. Very Slow as it uses Python lists and redundant code.
- `word2vec_optimized.py` - The final optimized version of word2vec_actual.py. Uses NumPy and faster algos. Very FAST!!!
- `hate/hatedetectionclassifier.ipynb` - FFNN classifier to classify hate sentences via word embeddings
- `sarcasm/sarcasmdetectionclassifier.ipynb` - FFNN classifier to classify sarcasm sentences via word embeddings
- `humor/humordetectionclassifier.ipynb` - FFNN classifier to classify humor sentences via word embeddings
Directories present:

- `models/word2vec` - Includes all the models and embeddings created during the process.
- `submission` - Includes the final submission file.


## How to create Vector Embeddings?
You can start creating the vector embeddings by running the word2vec_optimized.py file
``` console
python word2vec_optimized.py --data-path ../data/hate/train.csv --save-dir ./models/ --embedding_dims 100 --epochs 100 --lr 0.001 --batch_size 512 --window_size 3 --neg_samples 3

```

Arguments used in `word2vec_optimized.py`
1.  `--data_path`  -  type=str, default='../data/hate/train.csv', help='Path to training CSV'
2. `--save_dir`  -  type=str, default='models/word2vec', help='Directory to save model and embeddings'
3. `--embedding_dim`  -  type=int, default=100, help='Size of word embeddings'
4. `--epochs`  -  type=int, default=50, help='Number of training epochs'
5. `--lr`  -  type=float, default=0.001, help='Learning rate'
6. `--batch_size`  -  type=int, default=512, help='Batch size for training'
7. `--window_size`  -  type=int, default=3, help='Window size for context'
8. `--neg_samples`  -  type=int, default=3, help='Number of negative samples'

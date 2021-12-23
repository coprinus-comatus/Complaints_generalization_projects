# Complaints_generalization_projects

This is the github folder related to this Medium article.
You can find here more information on the models that were used and compared.

For the Data-Driven approaches, we selected a collection of models based on pre-trained transformers, spaCy optimizations, classic algorithms, and on-demand services. The ttransfomers rely on a self-attention mechanisms and they have been trained on large collections of text. They can be fine tuned in to several difference tasks, such as sequence classification, next sentence prediction or question answering. We selected a lightweight transformer (distilBERT) and two large models (BERT-large and RoBERTA-large) to compare their performance in our experiments. From the SpaCy NLP library, we selected the a text categorizer which that ensembles a linear model and a CNN. The scikit-learn library offers a set of classic algorithms which that can be applied to NLP, such as Support Vector Machines (SVM), Naïve Bayes (NB), Random Forest (RF) or Logaritmic Logistic Regression (LR). And fFinally, we tested included the cloud NLP services from provided by Google and Amazon. 

## SK Models
These are the settings that we used to produce the models:

text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=STOPWORDS)), ('clf', model)], verbose=True)
 
The ‘STOPWORDS’ are the ones from the nltk library, and the ‘model’ can be whatever of the following:
 
if model_name == "NB":
            model = MultinomialNB()
if model_name == "LR":
            model = LogisticRegression(solver='liblinear', multi_class='auto', random_state=42)
if model_name == "RF":
            model = RandomForestClassifier(n_estimators=100,max_features='sqrt', max_depth=None, min_samples_split=2, random_state=42)
if model_name == "SVM":
            model = LinearSVC(random_state=42)
 

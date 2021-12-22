# Complaints_generalization_projects

This is the github folder related to this Medium article.
You can find here more information on the models that were used and compared.

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
 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import impute
from multilabel import multilabel_train_test_split

df = pd.read_csv('~/Coding/School_budget/TrainingData.csv', index_col = 0)
df.head()
df.info()
df.describe()

plt.hist(df[0:1000]['FTE'].dropna())
plt.title('percentage of full time employee works')
plt.xlabel('employees')
plt.xlim([0, 1])
plt.show()

df.dtypes.value_counts()

labels = ['Function',
          'Use',
          'Sharing',
          'Reporting',
          'Student_Type',
          'Position_Type',
          'Object_Type', 
          'Pre_K',
          'Operating_Status']

categorize_label = lambda x: x.astype('category')

df[labels] = categorize_label(df[labels])
df[labels].dtypes

unique_values = df[labels].apply(pd.Series.nunique)

unique_values.plot(kind='bar')
plt.xlabel('Labels')
plt.ylabel('nums')
plt.show()

# Using alphanumeric token pattern
tokens = '[A-Za-z0-9]+(?=\\s+)'

numeric_columns = ['FTE', 'Total']
text_columns = [i for i in df.columns if i not in labels + numeric_columns]

def combine_text_colunms(data_frame = df, to_drop = numeric_columns + labels):

    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis = 1)

    text_data.fillna("", inplace = True)

    return text_data.apply(lambda x: " ".join(x), axis = 1)

non_label = [i for i in df.columns if i not in labels]

dummy_labels = pd.get_dummies(df[labels])

X_train, X_test, y_train, y_test = multilabel_train_test_split(df[non_label], dummy_labels, 0.2, seed=2021)

get_text = FunctionTransformer(combine_text_colunms, validate = False)
get_numeric = FunctionTransformer(lambda x: x[numeric_columns], validate = False)

# Log regression
# pl = Pipeline([
#     ('union', FeatureUnion(
#     transformer_list = [
#         ('numeric', Pipeline([('selector', get_numeric), ('imputer', impute.SimpleImputer(missing_values=np.nan, strategy='mean'))])), 
#         ('text_features', Pipeline([('selector', get_text), ('vectorizer', CountVectorizer(token_pattern=tokens, ngram_range=(1,2)))]))
#     ])),     
#     ('scale', MaxAbsScaler()),
#     ('lr', OneVsRestClassifier(LogisticRegression()))]
# )

# pl.fit(X_train, y_train)

# accuracy = pl.score(X_test, y_test)
# print(accuracy)

# Random forest
pl_2 = Pipeline([
    ('union', FeatureUnion(
    transformer_list = [('numeric', Pipeline([('selector', get_numeric), ('imputer', impute.SimpleImputer(missing_values=np.nan, strategy='mean'))])), 
    ('text_features', Pipeline([('selector', get_text), ('vectorizer', CountVectorizer(token_pattern=tokens, ngram_range=(1,2)))]))
    ])), 
    ('rm', RandomForestClassifier(n_estimators=20))]
)

pl_2.fit(X_train, y_train)

accuracy_2 = pl_2.score(X_test, y_test)
print(accuracy_2)
# The accuracy of random forest model is 90.96%
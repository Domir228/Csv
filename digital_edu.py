import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
df2 = pd.read_csv('test.csv')
df = pd.read_csv('train.csv')
df.drop(['bdate', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 'relation', 'life_main', 'people_main', 'city', 'last_seen', 'occupation_name', 'career_start', 'career_end'], axis = 1, inplace = True)
df2.drop(['bdate', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 'relation', 'life_main', 'people_main', 'city', 'last_seen', 'occupation_name', 'career_start', 'career_end'], axis = 1, inplace = True)
df['education_form'].fillna('Full-time', inplace=True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df["education_form"]) 
df2[list(pd.get_dummies(df2['education_form']).columns)] = pd.get_dummies(df2["education_form"]) 
df2.drop(['education_form'], axis=1, inplace=True)

df.drop(['education_form'], axis=1, inplace=True)
df2['education_status'].value_counts().plot(kind = 'pie', grid = True)
plt.show()
def edu_status_apply(edu_status):
    if edu_status == 'Undergraduate applicant':
        return 0
    elif edu_status == 'Student (Specialist)' or edu_status=="Student (Bachelor's)" or edu_status=="Student (Master's)": 
        return 1
    elif edu_status== "Alumnus (Bachelor's)" or edu_status == "Alumnus (Master's)" or edu_status == "Alumnus (Specialist)": 
        return 2
    else:
        return 3
df['education_status'] =  df['education_status'].apply(edu_status_apply)
df2['education_status'] =  df2['education_status'].apply(edu_status_apply)

def lang_aplly(langs):
    if langs.find('Русский') != -1 and langs.find('English') != -1:
        return 2
    return 1

df['langs'] = df['langs'].apply(lang_aplly)
df2['langs'] = df2['langs'].apply(lang_aplly)


def life_main_apply(life_main):
    if life_main == 'False':
        return 9
    if life_main == '0':
        return 0
    if life_main == '1':
        return 1
    if life_main == '2':
        return 2
    if life_main == '3':
        return 3
    if life_main == '4':
        return 4
    if life_main == '5':
        return 5
    if life_main == '6':
        return 6
    if life_main == '7':
        return 7
    if life_main == '8':
        return 8
def life_people_apply(life_main):
    if life_main == 'False':
        return 7
    if life_main == '0':
        return 0
    if life_main == '1':
        return 1
    if life_main == '2':
        return 2
    if life_main == '3':
        return 3
    if life_main == '4':
        return 4
    if life_main == '5':
        return 5
    if life_main == '6':
        return 6
    



def ocu_type_apply(ocu_type):
    if ocu_type == 'university':
        return 0
    return 1
df2['occupation_type'] =  df2['occupation_type'].apply(ocu_type_apply)

df['occupation_type'] = df['occupation_type'].apply(ocu_type_apply)
x = df.drop('result', axis = 1)
y = df['result']
ID = df2['id']
x2 = df.drop('id', axis = 1)
y2= df['id']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
percent = accuracy_score(y_test, y_pred) * 100
x_test = sc.transform(df2)
y_pred = classifier.predict(x_test)
result = pd.DataFrame({'id': ID, 'result': y_pred})
result.to_csv('res.csv', index=False)
print(percent)
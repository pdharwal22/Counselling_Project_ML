import pandas as pd
print(chr(27) + '[2J')

df = pd.read_csv('Marks.csv')
df = df.drop(['Unnamed: 0'], axis=1)
# print(df)

columns = [1, 2, 3, 4, 5, 6, 7, 8, 9]
eng_df = df[df['subject'] == 'English']
eng_df = eng_df.drop(['reg1', 'subject'], axis=1)
eng_df = eng_df.T
print(eng_df)
eng_df.to_csv('Datasets/eng.csv')
print('English dataset created')

math_df = df[df['subject'] == 'Maths']
math_df = math_df.drop(['reg1', 'subject'], axis=1)
math_df = math_df.T
# print(math_df)
# math_df.to_csv('Datasets/maths.csv')
# print('Maths dataset created')

sci_df = df[df['subject'] == 'Science']
sci_df = sci_df.drop(['reg1', 'subject'], axis=1)
sci_df = sci_df.T
# print(sci_df)
# sci_df.to_csv('Datasets/science.csv')
# print('Science dataset created')

soc_df = df[df['subject'] == 'Social Science']
soc_df = soc_df.drop(['reg1', 'subject'], axis=1)
soc_df = soc_df.T
# print(soc_df)
# soc_df.to_csv('Datasets/socio.csv')
# print('Social Science dataset created')

gk_df = df[df['subject'] == 'G.K.']
gk_df = gk_df.drop(['reg1', 'subject'], axis=1)
gk_df = gk_df.T
# print(gk_df)
# gk_df.to_csv('Datasets/gk.csv')
# print('G.K. dataset created')

ph_df = df[df['subject'] == 'Physical Education']
ph_df = ph_df.drop(['reg1', 'subject'], axis=1)
ph_df = ph_df.T
# print(ph_df)
# ph_df.to_csv('Datasets/phed.csv')
# print('Physical Education dataset created')

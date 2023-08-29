import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, r2_score


################# RB Model ######################

# read in rb_cbs_22, rb_cbs_21, rb_cbs_20, rb_cbs_19, rb_cbs_18, rb_cbs_17, rb_cbs_16, rb_cbs_15
rb_cbs_22 = pd.read_csv('rb_cbs_22.csv')
rb_cbs_21 = pd.read_csv('rb_cbs_21.csv')
rb_cbs_20 = pd.read_csv('rb_cbs_20.csv')
rb_cbs_19 = pd.read_csv('rb_cbs_19.csv')
rb_cbs_18 = pd.read_csv('rb_cbs_18.csv')
rb_cbs_17 = pd.read_csv('rb_cbs_17.csv')
rb_cbs_16 = pd.read_csv('rb_cbs_16.csv')
rb_cbs_15 = pd.read_csv('rb_cbs_15.csv')

# read in rb_fg_22, rb_fg_21, rb_fg_20, rb_fg_19, rb_fg_18, rb_fg_17, rb_fg_16, rb_fg_15
rb_fg_22 = pd.read_csv('rb_fg_22.csv')
rb_fg_21 = pd.read_csv('rb_fg_21.csv')
rb_fg_20 = pd.read_csv('rb_fg_20.csv')
rb_fg_19 = pd.read_csv('rb_fg_19.csv')
rb_fg_18 = pd.read_csv('rb_fg_18.csv')
rb_fg_17 = pd.read_csv('rb_fg_17.csv')
rb_fg_16 = pd.read_csv('rb_fg_16.csv')
rb_fg_15 = pd.read_csv('rb_fg_15.csv')

# combine rb_cbs_22 and rb_fg_22 on PLAYER
rb_22 = pd.merge(rb_cbs_22, rb_fg_22, on='PLAYER', how='inner')

# combine rb_cbs_21 and rb_fg_21 on PLAYER
rb_21 = pd.merge(rb_cbs_21, rb_fg_21, on='PLAYER', how='inner')

# combine rb_cbs_20 and rb_fg_20 on PLAYER
rb_20 = pd.merge(rb_cbs_20, rb_fg_20, on='PLAYER', how='inner')

# combine rb_cbs_19 and rb_fg_19 on PLAYER
rb_19 = pd.merge(rb_cbs_19, rb_fg_19, on='PLAYER', how='inner')

# combine rb_cbs_18 and rb_fg_18 on PLAYER
rb_18 = pd.merge(rb_cbs_18, rb_fg_18, on='PLAYER', how='inner')

# combine rb_cbs_17 and rb_fg_17 on PLAYER
rb_17 = pd.merge(rb_cbs_17, rb_fg_17, on='PLAYER', how='inner')
# combine rb_cbs_16 and rb_fg_16 on PLAYER
rb_16 = pd.merge(rb_cbs_16, rb_fg_16, on='PLAYER', how='inner')
# combine rb_cbs_15 and rb_fg_15 on PLAYER
rb_15 = pd.merge(rb_cbs_15, rb_fg_15, on='PLAYER', how='inner')

# combine all dataframes into one
rb = pd.concat([rb_22, rb_21, rb_20, rb_19, rb_18, rb_17, rb_16, rb_15])

# load the file pff_rank
pff_rank = pd.read_csv('pff_rank.csv')

# combine rb and pff_rank on TEAM and YEAR
rb = pd.merge(rb, pff_rank, on=['TEAM','YEAR'], how='inner')

# load the dataset rb_23
rb_23 = pd.read_csv('rb_23.csv')

# rb_23 and pff_rank on TEAM and YEAR
rb_23 = pd.merge(rb_23, pff_rank, on=['TEAM','YEAR'], how='inner')
rb_23 = rb_23.drop(columns=['YEAR'], axis=1)

# create copies that keeps all columns
rb_all = rb.copy()
rb_23_all = rb_23.copy()

# calculate the median Games Played for the dataset from column G for each player and create a new dataframe called rb_med that stores the values for each player
rb_med = rb.groupby('PLAYER')['G'].median().reset_index()
rb_avg = rb.groupby('PLAYER')['G'].mean().round().reset_index()

# calculate the median Games Played for the dataset from column G for each player that have EXP = 1 and create a new value called rb_rook_games
rb_rook = rb[rb['EXP'] == 1]
rb_rook_games = rb_rook['G'].mean().round()

# join rb_avg to rb_23 on PLAYER
rb_23 = pd.merge(rb_23, rb_avg, on='PLAYER', how='left')

# assign the rb_rook_games value to all rows in rb_23 where EXP = 1
rb_23.loc[rb_23['EXP'] == 1, 'G'] = rb_rook_games

# remove all rows with missing values
rb_23 = rb_23.dropna()

# drop columns that are not needed
rb = rb.drop(columns=['RECAVG','TAR','RSHYD', 'YRSH', 'RSHTD', 'REC', 'RECYD', 'RECTD', 'FGPG','FGRANK', 'CBSAVG', 'CBSTOT', 'YEAR', 'FGTP'], axis=1)

# create an ordinary least squares linear regression model for RSHATT
model = LinearRegression()

# create the feature matrix from all columns but RSHATT
X = rb.drop(columns=['RSHATT','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = rb['RSHATT']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - RSHATT Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for RSHATT for all players and add to rb_23 dataframe
rb_23 = rb_23[['PLAYER','POS','TEAM','BYE','LOST','START','AGE','EXP','G','PFF_ORNK_PRE']]
rb_23['RSHATT'] = model.predict(rb_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
rb_23['RSHATT'] = rb_23['RSHATT'].apply(lambda x: 0 if x < 0 else x)

# round all results for RSHATT
rb_23['RSHATT'] = rb_23['RSHATT'].round()

# create the feature matrix from all columns but RSHYD
rb['RSHYD'] = rb_all['RSHYD']
X = rb.drop(columns=['RSHYD','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = rb['RSHYD']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - RSHYD Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for RSHTD for all players and add to rb_23 dataframe
rb_23 = rb_23[['PLAYER','POS','TEAM','BYE','LOST','START','AGE','EXP','G','RSHATT','PFF_ORNK_PRE']]
rb_23['RSHYD'] = model.predict(rb_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
rb_23['RSHYD'] = rb_23['RSHYD'].apply(lambda x: 0 if x < 0 else x)

# round all results for RSHYD
rb_23['RSHYD'] = rb_23['RSHYD'].round()

# create the feature matrix from all columns but RSHTD
rb['RSHTD'] = rb_all['RSHTD']
X = rb.drop(columns=['RSHTD','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = rb['RSHTD']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - RSHTD Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for RSHTD for all players and add to rb_23 dataframe
rb_23 = rb_23[['PLAYER','POS','TEAM','BYE','LOST','START','AGE','EXP','G','RSHATT','PFF_ORNK_PRE','RSHYD',]]
rb_23['RSHTD'] = model.predict(rb_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
rb_23['RSHTD'] = rb_23['RSHTD'].apply(lambda x: 0 if x < 0 else x)

# round all results for RSHTD
rb_23['RSHTD'] = rb_23['RSHTD'].round()

# create the feature matrix from all columns but REC
rb['REC'] = rb_all['REC']
X = rb.drop(columns=['REC','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = rb['REC']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - REC Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for REC for all players and add to rb_23 dataframe
rb_23 = rb_23[['PLAYER','POS','TEAM','BYE','LOST','START','AGE','EXP','G','RSHATT','PFF_ORNK_PRE','RSHYD','RSHTD']]
rb_23['REC'] = model.predict(rb_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
rb_23['REC'] = rb_23['REC'].apply(lambda x: 0 if x < 0 else x)

# round all results for REC
rb_23['REC'] = rb_23['REC'].round()

# create the feature matrix from all columns but RECYD
rb['RECYD'] = rb_all['RECYD']
X = rb.drop(columns=['RECYD','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = rb['RECYD']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - RECYD Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for RECYD for all players and add to rb_23 dataframe
rb_23 = rb_23[['PLAYER','POS','TEAM','BYE','LOST','START','AGE','EXP','G','RSHATT','PFF_ORNK_PRE','RSHYD','RSHTD','REC']]
rb_23['RECYD'] = model.predict(rb_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
rb_23['RECYD'] = rb_23['RECYD'].apply(lambda x: 0 if x < 0 else x)

# round all results for RECYD
rb_23['RECYD'] = rb_23['RECYD'].round()

# create the feature matrix from all columns but RECTD
rb['RECTD'] = rb_all['RECTD']
X = rb.drop(columns=['RECTD','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = rb['RECTD']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - RECTD Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for RECTD for all players and add to rb_23 dataframe
rb_23 = rb_23[['PLAYER','POS','TEAM','BYE','LOST','START','AGE','EXP','G','RSHATT','PFF_ORNK_PRE','RSHYD','RSHTD','REC','RECYD']]
rb_23['RECTD'] = model.predict(rb_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
rb_23['RECTD'] = rb_23['RECTD'].apply(lambda x: 0 if x < 0 else x)

# round all results for RECTD
rb_23['RECTD'] = rb_23['RECTD'].round()

# create the feature matrix from all columns but CBSTOT
rb['CBSTOT'] = rb_all['CBSTOT']
X = rb.drop(columns=['CBSTOT','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = rb['CBSTOT']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - CBSTOT Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for CBSTOT for all players and add to rb_23 dataframe
rb_23 = rb_23[['PLAYER','POS','TEAM','BYE','LOST','START','AGE','EXP','G','RSHATT','PFF_ORNK_PRE','RSHYD','RSHTD','REC','RECYD','RECTD']]
rb_23['CBSTOT'] = model.predict(rb_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
rb_23['CBSTOT'] = rb_23['CBSTOT'].apply(lambda x: 0 if x < 0 else x)

# round all results for CBSTOT
rb_23['CBSTOT'] = rb_23['CBSTOT'].round()

# print rb_23 and sort by CBSTOT descending
print("\n RB MODEL RESULTS \n")
print(rb_23.sort_values(by=['CBSTOT'], ascending=False).head(20))
print('\n')

# export all results to a csv file name rb_model.csv
rb_23.to_csv('rb_model.csv', index=False)

################# WR Model ######################

# read in cbs_wr_stats and fg_wr_stats
cbs_wr_stats = pd.read_csv('cbs_wr_stats.csv')
fg_wr_stats = pd.read_csv('fg_wr_stats.csv')

cbs_wr_stats = cbs_wr_stats.drop(columns=['TEAM'], axis=1)

# combine cbs_wr_stats and fg_wr_stats on PLAYER AND YEAR
wr = pd.merge(cbs_wr_stats, fg_wr_stats, on=['PLAYER','YEAR'], how='inner')

# combine wr and pff_rank on TEAM and YEAR
wr = pd.merge(wr, pff_rank, on=['TEAM','YEAR'], how='inner')

wr_23 = pd.read_csv('wr_23.csv')

# create copies that keeps all columns
wr_all = wr.copy()
wr_23_all = wr_23.copy()

# calculate the median Games Played for the dataset from column G for each player and create a new dataframe called wr_med that stores the values for each player
wr_med = wr.groupby('PLAYER')['G'].median().reset_index()

# calculate the average Games Played for the dataset from column G for each player and create a new dataframe called wr_med that stores the values for each player
wr_avg = wr.groupby('PLAYER')['G'].mean().round().reset_index()

# calculate the median Games Played for the dataset from column G for each player that have EXP = 1 and create a new value called wr_rook_games
wr_rook = wr[wr['EXP'] == 1]
wr_rook_games = wr_rook['G'].median()

# combine wr_23 and pff_rank on TEAM and YEAR
wr_23 = pd.merge(wr_23, pff_rank, on=['TEAM','YEAR'], how='inner')
wr_23 = wr_23.drop(columns=['YEAR'], axis=1)

# join wr_med to wr_23 on PLAYER
wr_23 = pd.merge(wr_23, wr_med, on='PLAYER', how='left')

# assign the wr_rook_games value to all rows in wr_23 where G is null
wr_23.loc[wr_23['G'].isnull(), 'G'] = wr_rook_games

# calculate the max value for EXP for each player and create a new dataframe called wr_exp that stores the values for each player
wr_exp = wr.groupby('PLAYER')['EXP'].max().reset_index()

# join wr_exp to wr_23 on PLAYER
wr_23 = pd.merge(wr_23, wr_exp, on='PLAYER', how='left')

# add a value of 1 to each of the rows where EXP is not null
wr_23.loc[wr_23['EXP'].notnull(), 'EXP'] += 1

# add a value of 1 to each of the rows where EXP is null
wr_23.loc[wr_23['EXP'].isnull(), 'EXP'] = 1

# calculate the max value for AGE for each player and create a new dataframe called wr_age that stores the values for each player
wr_age = wr.groupby('PLAYER')['AGE'].max().reset_index()

# join wr_age to wr_23 on PLAYER
wr_23 = pd.merge(wr_23, wr_age, on='PLAYER', how='left')

# calculate the average value for AGE for each player where EXP = 1 and create a new dataframe called wr_rook_age that stores the value
wr_rook_age = wr[wr['EXP'] == 1]
wr_rook_age = wr_rook_age['AGE'].mean().round()

# assign the wr_rook_age value to all rows in wr_23 where AGE is null
wr_23.loc[wr_23['AGE'].isnull(), 'AGE'] = wr_rook_age

# add a column of year to wr_23 with a value of 2023 for all rows
wr_23['YEAR'] = 2023

wr_23 = wr_23.dropna()

wr_23 = wr_23[['PLAYER','YEAR','TEAM','POS','TOPREC','BYE','LOST','AGE','EXP','G','RSHATT','RSHYD','RSHTD','PFF_ORNK_PRE']]

# remove columns that are not needed
wr = wr.drop(columns=['CBSAVG','CBSTOT','TAR','YEAR','RSHYD','REC','RECYD','YDREC','RECTD','FGPG','FGTP','FGRNK'], axis=1)

# create an ordinary least squares linear regression model for RSHATT
model = LinearRegression()

# create the feature matrix from all columns but RSHATT
X = wr.drop(columns=['RSHATT','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = wr['RSHATT']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - RSHATT Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for RSHATT for all players and add to wr_23 dataframe
wr_23 = wr_23[['PLAYER','TEAM','POS','TOPREC','BYE','LOST','AGE','EXP','G','RSHTD','PFF_ORNK_PRE']]
wr_23['RSHATT'] = model.predict(wr_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
wr_23['RSHATT'] = wr_23['RSHATT'].apply(lambda x: 0 if x < 0 else x)

# round all results for RSHATT
wr_23['RSHATT'] = wr_23['RSHATT'].round()

# create the feature matrix from all columns but RSHYD
wr['RSHYD'] = wr_all['RSHYD']

X = wr.drop(columns=['RSHYD','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = wr['RSHYD']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - RSHYD Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for RSHYD for all players and add to wr_23 dataframe
wr_23 = wr_23[['PLAYER','TEAM','POS','TOPREC','BYE','LOST','AGE','EXP','G','RSHATT','RSHTD','PFF_ORNK_PRE']]
wr_23['RSHYD'] = model.predict(wr_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
wr_23['RSHYD'] = wr_23['RSHYD'].apply(lambda x: 0 if x < 0 else x)

# round all results for RSHYD
wr_23['RSHYD'] = wr_23['RSHYD'].round()

# create the feature matrix from all columns but TAR
wr['TAR'] = wr_all['TAR']
X = wr.drop(columns=['TAR','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = wr['TAR']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22, test_size=0.2)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

print('R-Squared - TAR Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for TAR for all players and add to wr_23 dataframe
wr_23 = wr_23[['PLAYER','TEAM','POS','TOPREC','BYE','LOST','AGE','EXP','G','RSHATT','RSHTD','PFF_ORNK_PRE','RSHYD']]
wr_23['TAR'] = model.predict(wr_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
wr_23['TAR'] = wr_23['TAR'].apply(lambda x: 0 if x < 0 else x)

# round all results for TAR
wr_23['TAR'] = wr_23['TAR'].round()

# create the feature matrix from all columns but REC
wr['REC'] = wr_all['REC']
X = wr.drop(columns=['REC','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = wr['REC']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22, test_size=0.2)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - REC Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for REC for all players and add to wr_23 dataframe
wr_23 = wr_23[['PLAYER','TEAM','POS','TOPREC','BYE','LOST','AGE','EXP','G','RSHATT','RSHTD','PFF_ORNK_PRE','RSHYD','TAR']]
wr_23['REC'] = model.predict(wr_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
wr_23['REC'] = wr_23['REC'].apply(lambda x: 0 if x < 0 else x)

# round all results for REC
wr_23['REC'] = wr_23['REC'].round()

# create the feature matrix from all columns but RECYD
wr['RECYD'] = wr_all['RECYD']
X = wr.drop(columns=['RECYD','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = wr['RECYD']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22, test_size=0.2)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - RECYD Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for RECYD for all players and add to wr_23 dataframe
wr_23 = wr_23[['PLAYER','TEAM','POS','TOPREC','BYE','LOST','AGE','EXP','G','RSHATT','RSHTD','PFF_ORNK_PRE','RSHYD','TAR','REC']]
wr_23['RECYD'] = model.predict(wr_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
wr_23['RECYD'] = wr_23['RECYD'].apply(lambda x: 0 if x < 0 else x)

# round all results for RECYD
wr_23['RECYD'] = wr_23['RECYD'].round()

# create the feature matrix from all columns but RECTD
wr['RECTD'] = wr_all['RECTD']
X = wr.drop(columns=['RECTD','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = wr['RECTD']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22, test_size=0.2)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - RECTD Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for RECTD for all players and add to wr_23 dataframe
wr_23 = wr_23[['PLAYER','TEAM','POS','TOPREC','BYE','LOST','AGE','EXP','G','RSHATT','RSHTD','PFF_ORNK_PRE','RSHYD','TAR','REC','RECYD']]
wr_23['RECTD'] = model.predict(wr_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
wr_23['RECTD'] = wr_23['RECTD'].apply(lambda x: 0 if x < 0 else x)

# round all results for RECTD
wr_23['RECTD'] = wr_23['RECTD'].round()

# create the feature matrix from all columns but CBSTOT
wr['CBSTOT'] = wr_all['CBSTOT']
X = wr.drop(columns=['CBSTOT','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = wr['CBSTOT']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22, test_size=0.2)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - CBSTOT Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for CBSTOT for all players and add to wr_23 dataframe
wr_23 = wr_23[['PLAYER','TEAM','POS','TOPREC','BYE','LOST','AGE','EXP','G','RSHATT','RSHTD','PFF_ORNK_PRE','RSHYD','TAR','REC','RECYD','RECTD']]
wr_23['CBSTOT'] = model.predict(wr_23.drop(columns=['PLAYER','POS','TEAM'], axis=1))

# convert all results less than 0 to 0
wr_23['CBSTOT'] = wr_23['CBSTOT'].apply(lambda x: 0 if x < 0 else x)

# round all results for CBSTOT
wr_23['CBSTOT'] = wr_23['CBSTOT'].round()

# print wr_23 and sort by CBSTOT descending
print("\n WR MODEL RESULTS \n")
print(wr_23.sort_values(by=['CBSTOT'], ascending=False).head(20))
print('\n')

# export all results to a csv file name wr_model.csv
wr_23.to_csv('wr_model.csv', index=False)

################# QB Model

# read in qb_hist
qb_hist = pd.read_csv('qb_hist.csv')

# read in qb_23
qb_23 = pd.read_csv('qb_23.csv')

# combine qb_hist and pff_rank on TEAM and YEAR
qb_hist = pd.merge(qb_hist, pff_rank, on=['TEAM','YEAR'], how='inner')

# combine qb_23 and pff_rank on TEAM and YEAR
qb_23 = pd.merge(qb_23, pff_rank, on=['TEAM','YEAR'], how='inner')

# create copies that keeps all columns
qb_hist_all = qb_hist.copy()
qb_23_all = qb_23.copy()

# calculate the median Games Played for the dataset from column G for each player and create a new dataframe called qb_med that stores the values for each player
qb_med = qb_hist.groupby('PLAYER')['G'].median().reset_index()

# calculate the average Games Played for the dataset from column G for each player that EXP = 1 and create a new dataframe called qb_avg that stores the values for each player
qb_avg = qb_hist.groupby('PLAYER')['G'].mean().round().reset_index()

# join qb_med to qb_23 on PLAYER
qb_23 = pd.merge(qb_23, qb_med, on='PLAYER', how='left')

# calculate the max value for EXP for each player and create a new dataframe called qb_exp that stores the values for each player
qb_exp = qb_hist.groupby('PLAYER')['EXP'].max().reset_index()

# join qb_exp to qb_23 on PLAYER
qb_23 = pd.merge(qb_23, qb_exp, on='PLAYER', how='left')

# add a value of 1 to each of the rows where EXP is not null
qb_23.loc[qb_23['EXP'].notnull(), 'EXP'] += 1

# add a value of 1 to each of the rows where EXP is null
qb_23.loc[qb_23['EXP'].isnull(), 'EXP'] = 1

# calculate the max value for AGE for each player and create a new dataframe called qb_age that stores the values for each player
qb_age = qb_hist.groupby('PLAYER')['AGE'].max().reset_index()

# join qb_age to qb_23 on PLAYER
qb_23 = pd.merge(qb_23, qb_age, on='PLAYER', how='left')

# add a value of 1 to each of the rows where AGE is not null
qb_23.loc[qb_23['AGE'].notnull(), 'AGE'] += 1

# calculate the average value for AGE for each player where EXP = 1 and create a new dataframe called qb_rook_age that stores the value
qb_rook_age = qb_hist[qb_hist['EXP'] == 1]
qb_rook_age = qb_rook_age['AGE'].mean().round()

# assign the qb_rook_age value to all rows in qb_23 where AGE is null
qb_23.loc[qb_23['AGE'].isnull(), 'AGE'] = qb_rook_age

# calculate the average Games Played for the dataset from column G for each player that EXP = 1 and create a new dataframe called qb_rook_games that stores the values for each player
qb_rook = qb_hist[qb_hist['EXP'] == 1]
qb_rook_games = qb_rook['G'].mean().round()

# assign the qb_rook_games value to all rows in qb_23 where G is null
qb_23.loc[qb_23['G'].isnull(), 'G'] = qb_rook_games

# create copies that keeps all columns
qb_hist_all = qb_hist.copy()
qb_23_all = qb_23.copy()

# remove all rows with missing values
qb_23 = qb_23.dropna()

# drop columns that are not needed
qb_hist = qb_hist.drop(columns=['YEAR','CBSRNK','FGRNK','FGPG','CBSTOT','CBSPG','FGTP','PASSATT','CMPER','PASSYD','YDATT','PASSTD','INT','PASS40','PASS2P','RUSH20','RUSH2P',], axis=1)

# create an ordinary least squares linear regression model for PASSCOM
model = LinearRegression()

# create the feature matrix from all columns but PASSCOM
X = qb_hist.drop(columns=['PASSCOM','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = qb_hist['PASSCOM']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)
print('R-Squared - PASSCOM Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for PASSCOM for all players and add to qb_23 dataframe
qb_23 = qb_23[['PLAYER','TEAM','POS','START','AGE','EXP','G','BYE','RSHATT','RSHYD','RSHTD','LOST','PFF_ORNK_PRE']]
qb_23['PASSCOM'] = model.predict(qb_23.drop(columns=['PLAYER','TEAM','POS'], axis=1))

# round all results for PASSCOM
qb_23['PASSCOM'] = qb_23['PASSCOM'].round()

# round all results for PASSCOM
qb_23['PASSCOM'] = qb_23['PASSCOM'].round()

# create an ordinary least squares linear regression model for PASSYD
model = LinearRegression()

# create the feature matrix from all columns but PASSYD
qb_hist['PASSYD'] = qb_hist_all['PASSYD']
X = qb_hist.drop(columns=['PASSYD','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = qb_hist['PASSYD']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - PASSYD Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for PASSYD for all players and add to qb_23 dataframe
qb_23 = qb_23[['PLAYER','TEAM','POS','START','AGE','EXP','G','BYE','PASSCOM','RSHATT','RSHYD','RSHTD','LOST','PFF_ORNK_PRE']]
qb_23['PASSYD'] = model.predict(qb_23.drop(columns=['PLAYER','TEAM','POS'], axis=1))

# round all results for PASSYD
qb_23['PASSYD'] = qb_23['PASSYD'].round()

# round all results for PASSYD
qb_23['PASSYD'] = qb_23['PASSYD'].round()

# create an ordinary least squares linear regression model for PASSTD
model = LinearRegression()

# create the feature matrix from all columns but PASSTD
qb_hist['PASSTD'] = qb_hist_all['PASSTD']
X = qb_hist.drop(columns=['PASSTD','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = qb_hist['PASSTD']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - PASSTD Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for PASSTD for all players and add to qb_23 dataframe
qb_23 = qb_23[['PLAYER','TEAM','POS','START','AGE','EXP','G','BYE','PASSCOM','RSHATT','RSHYD','RSHTD','LOST','PFF_ORNK_PRE','PASSYD']]
qb_23['PASSTD'] = model.predict(qb_23.drop(columns=['PLAYER','TEAM','POS'], axis=1))

# round all results for PASSTD
qb_23['PASSTD'] = qb_23['PASSTD'].round()

# round all results for PASSTD
qb_23['PASSTD'] = qb_23['PASSTD'].round()

# create an ordinary least squares linear regression model for PASSATT
model = LinearRegression()

# create the feature matrix from all columns but PASSATT
qb_hist['PASSATT'] = qb_hist_all['PASSATT']
X = qb_hist.drop(columns=['PASSATT','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = qb_hist['PASSATT']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# print the r-squared score for the model
print('R-Squared - PASSATT Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for PASSATT for all players and add to qb_23 dataframe
qb_23 = qb_23[['PLAYER','TEAM','POS','START','AGE','EXP','G','BYE','PASSCOM','RSHATT','RSHYD','RSHTD','LOST','PFF_ORNK_PRE','PASSYD','PASSTD']]
qb_23['PASSATT'] = model.predict(qb_23.drop(columns=['PLAYER','TEAM','POS'], axis=1))

# round all results for PASSATT
qb_23['PASSATT'] = qb_23['PASSATT'].round()

# create an ordinary least squares linear regression model for INT
model = LinearRegression()

# create the feature matrix from all columns but INT
qb_hist['INT'] = qb_hist_all['INT']
X = qb_hist.drop(columns=['INT','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = qb_hist['INT']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

print('R-Squared - INT Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for INT for all players and add to qb_23 dataframe
qb_23 = qb_23[['PLAYER','TEAM','POS','START','AGE','EXP','G','BYE','PASSCOM','RSHATT','RSHYD','RSHTD','LOST','PFF_ORNK_PRE','PASSYD','PASSTD','PASSATT']]
qb_23['INT'] = model.predict(qb_23.drop(columns=['PLAYER','TEAM','POS'], axis=1))

# round all results for INT
qb_23['INT'] = qb_23['INT'].round()

# create an ordinary least squares linear regression model for PASS40
model = LinearRegression()

# create the feature matrix from all columns but PASS40
qb_hist['PASS40'] = qb_hist_all['PASS40']
X = qb_hist.drop(columns=['PASS40','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = qb_hist['PASS40']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)

# fit the model to the training data
model.fit(X_train, y_train)

print('R-Squared - PASS40 Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for PASS40 for all players and add to qb_23 dataframe
qb_23 = qb_23[['PLAYER','TEAM','POS','START','AGE','EXP','G','BYE','PASSCOM','RSHATT','RSHYD','RSHTD','LOST','PFF_ORNK_PRE','PASSYD','PASSTD','PASSATT','INT']]
qb_23['PASS40'] = model.predict(qb_23.drop(columns=['PLAYER','TEAM','POS'], axis=1))

# round all results for PASS40
qb_23['PASS40'] = qb_23['PASS40'].round()

# create an ordinary least squares linear regression model for CBSTOT
model = LinearRegression()

# create the feature matrix from all columns but CBSTOT
qb_hist['CBSTOT'] = qb_hist_all['CBSTOT']
X = qb_hist.drop(columns=['CBSTOT','PLAYER','TEAM','POS'], axis=1)

# create the target vector
y = qb_hist['CBSTOT']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)

# fit the model to the training data
model.fit(X_train, y_train)

print('R-Squared - CBSTOT Model: {}'.format(r2_score(y_test, y_pred)))

# predict values for CBSTOT for all players and add to qb_23 dataframe
qb_23 = qb_23[['PLAYER','TEAM','POS','START','AGE','EXP','G','BYE','PASSCOM','RSHATT','RSHYD','RSHTD','LOST','PFF_ORNK_PRE','PASSYD','PASSTD','PASSATT','INT','PASS40']]
qb_23['CBSTOT'] = model.predict(qb_23.drop(columns=['PLAYER','TEAM','POS'], axis=1))

# round all results for CBSTOT
qb_23['CBSTOT'] = qb_23['CBSTOT'].round()

# print qb_23 and sort by CBSTOT descending
print("\n QB MODEL RESULTS \n")
print(qb_23.sort_values(by=['CBSTOT'], ascending=False).head(20))
print('\n')

# export all results to a csv file name qb_model.csv
qb_23.to_csv('qb_model.csv', index=False)


print('\n')
print('TOP SCORING QBs - Historical')
# print up the top scoring players from each position from qb_hist
print(qb_hist.sort_values(by=['CBSTOT'], ascending=False).head(20))

print('\n')
print('TOP SCORING RBs - Historical')
# print up the top scoring players from each position from rb
print(rb.sort_values(by=['CBSTOT'], ascending=False).head(20))

print('\n')
print('TOP SCORING WRs - Historical')
# print up the top scoring players from each position from wr
print(wr.sort_values(by=['CBSTOT'], ascending=False).head(20))

# CREATE DRAFTED TEAM #
# create a dataframe called team_qb that contains all values from qb_23 for Lamar Jackson
team_qb = qb_23[qb_23['PLAYER'] == 'Lamar Jackson']

# create a dataframe called team_rb that contains all values from rb_23 for Christian McCaffrey','D'Andre Swift','David Montgomery','Breece Hall','DeAndre Swift','Rashard Penny'
team_rb = rb_23[rb_23['PLAYER'].isin(['Christian McCaffrey','D\'Andre Swift','David Montgomery','Breece Hall','DeAndre Swift','Rashaad Penny'])]

# create a dataframe called team_wr that contains all values from wr_23 for A.J. Brown','Deebo Samuel','George Pickens','Romeo Doubs','Mike Evans','Courtland Sutton'
team_wr = wr_23[wr_23['PLAYER'].isin(['A.J. Brown','Deebo Samuel','George Pickens','Romeo Doubs','Mike Evans','Courtland Sutton'])]

# merge team_qb, team_rb, and team_wr into a single dataframe called rrats
team = pd.concat([team_qb, team_rb, team_wr])

team = team[['PLAYER','CBSTOT','TEAM','POS','BYE','AGE','EXP']]

print('\n')
print('########## DRAFTED TEAM ##########\n')

# print the results order by POS ascending and CBSTOT descending
print(team.sort_values(by=['POS','CBSTOT'], ascending=[True,False]))

# print the sum of the values for CBSTOT for team
print('\n')
print('TOTAL POINTS: {}'.format(team['CBSTOT'].sum()))

#Dimension reduction
if 7 in indexes_to_test:
    print('Testing accuracy after dimension reduction')
    df = pd.read_csv('UNSW_NB15_training-set.csv',index_col=0)
    df = df.drop(['proto','attack_cat'], axis=1)
    keep_back_col_names = ['label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names,axis=1)
    df = numerize_cat_vals(df,['service','state'])
    u,s,v = svd(df.values,full_matrices=False)
    new_s = s[:10]
    new_u = u[:,:10]
    new_u = new_u.dot(new_s)
    test_accuracy(new_u,keep_back['label'].values)


def remove_sparse_feauter_values(data_frame, columns, threshold=100):
    for column in columns:
        indexes=[]
        counts = dict(Counter(df[column].tolist()))
        for index, value in df.iterrows():
            if counts[value[column]] > 100:
                indexes.append(index)
        data_frame = data_frame.iloc[indexes[:len(indexes)-1000]]
    return data_frame

#remove sparse feature values
if 6 in indexes_to_test:
    print('Testing accuracy after removing outliers')
    df = pd.read_csv('UNSW_NB15_training-set.csv',index_col=0)
    df = remove_sparse_feauter_values(df,['proto','service','state'])
    df = numerize_cat_vals(df,['proto','service','state'])
    test_accuracy(df,'label')
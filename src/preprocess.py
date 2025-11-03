


def clean_data(df):
    #codes for cleaning


    #last thing to do
    y = df['Land Value']
    X = df[[
        'Postal Code',
        'Year', #numeric
        'Quarter', #categorical
        'Nature of Mutation', 
        'Residence Type',
        'Land Area', 
        'Living Area', 
        'Number of Rooms', 
        'Number of Lots'
    ]]
    return X, y
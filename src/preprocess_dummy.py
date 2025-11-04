

def clean_dummy(df):
    y = df['Land Value']
    X = df[[
        'Postal Code',
        'Year', #numeric
        'Quarter', #categorical
        'Residence Type_Apartment',
        'Residence Type_House',
        'Nature of Mutation_Sale',
        'Nature of Mutation_Before completion',
        'Land Area', 
        'Living Area', 
        'Number of Rooms', 
        'Number of Lots'
    ]]
    return X, y
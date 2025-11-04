


def clean_data(df):
    #codes for cleaning


    #last thing to do
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
        'Number of Lots',
        'Longitude',
        'Latitude'
    ]]
    return X, y
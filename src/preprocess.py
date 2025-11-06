import pandas as pd

def rename(df):
    column_rename_dict = {
    'Identifiant de document': 'Document ID',
    'Reference document': 'Document reference',
    '1 Articles CGI': 'CGI Article 1',
    '2 Articles CGI': 'CGI Article 2',
    '3 Articles CGI': 'CGI Article 3',
    '4 Articles CGI': 'CGI Article 4',
    '5 Articles CGI': 'CGI Article 5',
    'No disposition': 'Disposition number',
    'Date mutation': 'Date of mutation',
    'Nature mutation': 'Nature of mutation',
    'Valeur fonciere': 'Property value',
    'No voie': 'Street number',
    'B/T/Q': 'Building/Tower/Block',
    'Type de voie': 'Street type',
    'Code voie': 'Street code',
    'Voie': 'Street name',
    'Code postal': 'Postal code',
    'Commune': 'Municipality',
    'Code departement': 'Department code',
    'Code commune': 'Municipality code',
    'Prefixe de section': 'Section prefix',
    'Section': 'Section',
    'No plan': 'Plan number',
    'No Volume': 'Volume number',
    '1er lot': '1st unit',
    'Surface Carrez du 1er lot': 'Carrez surface of 1st unit',
    '2eme lot': '2nd unit',
    'Surface Carrez du 2eme lot': 'Carrez surface of 2nd unit',
    '3eme lot': '3rd unit',
    'Surface Carrez du 3eme lot': 'Carrez surface of 3rd unit',
    '4eme lot': '4th unit',
    'Surface Carrez du 4eme lot': 'Carrez surface of 4th unit',
    '5eme lot': '5th unit',
    'Surface Carrez du 5eme lot': 'Carrez surface of 5th unit',
    'Nombre de lots': 'Number of lots',
    'Code type local': 'Local type code',
    'Type local': 'Type of property',
    'Identifiant local': 'Local identifier',
    'Surface reelle bati': 'Actual built surface',
    'Nombre pieces principales': 'Number of rooms',
    'Nature culture': 'Type of land use',
    'Nature culture speciale': 'Special land use',
    'Surface terrain': 'Land area'
    }
    df.rename(columns = column_rename_dict, inplace = True)
    return df

def select_column(df):
    column_needed = [
        'Document ID',
        'Date of mutation',
        'Nature of mutation',
        'Property value',
        'Street number',
        'Street type',
        'Street name',
        'Postal code',
        'Municipality',
        'Number of rooms',
        'Land area',
        'Type of land use',
        'Type of property',
        'Actual built surface',
        'Number of lots'
    ]
    return df[[col for col in column_needed if col in df.columns]]

def sales_and_residential(df):
    sales_type = ['Vente', "Vente en l'état futur d'achèvement"]
    residential_type = ['Appartement', 'Maison']

    df_one = df[df['Type of property'].isin(residential_type)]
    df_two = df_one[df_one['Nature of mutation'].isin(sales_type)]
    
    return df_two

def drop_na(df):
    na_columns = ['Postal code', 'Property value', 'Actual built surface', 'Number of rooms']
    df_dropped = df.dropna(subset = na_columns)
    return df_dropped


def clean_data(df_raw):
    selected_cities = [
    'Paris',
    'Marseille',
    'Lyon',
    'Toulouse',
    'Nice',
    'Nantes',
    'Montepellier',
    'Bordeaux',
    'Lille'
]

    df = df_raw[df_raw['City'].isin(selected_cities)].copy()
    print(df.head())


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
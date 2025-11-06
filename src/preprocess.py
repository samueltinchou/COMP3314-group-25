
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
    'Type de voie': 'Type of street',
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
    'Nombre de lots': 'Number of units',
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

def select_column(df):
    column_needed = [
        'Document ID',
        'Date of mutation',
        'Nature of mutation',
        'Property value',
        'Street number',
        'Building/Tower/Block',
    ]


def clean_data(df):
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
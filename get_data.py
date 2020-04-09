""" Get data from database and first round process.

    NOTE: This file is meant to be run in the AWS EC2 instance
          to bypass the GFW
"""

import os
import re
import json
from mysql import connector
from mysql.connector.cursor import MySQLCursor
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

PATH = os.path.join(os.curdir, 'data')

def create_data_folder():
    """ Create folder for data stroage if it does not exist."""
    if os.path.exists(PATH) is False or os.path.isdir(PATH) is False:
        os.mkdir(PATH)

def get_db_cnx():
    """ Connect to the databse if database is not connected
        Return a database cursor
    """
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USERNAME'),
        'password': os.getenv('DB_PASSWORD')
    }

    try:
        print('Connecting to Database mysql://{}/{}'.format(db_config['host'], db_config['database']))
        cnx = connector.connect(**db_config)
        print('Database connected')
    except connector.Error as err:
        if err.errno == connector.errorcode.ER_ACCESS_DENIED_ERROR:
            print('Username and password are not correct')
        if err.errno == connector.errorcode.ER_BAD_DB_ERROR:
            print('Requested database doesn\'t exist')
        print(err)
        exit(1)
    
    return cnx

def select_and_strip_detailed_requirements(cursor):
    """ Select the data from database and strip out the text from html."""
    select_query = 'SELECT challengeId, detailedRequirements FROM challenge_meta;'
    cursor.execute(select_query)

    white_space_regex = r'/s+'
    striped_requirements = []

    for challenge_id, detailed_requirements in cursor:
        print(f'Selecting challenge {challenge_id}')
        striped_requirements.append(
            {
                'challenge_id': str(challenge_id),
                'requirements': ' '.join(BeautifulSoup(detailed_requirements, 'html.parser').get_text().lower().split()),
            }
        )

    print(f'Selected {len(striped_requirements)} challenges.')
    with open(os.path.join(PATH, 'detail_requirements.json'), 'w') as fwrite:
        json.dump(striped_requirements, fwrite, indent=4)

    cursor.close()

if __name__ == '__main__':
    create_data_folder()
    cnx = get_db_cnx()
    select_and_strip_detailed_requirements(cnx.cursor())
    cnx.close()

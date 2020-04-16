""" Get data from database and first round process.

    NOTE: This file is meant to be run in the AWS EC2 instance
          to bypass the GFW
"""

import os
import re
import json
from collections import defaultdict
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

def fmt_date(dt, fmt='%Y-%m-%d'):
    """ Format the date object to string."""
    return dt.strftime(fmt)

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

def get_number_of_challenges_by_project(cursor):
    """ Group the count of challenges by project id."""
    select_query = 'SELECT projectId, COUNT(*) AS numberOfChallenges FROM Challenge WHERE projectId != -1 GROUP BY projectId ORDER BY projectId DESC;'
    cursor.execute(select_query)

    challenge_count_by_project = []
    for project_id, number_of_challenges in cursor:
        print(f'Project {project_id} | {number_of_challenges} challenges')
        challenge_count_by_project.append({'project_id': project_id, 'number_of_challenges': number_of_challenges})

    print(f'{len(challenge_count_by_project)} project in total.')
    with open(os.path.join(PATH, 'number_of_challenges_by_project.json'), 'w') as fwrite:
        json.dump(challenge_count_by_project, fwrite, indent=4)

    cursor.close()

def get_tech_by_start_date(cursor):
    """ Get the technology string by start date."""
    select_query = """
        SELECT technologies, Date(registrationStartDate) AS Dt
        FROM Challenge
        WHERE technologies != ''
        ORDER BY Dt ASC;
    """
    cursor.execute(select_query)

    tech_by_start_date = defaultdict(lambda: defaultdict(int))
    for tech_string, registration_start_date in cursor:
        print(f'Counting tech on {registration_start_date}')
        for tech_term in [tech_term.lower() for tech_term in tech_string.split(', ')]:
            print(f'\t{tech_term}')
            tech_by_start_date[fmt_date(registration_start_date)][tech_term] += 1

    with open(os.path.join(PATH, 'tech_by_start_date.json'), 'w') as fwrite:
        json.dump(tech_by_start_date, fwrite, indent=4)

    cursor.close()

def get_total_prize_of_track_by_date(cursor):
    """ Get a table with track as column and date and index and total prize as value"""
    select_query = """
        SELECT
            DATE(registrationStartDate) as Dt,
            SUM(IF(track = 'DEVELOP', totalPrize, 0)) AS Develop,
            SUM(IF(track = 'DESIGN', totalPrize, 0)) AS Design,
            SUM(IF(track = 'DATA_SCIENCE', totalPrize, 0)) AS Data_Science
        FROM Challenge
        GROUP BY Dt
        ORDER BY Dt ASC;
    """
    cursor.execute(select_query)

    prize_of_track_by_date = []
    for dt, develop, design, data_science in cursor:
        print(f'{dt} | dev {develop} | des {design} | ds {data_science}')
        prize_of_track_by_date.append(
            {
                'date': fmt_date(dt),
                'develop': develop,
                'design': design,
                'data_science': data_science
            }
        )

    print(f'{len(prize_of_track_by_date)} days of data retrieved')
    with open(os.path.join(PATH, 'prize_of_track_by_date.json'), 'w') as fwrite:
        json.dump(prize_of_track_by_date, fwrite, indent=4)

    cursor.close()

def get_number_of_track_by_date(cursor):
    """ Get the number of challenges aggregated by tracks and group by date"""
    select_query = """
        SELECT
            DATE(registrationStartDate) as Dt,
            SUM(IF(track = 'DEVELOP', 1, 0)) AS Develop,
            SUM(IF(track = 'DESIGN', 1, 0)) AS Design,
            SUM(IF(track = 'DATA_SCIENCE', 1, 0)) AS Data_Science
        FROM Challenge
        GROUP BY Dt
        ORDER BY Dt ASC;
    """
    cursor.execute(select_query)

    number_of_track_by_date = []
    for dt, develop, design, data_science in cursor:
        print(f'{dt} | dev {develop} | des {design} | ds {data_science}')
        number_of_track_by_date.append({
            'date': fmt_date(dt),
            'develop': develop,
            'design': design,
            'data_science': data_science
        })

    print(f'{len(number_of_track_by_date)} days of data retrieved')
    with open(os.path.join(PATH, 'number_of_track_by_date.json'), 'w') as fwrite:
        json.dump(number_of_track_by_date, fwrite, indent=4)

    cursor.close()

def get_total_prize_of_dev_subtrack_by_date(cursor):
    """ Get the total prize for sub tracks under DEVELOP track by date."""
    select_query = """
        SELECT
            DATE(registrationStartDate) as Dt,
            SUM(IF(subTrack = 'ARCHITECTURE', totalPrize, 0)) AS ARCHITECTURE,
            SUM(IF(subTrack = 'ASSEMBLY_COMPETITION', totalPrize, 0)) AS ASSEMBLY_COMPETITION,
            SUM(IF(subTrack = 'BUG_HUNT', totalPrize, 0)) AS BUG_HUNT,
            SUM(IF(subTrack = 'CODE', totalPrize, 0)) AS CODE,
            SUM(IF(subTrack = 'CONCEPTUALIZATION', totalPrize, 0)) AS CONCEPTUALIZATION,
            SUM(IF(subTrack = 'CONTENT_CREATION', totalPrize, 0)) AS CONTENT_CREATION,
            SUM(IF(subTrack = 'COPILOT_POSTING', totalPrize, 0)) AS COPILOT_POSTING,
            SUM(IF(subTrack = 'DEVELOP_MARATHON_MATCH', totalPrize, 0)) AS DEVELOP_MARATHON_MATCH,
            SUM(IF(subTrack = 'FIRST_2_FINISH', totalPrize, 0)) AS FIRST_2_FINISH,
            SUM(IF(subTrack = 'SPECIFICATION', totalPrize, 0)) AS SPECIFICATION,
            SUM(IF(subTrack = 'TEST_SCENARIOS', totalPrize, 0)) AS TEST_SCENARIOS,
            SUM(IF(subTrack = 'TEST_SUITES', totalPrize, 0)) AS TEST_SUITES,
            SUM(IF(subTrack = 'UI_PROTOTYPE_COMPETITION', totalPrize, 0)) AS UI_PROTOTYPE_COMPETITION
        FROM Challenge
        GROUP BY Dt
        ORDER BY Dt ASC;
    """
    curosr.execute(select_query)

    prize_of_dev_subtrack_by_dt = []
    for dt, architecture, assembly, bug_hunt, code, conceptual, ctn_creation, copilot, marathon, f2f, spec, test_scenario, test_suite, ui_proto in cursor:
        prize_of_dev_subtrack_by_dt.append({
            'date': fmt_date(dt),
            'architecture': architecture,
            'assembly_competition': assembly,
            'bug_hunt': bug_hunt,
            'code': code,
            'conceptualization': conceptual,
            'content_creation': ctn_creation,
            'copilot_posting': copilot,
            'dev_marathon': marathon,
            'first_to_finish': f2f,
            'specification': spec,
            'test_scenario': test_scenario,
            'test_suite': test_suite,
            'ui_prototype': ui_proto
        })

    print(f'{len(prize_of_dev_subtrack_by_dt)} of dates data')
    with open(os.path.join(PATH, 'prize_of_dev_subtrack_by_dt.json'), 'w') as fwrite:
        json.dump(prize_of_dev_subtrack_by_dt, fwrite, indent=4)

    cursor.close()

def get_number_of_dev_subtrack_by_date(cursor):
    """ Get the number for sub tracks under DEVELOP track by date."""
    select_query = """
        SELECT
            DATE(registrationStartDate) as Dt,
            SUM(IF(subTrack = 'ARCHITECTURE', 1, 0)) AS ARCHITECTURE,
            SUM(IF(subTrack = 'ASSEMBLY_COMPETITION', 1, 0)) AS ASSEMBLY_COMPETITION,
            SUM(IF(subTrack = 'BUG_HUNT', 1, 0)) AS BUG_HUNT,
            SUM(IF(subTrack = 'CODE', 1, 0)) AS CODE,
            SUM(IF(subTrack = 'CONCEPTUALIZATION', 1, 0)) AS CONCEPTUALIZATION,
            SUM(IF(subTrack = 'CONTENT_CREATION', 1, 0)) AS CONTENT_CREATION,
            SUM(IF(subTrack = 'COPILOT_POSTING', 1, 0)) AS COPILOT_POSTING,
            SUM(IF(subTrack = 'DEVELOP_MARATHON_MATCH', 1, 0)) AS DEVELOP_MARATHON_MATCH,
            SUM(IF(subTrack = 'FIRST_2_FINISH', 1, 0)) AS FIRST_2_FINISH,
            SUM(IF(subTrack = 'SPECIFICATION', 1, 0)) AS SPECIFICATION,
            SUM(IF(subTrack = 'TEST_SCENARIOS', 1, 0)) AS TEST_SCENARIOS,
            SUM(IF(subTrack = 'TEST_SUITES', 1, 0)) AS TEST_SUITES,
            SUM(IF(subTrack = 'UI_PROTOTYPE_COMPETITION', 1, 0)) AS UI_PROTOTYPE_COMPETITION
        FROM Challenge
        GROUP BY Dt
        ORDER BY Dt ASC;
    """
    curosr.execute(select_query)

    number_of_dev_subtrack_by_dt = []
    for dt, architecture, assembly, bug_hunt, code, conceptual, ctn_creation, copilot, marathon, f2f, spec, test_scenario, test_suite, ui_proto in cursor:
        number_of_dev_subtrack_by_dt.append({
            'date': fmt_date(dt),
            'architecture': architecture,
            'assembly_competition': assembly,
            'bug_hunt': bug_hunt,
            'code': code,
            'conceptualization': conceptual,
            'content_creation': ctn_creation,
            'copilot_posting': copilot,
            'dev_marathon': marathon,
            'first_to_finish': f2f,
            'specification': spec,
            'test_scenario': test_scenario,
            'test_suite': test_suite,
            'ui_prototype': ui_proto
        })

    print(f'{len(number_of_dev_subtrack_by_dt)} of dates data')
    with open(os.path.join(PATH, 'number_of_dev_subtrack_by_dt.json'), 'w') as fwrite:
        json.dump(number_of_dev_subtrack_by_dt, fwrite, indent=4)

    cursor.close()

def main():
    """ Main entrance"""
    create_data_folder()
    cnx = get_db_cnx()
    get_number_of_challenges_by_project(cnx.cursor())
    get_tech_by_start_date(cnx.cursor())
    get_total_prize_of_track_by_date(cnx.cursor())
    get_number_of_track_by_date(cnx.cursor())
    get_total_prize_of_dev_subtrack_by_date(cnx.cursor())
    get_number_of_dev_subtrack_by_date(cnx.cursor())
    cnx.close()

if __name__ == '__main__':
    main()

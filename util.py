""" Some utility funcitons."""
import os
import json

from tc_main import TopCoder

def clean_tech_lst(top_n=5):
    """ Clean up technology list of challenges."""
    tc = TopCoder() # trying to reduce the length of variable name here
    filt_cha_info = tc.get_filtered_challenge_basic_info() # it's readable for me anyway ;-)
    most_popular_tech = tc.get_tech_popularity().head(top_n).tech_name.to_list()

    with open(os.path.join(os.curdir, 'data', 'tech_by_challenge.json')) as f:
        tech_by_cha_rough = {cha['challenge_id']: cha['tech_lst'] for cha in json.load(f) if cha['challenge_id'] in filt_cha_info.index}

    print(f'Top {top_n} most popular technologies', most_popular_tech)

    tech_by_cha = []
    for cha_id, tech_lst in tech_by_cha_rough.items():
        cleaned_tech_lst = ['angularjs' if 'angular' in tech.lower() else tech.lower() for tech in tech_lst]
        filtered_tech_lst = [tech for tech in cleaned_tech_lst if tech in most_popular_tech]
        if filtered_tech_lst:
            tech_by_cha.append({
                'challenge_id': cha_id,
                'tech_lst': filtered_tech_lst
            })

    print(f'Challenge with tech after filtering: {len(tech_by_cha)}')

    with open(os.path.join(os.curdir, 'data', 'tech_by_challenge_clean.json'), 'w') as f:
        json.dump(tech_by_cha, f, indent=4)

if __name__ == "__main__":
    clean_tech_lst()

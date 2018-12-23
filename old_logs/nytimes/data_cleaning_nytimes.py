import glob
import json
import re
from os.path import basename

import pandas as pd


def _is_keywords_in_string(keywords, s):
    if s is None:
        return False
    return any(re.search(r"\b" + re.escape(x) + r"\b", s) for x in keywords)


def clean_data():
    free_text_keywords = ['natural gas', 'oil', 'coal']
    exclude_sections = ['Corrections', 'Opinion', 'Arts', 'Real Estate', 'Education', 'Week in Review', "N.Y. / Region"]
    exclude_news_desk = ['Travel', 'Sport', 'OpEd', 'Classified', 'Sports', 'Books', 'Movies', 'BookReview', 'Metro']
    exclude_kicker = ['Dot Earth']
    include_keywords = ["Alternative and Renewable Energy", "Oil (Petroleum) and Gasoline", "Natural Gas",
                        "Energy and Power"]
    exclude_type_of_material = ["Blog", 'Review', 'Video']
    exclude_subsection_name = ["Middle East"]
    for i in glob.glob('raw/*.json'):
        print(i)
        processed = []
        with open(i, encoding='utf8') as infile:
            month_news = json.loads(infile.read())
            for article in month_news:
                # absolutely weed these out
                if 'headline' in article and len(article['headline']) > 0:
                    if 'kicker' in article['headline']:
                        if _is_keywords_in_string(exclude_kicker, article['headline']['kicker']):
                            continue
                if 'new_desk' in article and _is_keywords_in_string(exclude_news_desk, article['new_desk']):
                    continue
                if 'news_desk' in article and _is_keywords_in_string(exclude_news_desk, article['news_desk']):
                    continue
                if 'section_name' in article and _is_keywords_in_string(exclude_sections, article['section_name']):
                    continue
                if "type_of_material" in article and _is_keywords_in_string(exclude_type_of_material,
                                                                            article["type_of_material"]):
                    continue
                if 'subsection_name' in article and _is_keywords_in_string(exclude_subsection_name,
                                                                           article['subsection_name']):
                    continue
                # time to include
                # might got mentioned at the end of the articles as not the main object of the news
                # enforce interested keywords to be in lead paragraph
                if any(d['value'] in include_keywords for d in article['keywords']):
                    if 'lead_paragraph' in article and _is_keywords_in_string(free_text_keywords,
                                                                              article['lead_paragraph']):
                        processed.append(article)
                        continue

        with open('processed_%s' % basename(i), 'w', encoding='utf-8') as outfile:
            json.dump(processed, outfile)


print('Clean data bgein')
clean_data()
print('Clean data finished')
news = pd.DataFrame()
for i in glob.glob('*.json'):
    print(i)
    news = news.append(pd.read_json(i))
print('concatenate news completed')
news = news.set_index('pub_date')
news['info'] = news['abstract'].fillna(news['lead_paragraph']).fillna(news['snippet'])
# news['info'] = [x['main'] for x in news.headline.values]
news = news[['info', 'abstract', 'headline', 'lead_paragraph', 'snippet', 'keywords']]
news.dropna(inplace=True, subset=['info'])
# news['info'].str.split('.')[0]
news.to_csv('data.csv')

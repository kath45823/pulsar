from preprocessor import fetch_papers
from impact_factor.core import Factor
from datetime import date, datetime
import numpy as np

papers = fetch_papers()
scores = []
fa = Factor()
scored_papers = []

for paper in papers:
    # Scoring 
    results = fa.search(paper["journal"])
    if not results: 
        continue
    
    jif = results[0]["factor"]

    p_date = paper["publication_date"]
    if "Year" not in p_date:
        continue
    
    if "Month" in p_date:
        try: 
            month_num = datetime.strptime(p_date["Month"], "%B").month
        except: 
            try: 
                month_num = int(p_date["Month"])
            except:
                month_num = 1
    else: 
        month_num = 1

    if "Day" in p_date:
        day = int(p_date["Day"])
    else: 
        day = 1
    
    pub_date = date(int(p_date["Year"]), month_num, day)
    days_since_pub = (datetime.now().date() - pub_date).days

    score = 0.7 * jif + 0.3 * (1 / (days_since_pub + 1))   # maybe normalize later + experiment with weights 
    paper["score"] = score
    scores.append(score)
    scored_papers.append(paper)

threshold = np.percentile(scores, 70)   # experiment on percentile later
highest_scores_papers = []

for paper in scored_papers: 
    score = paper["score"]

    if score >= threshold:
        highest_scores_papers.append(paper)
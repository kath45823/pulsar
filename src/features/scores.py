from preprocessor import fetch_papers
from impact_factor.core import Factor
from datetime import date, datetime
import numpy as np

papers = fetch_papers()
highest_scores_papers = []

for paper in papers:
    # Scoring 
    fa = Factor()
    results = fa.search(paper["journal"])
    jif = results[0]["factor"]

    p_date = paper["publication_date"]
    month_num = datetime.strptime(p_date["Month"], "%B").month
    pub_date = date(int(p_date["Year"]), month_num, int(p_date["Day"]))
    days_since_pub = (datetime.now() - pub_date).days

    score = 0.7 * jif + 0.3 * (1/days_since_pub)    # maybe normalize later + experiment with weights 
    paper["score"] = score




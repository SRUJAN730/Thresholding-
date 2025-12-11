import numpy as np
import pandas as pd
import argparse

def generate_synthetic_hiring(n=5000, seed=42, penalty_race=None, penalty_gender=None):
    np.random.seed(seed)
    gender = np.random.choice(['Male','Female'], size=n, p=[0.5,0.5])
    race = np.random.choice(['White','Black','Hispanic','Asian'], size=n, p=[0.55,0.2,0.15,0.1])
    years_experience = np.round(np.random.normal(loc=5, scale=3, size=n).clip(0),1)
    education_level = np.random.choice(['HS','Bachelors','Masters','PhD'], size=n, p=[0.15,0.55,0.25,0.05])
    skill_score = np.round(np.clip(np.random.normal(loc=70, scale=15, size=n),0,100),1)
    interview_score = np.round(np.clip(np.random.normal(loc=70, scale=12, size=n),0,100),1)
    base_score = 0.4*skill_score + 0.4*interview_score + 0.2*(years_experience*10)
    penalty = np.zeros(n)
    if penalty_race is None:
        penalty_race = {'Black': -5}
    if penalty_gender is None:
        penalty_gender = {'Female': -2}
    for r,p in penalty_race.items():
        penalty += np.where(race==r, p, 0)
    for g,p in penalty_gender.items():
        penalty += np.where(gender==g, p, 0)
    logits = (base_score + penalty - 70) / 10
    prob = 1 / (1 + np.exp(-logits))
    hire = np.random.binomial(1, prob)
    edu_map = {'HS':0,'Bachelors':1,'Masters':2,'PhD':3}
    df = pd.DataFrame({
        'gender': gender,
        'race': race,
        'years_experience': years_experience,
        'education': education_level,
        'education_num': [edu_map[e] for e in education_level],
        'skill_score': skill_score,
        'interview_score': interview_score,
        'base_score': base_score,
        'penalty': penalty,
        'hire': hire
    })
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/synthetic_hiring_small.csv')
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    df = generate_synthetic_hiring(n=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    print('Saved', args.out, 'with', len(df), 'rows')
import os
import random
import pandas as pd
import numpy as np

os.makedirs('data_engineer/sample_data', exist_ok=True)

random.seed(42)
np.random.seed(42)

n_samples = 600

jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
        'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
job_weights = [0.11, 0.21, 0.03, 0.03, 0.21, 0.05, 0.03, 0.09, 0.02, 0.17, 0.03, 0.02]

maritals = ['married', 'single', 'divorced']
marital_weights = [0.60, 0.28, 0.12]

educations = ['primary', 'secondary', 'tertiary', 'unknown']
edu_weights = [0.15, 0.51, 0.29, 0.05]

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_weights = [0.03, 0.06, 0.01, 0.06, 0.31, 0.12, 0.15, 0.14, 0.01, 0.02, 0.08, 0.01]

contacts = ['cellular', 'telephone', 'unknown']
contact_weights = [0.65, 0.06, 0.29]

poutcomes = ['unknown', 'failure', 'other', 'success']
poutcome_weights = [0.82, 0.11, 0.04, 0.03]

records = []
for i in range(1, n_samples + 1):
    job = random.choices(jobs, weights=job_weights)[0]
    if job == 'retired':
        age = int(np.random.normal(64, 5))
    elif job == 'student':
        age = int(np.random.normal(23, 3))
    else:
        age = int(np.random.normal(40, 10))
    age = max(18, min(95, age))

    marital = random.choices(maritals, weights=marital_weights)[0]
    education = random.choices(educations, weights=edu_weights)[0]
    default = 'yes' if random.random() < 0.018 else 'no'

    # Balance distribution
    base_balance = np.random.exponential(1400) - 100
    if job in ['management', 'retired']:
        base_balance += np.random.uniform(500, 3000)
    if default == 'yes':
        base_balance = -abs(int(np.random.uniform(10, 800)))
    balance = int(base_balance)

    housing = 'yes' if (random.random() < 0.55 and age < 60) else 'no'
    loan = 'yes' if (random.random() < 0.16 and default == 'no') else 'no'
    contact = random.choices(contacts, weights=contact_weights)[0]
    day = random.randint(1, 31)
    month = random.choices(months, weights=month_weights)[0]
    
    poutcome = random.choices(poutcomes, weights=poutcome_weights)[0]
    if poutcome == 'unknown':
        pdays = -1
        previous = 0
    else:
        pdays = random.randint(1, 400)
        previous = random.randint(1, 7)
    
    campaign = max(1, int(np.random.exponential(2.5)))
    duration = max(5, int(np.random.exponential(250)))

    # Compute realistic conversion probability for synthetic data
    logit = -2.8
    logit += 0.004 * min(duration, 1000)
    if poutcome == 'success':
        logit += 2.2
    if balance > 3000:
        logit += 0.8
    elif balance < 0:
        logit -= 0.6
    if housing == 'yes':
        logit -= 0.5
    if loan == 'yes':
        logit -= 0.4
    if job in ['retired', 'student']:
        logit += 0.6
    if campaign > 5:
        logit -= 0.5

    prob = 1.0 / (1.0 + np.exp(-logit))
    y = 'yes' if random.random() < prob else 'no'

    records.append({
        'customer_id': f'CUST_{i:05d}',
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'y': y
    })

df = pd.DataFrame(records)
df.to_csv('data_engineer/sample_data/bank_raw_sample.csv', index=False)
print(f"Generated {len(df)} sample records with conversion rate: {(df['y'] == 'yes').mean():.2%}")

# Generate sample leads to score
leads = df.drop(columns=['y']).sample(100, random_state=123).copy()
leads['customer_id'] = [f'LEAD_{i:05d}' for i in range(1, 101)]
leads.to_csv('data_engineer/sample_data/leads_to_score.csv', index=False)
print("Generated 100 leads to score.")

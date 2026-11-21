import sys, json, math, os, time
from pathlib import Path
sys.path.insert(0,'.')
import pandas as pd, numpy as np, joblib
from fastapi.testclient import TestClient
from cortexhr_api.main import create_app
X=pd.read_excel(os.environ['CORTEXHR_WORKBOOK'],sheet_name=None)
def rec(df,n,drop=()):
    out=[]
    for _,r in df.head(n).iterrows():
        d={k:(None if (isinstance(v,float) and math.isnan(v)) else (v.item() if hasattr(v,'item') else (str(v) if hasattr(v,'isoformat') else v))) for k,v in r.items() if k not in drop and not (isinstance(v,float) and math.isnan(v))}
        out.append(d)
    return out
res=[]
def log(svc,ep,method,result,verified,detail=''): res.append(dict(service=svc,endpoint=ep,test=method,result=result,verified=verified,detail=str(detail)[:400])); print(svc,ep,'|',result,'|',str(detail)[:160])
with TestClient(create_app()) as c:
    H={'X-API-Key':'testkey'}
    r=c.get('/v1/health'); log('platform','GET /v1/health','local TestClient',r.status_code,'local only',r.json()['models_available'])
    r=c.get('/v1/ready',headers=H); log('platform','GET /v1/ready','local TestClient (expected 503: forecast not loadable)',r.status_code,'local only')
    # auth
    for k,hd in [('no key',{}),('bad key',{'X-API-Key':'nope'})]:
        r=c.post('/v1/chat',json={'message':'hi'},headers=hd); log('security','POST /v1/chat',k,r.status_code,'local only')
    # attrition
    sc=pd.read_csv(os.environ['SCORING_CSV'])
    meta=json.load(open(os.environ['CORTEXHR_ARTIFACT_DIR']+'/attrition-v2.metadata.json'))
    recs=rec(sc,5,drop=('row_id','employee_id','observation_date','source','left_within_12m'))
    r=c.post('/v1/attrition/predict',json={'records':recs},headers=H)
    j=r.json(); log('attrition','POST /v1/attrition/predict','5 real scoring rows',r.status_code,'local only',json.dumps(j)[:300])
    if r.status_code==200:
        api=[x.get('probability',x.get('risk_score')) for x in j['results']] if 'results' in j else None
        print('API probs',api)
        # compare with official sample_attrition_risk
        s=X['sample_attrition_risk']; print('sample cols',list(s.columns)); 
    # resume
    jd=X['job_descriptions'].iloc[0]; rs=X['resumes'].head(6)
    body={'job':{'job_id':str(jd['job_id']),'job_title':str(jd['job_title']),'description_text':str(jd.get('description_text','')),'required_skills':str(jd.get('required_skills','')),'nice_to_have_skills':str(jd.get('nice_to_have_skills','')),'min_experience_years':float(jd.get('min_experience_years',0) or 0)},'resumes':[{'resume_id':str(a),'resume_text':str(b)} for a,b in zip(rs['resume_id'],rs['resume_text'])]}
    r=c.post('/v1/resume/rank',json=body,headers=H); log('resume','POST /v1/resume/rank','1 job x 6 resumes',r.status_code,'local only',json.dumps(r.json())[:260])
    # payband
    s=pd.read_csv(os.environ['CORTEXHR_SALARY_CSV']).iloc[0]
    pb={k:(s[k].item() if hasattr(s[k],'item') else s[k]) for k in ['job_level','prior_experience_years','market_index','role_id','department_id','work_location','highest_education']}
    r=c.post('/v1/payband/analyze',json=pb,headers=H); log('payband','POST /v1/payband/analyze','1 real hire row',r.status_code,'local only',json.dumps(r.json())[:300])
    # anomaly
    lab5=pd.read_csv(os.environ['UNLOCK_CX05']) if os.environ.get('UNLOCK_CX05') else None
    at=X['attendance_monthly_features'];     if lab5 is not None:
        sel=pd.concat([at[at.attendance_feature_id.isin(lab5.attendance_feature_id)].head(3),at[~at.attendance_feature_id.isin(lab5.attendance_feature_id)].head(3)])
    else:
        sel=at.head(6)
    r=c.post('/v1/anomaly/attendance',json={'records':rec(sel,6)},headers=H); log('attendance_anomaly','POST /v1/anomaly/attendance','3 injected + 3 unlabelled rows',r.status_code,'local only',json.dumps(r.json())[:300])
    lab6=pd.read_csv(os.environ['UNLOCK_CX06']) if os.environ.get('UNLOCK_CX06') else None
    pr=X['payroll_monthly_features']
    if lab6 is not None:
        sel=pd.concat([pr[pr.payroll_item_id.isin(lab6.payroll_item_id)].head(3),pr[~pr.payroll_item_id.isin(lab6.payroll_item_id)].head(3)])
    else:
        sel=pr.head(6)
    r=c.post('/v1/anomaly/payroll',json={'records':rec(sel,6)},headers=H); log('payroll_anomaly','POST /v1/anomaly/payroll','3 injected + 3 unlabelled rows',r.status_code,'local only',json.dumps(r.json())[:300])
    # chat
    for q in ['How many casual leaves do I get in a year?','what is the notice period policy','tell me a joke']:
        r=c.post('/v1/chat',json={'message':q},headers=H); j=r.json(); log('chatbot','POST /v1/chat',q,r.status_code,'local only',{k:j.get(k) for k in ['intent','confidence','fallback']})
    # forecast (expected unavailable)
    r=c.post('/v1/forecast/headcount',json={'department_id':'D01','horizon_months':6},headers=H); log('forecast','POST /v1/forecast/headcount','expected unavailable (Task 16 code missing)',r.status_code,'blocked',r.text[:120])
    # negative / security
    r=c.post('/v1/chat',content=b'{bad json',headers={**H,'Content-Type':'application/json'}); log('security','POST /v1/chat','malformed JSON',r.status_code,'local only',r.text[:100])
    r=c.post('/v1/chat',json={'message':'x'*200000},headers=H); log('security','POST /v1/chat','200 KB body',r.status_code,'local only',r.text[:100])
    r=c.post('/v1/payband/analyze',json={'job_level':99,'role_id':'R001'},headers=H); log('security','POST /v1/payband/analyze','out-of-range job_level',r.status_code,'local only',r.text[:120])
    r=c.post('/v1/attrition/predict',json={'records':[]},headers=H); log('security','POST /v1/attrition/predict','empty batch',r.status_code,'local only')
    r=c.get('/v1/monitoring',headers=H); log('monitoring','GET /v1/monitoring','no report published yet',r.status_code,'local only',r.text[:200])
    r=c.get('/docs'); log('platform','GET /docs','docs enabled in development',r.status_code,'local only')
out=Path(os.environ.get('CORTEXHR_EVIDENCE_OUT','docs/evidence/service_verification.json'))
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(res,open(out,'w'),indent=1)

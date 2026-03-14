const API = "https://medipredict-ai-2dgd.onrender.com";

function getSession(){try{return JSON.parse(sessionStorage.getItem("pms_user")||"null")}catch{return null}}
function setSession(u){sessionStorage.setItem("pms_user",JSON.stringify(u))}
function clearSession(){sessionStorage.removeItem("pms_user")}
function requireAuth(){if(!getSession())window.location.href="login.html"}

function updateNav(){
  const u=getSession();if(!u)return;
  const nu=document.getElementById("navUser");if(nu)nu.textContent=u.name;
  const na=document.getElementById("navAvatar");if(na)na.textContent=u.name[0].toUpperCase();
}

function logout(){clearSession();window.location.href="login.html"}

function showToast(msg,type="info"){
  let t=document.getElementById("toast");
  if(!t){t=document.createElement("div");t.id="toast";t.className="toast";document.body.appendChild(t)}
  const ic={info:"ℹ️",success:"✅",error:"❌",warning:"⚠️"};
  t.innerHTML=`${ic[type]||"ℹ️"} ${msg}`;t.classList.add("show");
  setTimeout(()=>t.classList.remove("show"),3500);
}

// Step form
let currentStep=0,totalSteps=0;
function showStep(n){
  document.querySelectorAll(".form-step").forEach((s,i)=>s.classList.toggle("active",i===n));
  document.querySelectorAll(".step-btn").forEach((b,i)=>{
    b.classList.toggle("active",i===n);b.classList.toggle("done",i<n);
  });
  const bp=document.getElementById("btnPrev"),bn=document.getElementById("btnNext"),bs=document.getElementById("btnSubmit");
  if(bp)bp.style.display=n===0?"none":"inline-flex";
  if(bn)bn.style.display=n<totalSteps-1?"inline-flex":"none";
  if(bs)bs.style.display=n===totalSteps-1?"inline-flex":"none";
}
function nextStep(){if(validateStep(currentStep)){currentStep++;showStep(currentStep)}}
function prevStep(){if(currentStep>0){currentStep--;showStep(currentStep)}}
function validateStep(n){
  const step=document.querySelectorAll(".form-step")[n];if(!step)return true;
  const inputs=step.querySelectorAll("input[required],select[required]");let ok=true;
  inputs.forEach(i=>{if(!i.value.trim()){i.style.borderColor="var(--heart)";ok=false;setTimeout(()=>i.style.borderColor="",2000)}});
  if(!ok)showToast("Fill in all required fields","warning");return ok;
}

// API
async function predictHeart(d){
  const r=await fetch(`${API}/predict-heart`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)});
  if(!r.ok)throw new Error(`Server error ${r.status}`);return r.json();
}
async function predictLiver(d){
  const r=await fetch(`${API}/predict-liver`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)});
  if(!r.ok)throw new Error(`Server error ${r.status}`);return r.json();
}
async function predictDiabetes(d){
  const r=await fetch(`${API}/predict-diabetes`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)});
  if(!r.ok)throw new Error(`Server error ${r.status}`);return r.json();
}

function renderResult(result,accentVar){
  const rc=document.getElementById("resultCard");if(!rc)return;
  const pct=Math.round(result.probability*100);
  const riskClass={"High":"risk-high","Moderate":"risk-moderate","Low":"risk-low"}[result.risk_level];
  const isDanger=result.risk_level==="High";
  rc.innerHTML=`
  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:16px">
      <h3 style="font-family:var(--font-d);font-size:1.1rem;font-weight:800;letter-spacing:-0.02em">🧾 AI Medical Report</h3>
      <span class="risk-pill ${riskClass}">${result.risk_level} Risk</span>
    </div>
    <p style="font-size:0.875rem;color:var(--text2);margin-bottom:16px">${result.prediction_label}</p>
    <div class="prob-wrap">
      <div class="prob-header"><span>Disease Probability</span><span style="font-family:var(--font-d);font-weight:800;color:${accentVar}">${pct}%</span></div>
      <div class="prob-track"><div class="prob-fill ${isDanger?"danger":""}" id="probFill"></div></div>
    </div>
    <div class="result-section"><div class="rs-title">🥗 Diet Recommendations</div>
      <ul class="result-list">${result.diet_recommendations.map(d=>`<li>${d}</li>`).join("")}</ul></div>
    <div class="result-section"><div class="rs-title">🏃 Lifestyle Suggestions</div>
      <ul class="result-list">${result.lifestyle_suggestions.map(s=>`<li>${s}</li>`).join("")}</ul></div>
    <div class="result-section"><div class="rs-title">💊 Medication Suggestions</div>
      <ul class="result-list">${result.medication_suggestions.map(m=>`<li>${m}</li>`).join("")}</ul></div>
    <div class="consult-banner ${result.doctor_consultation_recommended?"consult-yes":"consult-no"}">
      ${result.doctor_consultation_recommended
        ?"🔴 <strong>Doctor consultation recommended</strong> — Please visit a specialist for evaluation."
        :"🟢 <strong>No immediate consultation needed</strong> — Maintain your healthy habits."}
    </div>
    <p style="margin-top:16px;font-size:0.72rem;color:var(--text3);padding:12px 14px;background:var(--surface2);border-radius:10px">
      ⚠️ This AI report is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional.
    </p>
  </div>`;
  rc.classList.add("show");
  setTimeout(()=>{const f=document.getElementById("probFill");if(f)f.style.width=pct+"%"},120);
  rc.scrollIntoView({behavior:"smooth",block:"start"});
}

function animateCounter(el,target,dur=1200){
  if(!el)return;const start=performance.now();
  function step(now){const t=Math.min((now-start)/dur,1);const e=1-Math.pow(1-t,3);
    el.textContent=Math.round(e*target);if(t<1)requestAnimationFrame(step)}
  requestAnimationFrame(step);
}

// Scroll reveal
function initReveal(){
  const obs=new IntersectionObserver(entries=>{
    entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add("visible");obs.unobserve(e.target)}})
  },{threshold:0.12});
  document.querySelectorAll(".reveal").forEach(el=>obs.observe(el));
}

document.addEventListener("DOMContentLoaded",()=>{updateNav();initReveal()});

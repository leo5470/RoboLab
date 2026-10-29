/* Dataset membership comes exclusively from the published 50-demo manifests. */
(() => {
  const $ = s => document.querySelector(s), params = new URLSearchParams(location.search);
  const state = {session: params.get('session') || '', kind: params.get('kind') || '', layout: params.get('layout') || '',
    q: params.get('q') || '', offset: Math.max(0, Number(params.get('offset')) || 0), limit: 24,
    demo: params.get('demo'), datasets: [], episodes: [], total: 0, request: 0, viewer: 0};
  let previewTimer, summarySession;
  const human = s => String(s || '').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
  function node(tag, text, cls) {const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n;}
  async function api(path, options) {const r = await fetch(path, options); if (!r.ok) throw new Error((await r.text()).slice(0, 250)); return r.json();}
  function showError(e) {$('#errors').textContent = e.message;}
  function url() {const p = new URLSearchParams(); ['session','kind','layout','q','offset','demo'].forEach(k => {if(state[k]) p.set(k,state[k]);}); history.replaceState(null,'',`${location.pathname}?${p}`);}
  function choose(session) {state.session = session; state.layout = ''; state.offset = 0; load().catch(showError);}
  function badge(text, cls='') {return node('span',text,`badge ${cls}`);}
  function link(text, href) {const a = node('a',text); a.href = href; return a;}
  function renderSummary() {
    $('#tasks').replaceChildren(); $('#all-datasets').classList.toggle('active',!state.session);
    state.datasets.forEach(d => {const b = node('button',null,`task-button ${state.session === d.session ? 'active' : ''}`);
      b.append(node('span',human(d.family)),node('small',d.selected_count)); b.title = d.session; b.onclick = () => choose(d.session); $('#tasks').append(b);});
    const selected = state.datasets.filter(d => !state.session || d.session === state.session);
    const sum = k => selected.reduce((n,d) => n+d[k],0);
    $('#export-summary').textContent = `Family export status · ${selected.filter(d=>d.export_status==='complete').length}/${selected.length} training files complete`;
    if(summarySession !== state.session) {$('#export-panel').open = Boolean(state.session); summarySession = state.session;}
    $('#title').textContent = state.session && selected.length ? human(selected[0].family) : 'Training dataset';
    $('#stats').replaceChildren();
    [[sum('selected_count'),'Selected demos'],[sum('collected'),'Human demos'],[sum('generated'),'Generated demos'],[sum('cameras_ready'),'Camera episodes ready']].forEach(([v,label]) => {
      const n = node('div',null,'stat'); n.append(node('strong',v),node('span',label)); $('#stats').append(n);
    });
    $('#exports').replaceChildren(...selected.map(d => {
      const c = node('div',null,'export'), b = node('button',human(d.family),'export-button'); b.onclick = () => choose(d.session); c.append(b);
      c.append(node('p',`${d.collected} human + ${d.generated} generated · ${d.selected_count} demos`));
      const labels = {complete:'Training file complete',processing:'Preparing camera export',pending:'Waiting for camera export',error:'Export needs attention'};
      c.append(badge(labels[d.export_status],d.export_status === 'error' ? 'warn' : 'neutral'));
      if(d.replacement_status && d.replacement_status !== 'complete') c.append(node('p',`Multisource refresh: ${human(d.replacement_status)}`));
      if(d.generation_mode === 'multi_source') c.append(node('p',`Generated from ${d.qualified_human_sources} qualified human sources`));
      const progress = node('progress'); progress.max=d.selected_count; progress.value=d.cameras_ready; progress.setAttribute('aria-label',`${human(d.family)} camera preparation`); c.append(progress);
      c.append(node('p',`${d.cameras_ready}/${d.selected_count} camera episodes available`));
      c.append(node('p',d.session,'session-name'));
      c.append(link('Selection manifest',d.manifest));
      if(d.download) c.append(link('Training HDF5',d.download));
      if(d.export_error) c.append(node('p',d.export_error,'notice'));
      return c;
    }));
    document.querySelectorAll('[data-kind]').forEach(b => b.classList.toggle('selected',state.kind === b.dataset.kind));
  }
  function card(ep) {
    const c = node('button',null,'card'); c.type='button'; c.setAttribute('aria-label',`View ${human(ep.family)} demo ${ep.number}, ${ep.kind}`);
    const thumb = node('div',null,'thumb');
    if(ep.poster) {const img=node('img'); img.src=ep.poster; img.loading='lazy'; img.alt=`${human(ep.family)} policy cameras`; img.onerror=()=>{img.remove();thumb.prepend(node('span','Preview unavailable','muted'));}; thumb.append(img);}
    else thumb.append(node('div','Cameras pending','muted'));
    if(ep.video || ep.cameras_ready) thumb.append(node('div','▶','play'));
    thumb.append(node('span',`${ep.duration_s.toFixed(1)}s`));
    const body=node('div',null,'card-body'); body.append(badge(ep.kind === 'collected' ? 'HUMAN' : 'GENERATED',ep.kind === 'generated' ? 'generated' : ''));
    body.append(node('div',`${human(ep.family)} · Demo ${ep.number}`,'card-title'),node('div',`Layout ${ep.layout} · ${ep.sample_count} steps`,'card-meta'),
      node('div',ep.task,'card-meta'),badge(ep.training_group ? 'In training file' : ep.cameras_ready ? 'Cameras ready' : 'Cameras pending','neutral'));
    c.append(thumb,body); c.onclick=()=>open(ep.id).catch(showError); return c;
  }
  async function load() {
    const version=++state.request; url();
    const p=new URLSearchParams({limit:state.limit,offset:state.offset}); ['session','kind','layout','q'].forEach(k=>{if(state[k])p.set(k,state[k]);});
    const [summary,data]=await Promise.all([api('/api/datasets'),api('/api/datasets/episodes?'+p)]);
    if(version!==state.request)return;
    state.datasets=summary.datasets; state.episodes=data.episodes; state.total=data.total;
    renderSummary(); $('#grid').replaceChildren(...data.episodes.map(card));
    if(!data.episodes.length)$('#grid').append(node('div','No selected demos match these filters.','empty'));
    $('#result-count').textContent=`${data.total} selected demos`; $('#updated').textContent=`Updated ${new Date(summary.updated*1000).toLocaleTimeString()}`;
    $('#errors').textContent=summary.warnings.join(' · ');
    $('#page-count').textContent=`Page ${Math.floor(state.offset/state.limit)+1} of ${Math.max(1,Math.ceil(data.total/state.limit))}`;
    $('#previous').disabled=state.offset===0; $('#next').disabled=state.offset+state.limit>=data.total;
    const values=new Set(summary.datasets.filter(d=>!state.session||state.session===d.session).flatMap(d=>d.layouts || []));
    $('#layout').replaceChildren(new Option('All layouts','')); [...values].sort().forEach(v=>$('#layout').add(new Option(v,v))); $('#layout').value=state.layout;
  }
  async function open(id) {
    clearTimeout(previewTimer); const version=++state.viewer;
    const ep=await api('/api/datasets/episodes/'+id); if(version!==state.viewer)return;
    state.demo=id; url(); const dialog=$('#viewer'), body=$('#viewer-body'); body.querySelectorAll('video').forEach(v=>v.pause()); body.replaceChildren();
    $('#viewer-origin').textContent=ep.kind==='collected'?'HUMAN / SELECTED FOR TRAINING':'MIMICGEN / SELECTED FOR TRAINING';
    $('#viewer-title').textContent=`${human(ep.family)} · Demo ${ep.number} · Layout ${ep.layout}`;
    body.append(node('p','Shoulder / wrist cameras · 15 frames per second','player-label'));
    const player=node('div',null,'players'), status=node('p',null,'preview-message'); body.append(player,status);
    function video(src) {player.replaceChildren(); const v=node('video');v.controls=true;v.preload='metadata';v.src=src;player.append(v);status.textContent='';
      const label=node('label','Playback speed '),speed=node('select');[.25,.5,1,1.5,2].forEach(n=>speed.add(new Option(`${n}×`,n)));speed.value='1';speed.onchange=()=>v.playbackRate=Number(speed.value);label.append(speed);player.append(label);}
    if(ep.video)video(ep.video);
    else if(ep.frame) {
      const image=node('img');image.className='frame-preview';image.alt='Selected demo shoulder and wrist cameras';image.src=ep.frame;player.append(image);
      const control=node('div',null,'frame-control'),slider=node('input'),count=node('span',`1 / ${ep.sample_count}`);slider.type='range';slider.min=0;slider.max=ep.sample_count-1;slider.value=0;slider.setAttribute('aria-label','Recorded frame');
      let debounce;slider.oninput=()=>{count.textContent=`${Number(slider.value)+1} / ${ep.sample_count}`;clearTimeout(debounce);debounce=setTimeout(()=>{if(version===state.viewer)image.src=ep.frame+'?step='+slider.value;},80);};control.append(slider,count);player.append(control);
      status.textContent='Preparing video from the camera frames… You can inspect individual frames while it prepares.';
      async function poll(start=false) {
        if(version!==state.viewer || !dialog.open)return;
        try {const result=await api(`/api/datasets/episodes/${id}${start?'/preview':''}`,start?{method:'POST'}:undefined);
          if(version!==state.viewer || !dialog.open)return;
          if(result.video)video(result.video);
          else if(result.preview_status==='failed') {status.textContent='Video preparation failed. ';const retry=node('button','Retry');retry.onclick=()=>poll(true);status.append(retry);}
          else previewTimer=setTimeout(()=>poll(),2000);
        } catch(e) {if(version===state.viewer){status.textContent='Preview could not be prepared. ';const retry=node('button','Retry');retry.onclick=()=>poll(true);status.append(retry);}}
      }
      previewTimer=setTimeout(()=>poll(true),0);
    } else {
      status.textContent='Camera export is pending for this selected demo. This viewer updates when the frames are available.';
      previewTimer=setTimeout(()=>{if(dialog.open && state.demo===id)open(id).catch(showError);},15000);
    }
    const tools=node('div',null,'player-tools');body.append(tools);
    if(ep.source_url)tools.append(link('View human source',ep.source_url));
    const copy=node('button','Copy link');copy.onclick=async()=>{try{await navigator.clipboard.writeText(location.href);copy.textContent='Link copied';}catch{copy.textContent='Use the address bar to copy this link';}};tools.append(copy);
    const index=state.episodes.findIndex(e=>e.id===id);
    for(const [label,delta] of [['← Previous',-1],['Next →',1]]) {const target=state.offset+index+delta,b=node('button',label);b.disabled=index<0||target<0||target>=state.total;
      b.onclick=async()=>{try{state.offset=Math.floor(target/state.limit)*state.limit;await load();await open(state.episodes[target-state.offset].id);}catch(e){showError(e);}};tools.append(b);}
    const meta=node('div',null,'metadata');
    [['Task',ep.task],['Session',ep.session],['Origin',ep.kind==='collected'?'Human':'Generated'],['Steps',ep.sample_count],['Training episode',ep.training_group || 'Export pending'],['Trajectory ID',ep.fingerprint]].forEach(([k,v])=>{const n=node('div');n.append(node('small',k),node('span',v,'meta-value'));meta.append(n);});body.append(meta);
    if(!dialog.open)dialog.showModal();
  }
  $('#viewer').addEventListener('close',()=>{clearTimeout(previewTimer);state.viewer++;$('#viewer-body').querySelectorAll('video').forEach(v=>v.pause());state.demo=null;url();});
  $('#close-viewer').onclick=()=>$('#viewer').close();$('#viewer').onclick=e=>{if(e.target===$('#viewer'))$('#viewer').close();};
  $('#all-datasets').onclick=()=>choose('');
  document.querySelectorAll('[data-kind]').forEach(b=>b.onclick=()=>{state.kind=b.dataset.kind;state.offset=0;load().catch(showError);});
  $('#layout').onchange=e=>{state.layout=e.target.value;state.offset=0;load().catch(showError);};
  let debounce;$('#search').value=state.q;$('#search').oninput=e=>{state.q=e.target.value;state.offset=0;clearTimeout(debounce);debounce=setTimeout(()=>load().catch(showError),250);};
  $('#previous').onclick=()=>{state.offset=Math.max(0,state.offset-state.limit);load().catch(showError);};$('#next').onclick=()=>{state.offset+=state.limit;load().catch(showError);};
  $('#refresh').onclick=async()=>{try{await api('/api/datasets/refresh',{method:'POST'});await load();}catch(e){showError(e);}};
  setInterval(()=>{if(!document.hidden)load().catch(showError);},30000);
  load().then(()=>{if(state.demo)return open(state.demo);}).catch(showError);
})();

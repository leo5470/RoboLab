/* Task-by-task library. All data labels are written with textContent. */
(() => {
  const $ = s => document.querySelector(s);
  const filters = ['layout', 'session', 'outcome', 'status', 'review'];
  const params = new URLSearchParams(location.search);
  const state = {family: params.get('family') || '', kind: params.get('kind') || '',
    q: params.get('q') || '', offset: Number(params.get('offset')) || 0, limit: 24,
    tasks: [], episodes: [], total: 0, demo: params.get('demo'), request: 0};
  filters.forEach(k => state[k] = params.get(k) || '');
  const human = s => String(s || '').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
  function node(tag, text, cls) {
    const n = document.createElement(tag);
    if (text != null) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  }
  async function api(path, options) {
    const r = await fetch(path, options);
    if (!r.ok) throw new Error((await r.text()).slice(0, 250));
    return r.json();
  }
  function url() {
    const p = new URLSearchParams();
    ['family','kind','q','offset',...filters].forEach(k => {if (state[k]) p.set(k, state[k]);});
    if (state.demo) p.set('demo', state.demo);
    history.replaceState(null, '', `${location.pathname}?${p}`);
  }
  function badge(label, cls = '') {return node('span', label, `badge ${cls}`);}
  function chooseFamily(family) {
    state.family = family; state.offset = 0; state.layout = ''; state.session = '';
    load().catch(showError);
  }
  function showError(e) {$('#errors').textContent = e.message;}
  function options(select, values, label, selected) {
    select.replaceChildren(new Option(label, ''));
    values.forEach(v => select.add(new Option(v, v)));
    select.value = selected;
  }
  function sidebar() {
    $('#tasks').replaceChildren();
    $('#all-tasks').classList.toggle('active', !state.family);
    state.tasks.forEach(t => {
      const b = node('button', null, `task-button ${state.family === t.family ? 'active' : ''}`);
      b.append(node('span', human(t.family)), node('small', t.collected + t.generated));
      b.onclick = () => chooseFamily(t.family); $('#tasks').append(b);
    });
    const tasks = state.tasks.filter(t => !state.family || t.family === state.family);
    const sum = key => tasks.reduce((n,t) => n + t[key], 0);
    $('#stats').replaceChildren();
    [['collected','Collected episodes'],['generated','Selected generated'],['ready','Videos ready'],['held','Contact review required']].forEach(([key,label]) => {
      const stat = node('div', null, 'stat'); stat.append(node('strong', sum(key)), node('span', label)); $('#stats').append(stat);
    });
    $('#title').textContent = state.family ? human(state.family) : 'All demonstrations';
    $('.subtitle').textContent = sum('generated') ?
      `Collected motion and generated variations · ${sum('native_ready')} / ${sum('generated')} selected videos at configured resolution.` :
      'Explore collected motion, one task at a time.';
    $('#hold').hidden = !sum('held');
    $('#hold').textContent = `${sum('held')} selected demonstrations require contact review. Videos are available for inspection; recorded success and selection do not certify contact fidelity.`;
    options($('#layout'), [...new Set(tasks.flatMap(t => t.layouts))].sort(), 'All layouts', state.layout);
    options($('#session'), [...new Set(tasks.flatMap(t => t.sessions))].sort(), 'All sessions', state.session);
    document.querySelectorAll('[data-kind]').forEach(b => b.classList.toggle('selected', b.dataset.kind === state.kind));
    ['outcome','status','review'].forEach(k => $('#' + k).value = state[k]);
  }
  function card(ep) {
    const c = node('button', null, 'card'); c.type = 'button';
    c.setAttribute('aria-label', `View ${human(ep.family)}, layout ${ep.layout}, ${ep.kind}`);
    const thumb = node('div', null, 'thumb');
    if (ep.poster) {
      const img = node('img'); img.src = ep.poster; img.loading = 'lazy'; img.alt = `${human(ep.family)} camera preview`; thumb.append(img);
    } else thumb.append(node('div', human(ep.video_status), 'muted'));
    if (ep.video) thumb.append(node('div', '▶', 'play'));
    thumb.append(node('span', ep.duration_s ? `${ep.duration_s.toFixed(1)}s` : '—'));
    const body = node('div', null, 'card-body');
    body.append(badge(ep.kind === 'generated' ? 'GENERATED · SELECTED' : 'COLLECTED', ep.kind === 'generated' ? 'generated' : ''));
    body.append(node('div', `${human(ep.family)} · Layout ${ep.layout}`, 'card-title'));
    body.append(node('div', ep.session, 'card-meta'));
    body.append(node('div', `${ep.task} · ${ep.id.slice(0,8)}`, 'card-meta'));
    body.append(badge(`Recorded ${ep.outcome}`, 'neutral'), badge(human(ep.video_status), 'neutral'));
    if (ep.video_upgrade_pending) body.append(badge(ep.render_status === 'running' ? 'Rendering native resolution' :
      ep.render_status === 'failed' ? 'Resolution upgrade failed' : 'Resolution upgrade queued', 'neutral'));
    if (ep.review === 'contact_review_required') body.append(badge('Contact review required', 'warn'));
    c.append(thumb, body); c.onclick = () => open(ep.id).catch(showError); return c;
  }
  async function load() {
    const request = ++state.request;
    url(); $('#errors').textContent = '';
    const p = new URLSearchParams({limit:state.limit, offset:state.offset});
    ['family','kind','q',...filters].forEach(k => {if (state[k]) p.set(k,state[k]);});
    const [tasks, data] = await Promise.all([api('/api/demonstrations/tasks'), api('/api/demonstrations?' + p)]);
    if (request !== state.request) return;
    state.tasks = tasks.tasks; state.episodes = data.episodes; state.total = data.total;
    sidebar(); $('#grid').replaceChildren(...data.episodes.map(card));
    if (!data.episodes.length) $('#grid').append(node('div', 'No demonstrations match these filters. New recordings and selected videos appear automatically.', 'empty'));
    $('#result-count').textContent = `${data.total} demonstrations · ${state.kind === 'generated' ? 'selected episodes only' : 'collected and selected generated'}`;
    $('#updated').textContent = `Updated ${new Date(data.updated * 1000).toLocaleTimeString()}`;
    $('#page-count').textContent = `Page ${Math.floor(state.offset / state.limit)+1} of ${Math.max(1,Math.ceil(data.total/state.limit))}`;
    $('#previous').disabled = state.offset === 0; $('#next').disabled = state.offset + state.limit >= data.total;
    if (tasks.warnings.length) $('#errors').textContent = `${tasks.warnings.length} catalog notices. Some active or incomplete recordings will be retried.`;
  }
  async function open(id) {
    const ep = await api('/api/demonstrations/' + id);
    state.demo = id; url();
    const dialog = $('#viewer'); const body = $('#viewer-body'); body.replaceChildren();
    $('#viewer-origin').textContent = ep.kind === 'generated' ? 'MIMICGEN / SELECTED DEMONSTRATION' : 'COLLECTED DEMONSTRATION';
    $('#viewer-title').textContent = `${human(ep.family)} · Layout ${ep.layout}`;
    const players = node('div', null, 'players'); body.append(players);
    const videos = [];
    function player(row, label) {
      const wrap = node('div'); wrap.append(node('div', label, 'player-label'));
      if (row.video) {
        const video = node('video'); video.controls = true; video.preload = 'metadata'; video.src = row.video;
        wrap.append(video); videos.push(video);
        video.addEventListener('error', () => wrap.append(node('p', 'Video could not be loaded. Refresh to check its availability.', 'notice')));
      } else wrap.append(node('p', `Video ${human(row.video_status)}${row.render_error ? ': ' + row.render_error : ''}`, 'notice'));
      players.append(wrap);
    }
    player(ep, 'Shoulder / wrist · one frame per recorded step at 15 FPS');
    const tools = node('div', null, 'player-tools'); body.append(tools);
    const speedLabel = node('label', 'Playback speed '); const speed = node('select');
    [.25,.5,1,1.5,2].forEach(v => speed.add(new Option(v + '×',v))); speed.value='1';
    speed.onchange = () => videos.forEach(v => v.playbackRate = Number(speed.value)); speedLabel.append(speed); tools.append(speedLabel);
    const loopLabel = node('label', ' Loop '); const loop = node('input'); loop.type='checkbox';
    loop.onchange = () => videos.forEach(v => v.loop = loop.checked); loopLabel.prepend(loop); tools.append(loopLabel);
    const copy = node('button', 'Copy link'); copy.onclick = async () => {
      try {await navigator.clipboard.writeText(location.href); copy.textContent='Link copied';}
      catch {window.prompt('Copy this demonstration link', location.href);}
    }; tools.append(copy);
    if (ep.source_id) {
      const source = node('button', 'View human source'); source.onclick = () => open(ep.source_id).catch(showError); tools.append(source);
      const compare = node('button', 'Compare with source'); compare.onclick = async () => {
        try {
          const src = await api('/api/demonstrations/' + ep.source_id); player(src, 'Human source · independent trajectory timing');
          players.classList.add('compare'); compare.disabled = true;
          const linked = node('button','Play / pause both'); linked.onclick = () => {
            const play = videos[0].paused;
            videos.forEach(v => play ? v.play().catch(showError) : v.pause());
          }; tools.append(linked);
          let syncing = false;
          videos[0]?.addEventListener('seeking', () => {if (!syncing && videos[1]) {syncing=true; videos[1].currentTime=Math.min(videos[0].currentTime,videos[1].duration || 0); syncing=false;}});
        } catch(e) {showError(e);}
      }; tools.append(compare);
    }
    const index = state.episodes.findIndex(r => r.id === ep.id);
    for (const [label,delta] of [['← Previous',-1],['Next →',1]]) {
      const target = state.offset + index + delta;
      const b = node('button',label); b.disabled = index < 0 || target < 0 || target >= state.total;
      b.onclick = async () => {
        try {
          state.offset = Math.floor(target/state.limit)*state.limit;
          await load(); await open(state.episodes[target-state.offset].id);
        } catch(e) {showError(e);}
      }; tools.append(b);
    }
    const review = node('p', ep.review === 'contact_review_required' ? 'Contact review required. Inspect for gripper/object intersection and controlled release.' : 'Contact fidelity has not been certified.', ep.review === 'contact_review_required' ? 'notice' : 'muted'); body.append(review);
    const meta = node('div', null, 'metadata');
    [['Task',ep.task],['Session',ep.session],['Episode',ep.id],['Recorded outcome',ep.outcome],['Origin',ep.kind],['Duration',`${ep.duration_s.toFixed(2)}s`],['Seed',ep.seed ?? '—'],['Variation',ep.tier ?? '—'],['Video',ep.video_width ? `${ep.video_width} × ${ep.video_height}` : ep.video_status]].forEach(([k,v]) => {
      const entry = node('div'); entry.append(node('small', k), node('span',v,'meta-value')); meta.append(entry);
    }); body.append(meta); if (!dialog.open) dialog.showModal();
  }
  $('#viewer').addEventListener('close', () => {$('#viewer-body').querySelectorAll('video').forEach(v=>v.pause()); state.demo=null; url();});
  $('#close-viewer').onclick = () => $('#viewer').close();
  $('#viewer').addEventListener('click', e => {if(e.target === $('#viewer')) $('#viewer').close();});
  $('#all-tasks').onclick = () => chooseFamily('');
  document.querySelectorAll('[data-kind]').forEach(b => b.onclick = () => {state.kind=b.dataset.kind; state.offset=0; load().catch(showError);});
  filters.forEach(k => $('#' + k).onchange = e => {state[k]=e.target.value; state.offset=0; load().catch(showError);});
  let debounce; $('#search').value = state.q; $('#search').oninput = e => {clearTimeout(debounce); state.q=e.target.value; state.offset=0; debounce=setTimeout(()=>load().catch(showError),250);};
  $('#previous').onclick = () => {state.offset=Math.max(0,state.offset-state.limit); load().catch(showError);};
  $('#next').onclick = () => {state.offset+=state.limit; load().catch(showError);};
  $('#refresh').onclick = async () => {try {await api('/api/demonstrations/refresh',{method:'POST'}); await load();} catch(e){showError(e);}};
  setInterval(() => {if(!document.hidden) load().catch(showError);},30000);
  load().then(() => {if(state.demo) return open(state.demo);}).catch(showError);
})();

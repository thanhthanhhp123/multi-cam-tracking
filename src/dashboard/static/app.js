/* Dashboard MTMCT — vanilla JS, không framework, không CDN.
 *
 * Ba việc:
 *   1. nạp sơ đồ camera (/api/layout) và vẽ vùng phủ mặt đất một lần;
 *   2. giữ kết nối WebSocket /ws/live, cập nhật chấm vị trí theo thời gian thực;
 *   3. tra cứu hành trình theo Global ID từ SQLite (/api/tracks/{id}).
 *
 * Toạ độ: server trả về MÉT trên mặt phẳng tham chiếu. Việc quy đổi mét -> pixel làm ở
 * đây (hàm `toScreen`) chứ không nhét vào viewBox của SVG — làm vậy thì nét vẽ và bán
 * kính chấm cũng bị co giãn theo, rất khó chỉnh.
 */

const SVG_NS = "http://www.w3.org/2000/svg";
const el = (id) => document.getElementById(id);

const state = {
  layout: null,
  bounds: null,
  tracks: new Map(),
  selected: null,
  view: { w: 0, h: 0, scale: 1, ox: 0, oy: 0 },
};

/* Màu ổn định theo Global ID: cùng một người luôn cùng màu qua các lần nạp trang,
 * nên nhìn bản đồ là nhận ra ngay. Băm đơn giản -> góc màu. */
function colorOf(gid) {
  const hue = (gid * 137.508) % 360;
  return `hsl(${hue.toFixed(1)} 70% 60%)`;
}

function fmtTime(ms) {
  if (!ms) return "—";
  return new Date(ms).toLocaleTimeString("vi-VN", { hour12: false });
}

function fmtDuration(ms) {
  const s = ms / 1000;
  return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, "0")}s`;
}

/* ------------------------------------------------------------------ bản đồ */

function computeView() {
  const svg = el("map");
  const rect = svg.getBoundingClientRect();
  state.view.w = rect.width;
  state.view.h = rect.height;
  const b = state.bounds;
  if (!b || !rect.width || !rect.height) return;

  const spanX = Math.max(b.max_x - b.min_x, 1e-6);
  const spanY = Math.max(b.max_y - b.min_y, 1e-6);
  // Giữ đúng tỉ lệ: 1 mét theo X phải bằng 1 mét theo Y, nếu không đường đi của người
  // sẽ méo và mọi ước lượng bằng mắt đều sai.
  const scale = Math.min(rect.width / spanX, rect.height / spanY) * 0.92;
  state.view.scale = scale;
  state.view.ox = (rect.width - spanX * scale) / 2 - b.min_x * scale;
  state.view.oy = (rect.height + spanY * scale) / 2 + b.min_y * scale;
}

// Y đảo dấu: trục Y của mặt phẳng hướng lên, trục Y của màn hình hướng xuống.
const toScreen = (x, y) => [x * state.view.scale + state.view.ox, state.view.oy - y * state.view.scale];

function svgEl(name, attrs, parent) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function drawStatic() {
  const svg = el("map");
  svg.innerHTML = "";
  computeView();
  if (!state.bounds) return;

  const grid = svgEl("g", { id: "grid" }, svg);
  const b = state.bounds;
  const step = niceStep(Math.max(b.max_x - b.min_x, b.max_y - b.min_y));
  for (let x = Math.ceil(b.min_x / step) * step; x <= b.max_x; x += step) {
    const [px] = toScreen(x, 0);
    svgEl("line", { x1: px, y1: 0, x2: px, y2: state.view.h, stroke: "#1b2029" }, grid);
  }
  for (let y = Math.ceil(b.min_y / step) * step; y <= b.max_y; y += step) {
    const [, py] = toScreen(0, y);
    svgEl("line", { x1: 0, y1: py, x2: state.view.w, y2: py, stroke: "#1b2029" }, grid);
  }
  // Thước tỉ lệ: không có nó thì bản đồ chỉ là hình vẽ, có nó mới đọc được khoảng cách.
  const barPx = step * state.view.scale;
  const y0 = state.view.h - 18;
  svgEl("line", { x1: 16, y1: y0, x2: 16 + barPx, y2: y0, stroke: "#5b6577", "stroke-width": 2 }, grid);
  const label = svgEl("text", { x: 16 + barPx + 8, y: y0 + 4, fill: "#98a0b0", "font-size": 11 }, grid);
  label.textContent = `${step} m`;

  svgEl("g", { id: "fov" }, svg);
  svgEl("g", { id: "trails" }, svg);
  svgEl("g", { id: "marks" }, svg);
  drawFov();
}

function niceStep(span) {
  const raw = span / 6;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  return [1, 2, 5, 10].map((m) => m * pow).find((v) => v >= raw) || 10 * pow;
}

function drawFov() {
  const layer = el("fov");
  if (!layer) return;
  layer.innerHTML = "";
  if (!el("show-fov").checked || !state.layout) return;

  state.layout.cameras.forEach((cam, i) => {
    if (!cam.footprint.length) return;
    const hue = (i * 47) % 360;
    const points = cam.footprint.map(([x, y]) => toScreen(x, y).join(",")).join(" ");
    svgEl("polygon", {
      points,
      fill: `hsl(${hue} 60% 55% / 0.10)`,
      stroke: `hsl(${hue} 60% 60% / 0.55)`,
      "stroke-width": 1,
    }, layer);
    const cx = cam.footprint.reduce((s, p) => s + p[0], 0) / cam.footprint.length;
    const cy = cam.footprint.reduce((s, p) => s + p[1], 0) / cam.footprint.length;
    const [px, py] = toScreen(cx, cy);
    const text = svgEl("text", {
      x: px, y: py, fill: `hsl(${hue} 60% 72%)`, "font-size": 11,
      "text-anchor": "middle", "pointer-events": "none",
    }, layer);
    text.textContent = cam.cam_id;
  });
}

function drawTracks() {
  const marks = el("marks");
  const trails = el("trails");
  if (!marks) return;
  marks.innerHTML = "";
  trails.innerHTML = "";
  const showTrails = el("show-trails").checked;

  for (const track of state.tracks.values()) {
    if (track.x_m === null || track.x_m === undefined) continue;
    const color = colorOf(track.global_id);
    const selected = state.selected === track.global_id;

    if (showTrails && track.trail && track.trail.length > 1) {
      svgEl("polyline", {
        points: track.trail.map(([x, y]) => toScreen(x, y).join(",")).join(" "),
        fill: "none", stroke: color, "stroke-width": selected ? 2.5 : 1.5,
        "stroke-opacity": 0.55, "stroke-linejoin": "round",
      }, trails);
    }

    const [px, py] = toScreen(track.x_m, track.y_m);
    svgEl("circle", {
      cx: px, cy: py, r: selected ? 8 : 5,
      fill: color, stroke: "#0b0d11", "stroke-width": 1.5,
    }, marks);
    // Nhãn chỉ cho người xuyên camera hoặc người đang được chọn: gắn nhãn tất cả thì
    // đông người là chữ chồng lên nhau, không đọc được gì.
    if (selected || track.n_cameras > 1) {
      const text = svgEl("text", {
        x: px + 10, y: py + 4, fill: color, "font-size": 11, "pointer-events": "none",
      }, marks);
      text.textContent = `#${track.global_id}`;
    }
  }
}

/* ------------------------------------------------------------- danh sách live */

function renderLive() {
  const list = el("live-list");
  const tracks = [...state.tracks.values()].sort((a, b) => b.ts_ms - a.ts_ms);
  el("n-live").textContent = tracks.length;
  el("n-cross").textContent = tracks.filter((t) => t.n_cameras > 1).length;

  if (!tracks.length) {
    list.innerHTML = '<li class="empty">Chưa có ai trong khung hình.</li>';
    return;
  }
  list.innerHTML = "";
  for (const track of tracks) {
    const li = document.createElement("li");
    li.className = state.selected === track.global_id ? "active" : "";
    li.innerHTML = `
      <span class="dot" style="background:${colorOf(track.global_id)}"></span>
      <span class="gid">#${track.global_id}</span>
      <span class="cams">${track.cameras.map((c) => `<span class="cam-tag">${c}</span>`).join("")}</span>
      <span class="meta">${fmtTime(track.ts_ms)}</span>`;
    li.onclick = () => selectTrack(track.global_id);
    list.appendChild(li);
  }
}

function selectTrack(gid) {
  state.selected = state.selected === gid ? null : gid;
  renderLive();
  drawTracks();
  if (state.selected) loadTrajectory(state.selected);
}

/* -------------------------------------------------------------- tra cứu SQLite */

async function loadTrajectory(gid) {
  const box = el("trajectory");
  box.innerHTML = "<p class='empty'>Đang tải…</p>";
  el("gid").value = gid;
  try {
    const res = await fetch(`/api/tracks/${gid}`);
    if (!res.ok) {
      const detail = (await res.json().catch(() => ({}))).detail || res.statusText;
      box.innerHTML = `<p class="error">${detail}</p>`;
      return;
    }
    renderTrajectory(await res.json());
  } catch (err) {
    box.innerHTML = `<p class="error">Không tải được: ${err}</p>`;
  }
}

function renderTrajectory(trip) {
  const total = Math.max(trip.end_ms - trip.start_ms, 1);
  const rows = trip.appearances.map((a) => {
    const width = Math.max(2, (a.duration_ms / total) * 100);
    const offset = ((a.start_ms - trip.start_ms) / total) * 100;
    return `<tr>
      <td>${a.cam_id}</td>
      <td class="num">${fmtTime(a.start_ms)}</td>
      <td class="num">${fmtDuration(a.duration_ms)}</td>
      <td class="num">${a.n_frames}</td>
      <td style="width:35%"><div class="bar" style="margin-left:${offset.toFixed(1)}%;width:${width.toFixed(1)}%;background:${colorOf(trip.global_id)}"></div></td>
    </tr>`;
  });
  el("trajectory").innerHTML = `
    <div class="trip-head">
      <b style="color:${colorOf(trip.global_id)}">#${trip.global_id}</b>
      <span>${trip.n_cameras} camera · ${trip.appearances.length} lượt ·
        ${fmtDuration(trip.end_ms - trip.start_ms)}</span>
    </div>
    <table class="trip">
      <thead><tr><th>camera</th><th>bắt đầu</th><th>kéo dài</th><th>khung</th><th>dòng thời gian</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>`;
}

async function loadHistory() {
  const list = el("history-list");
  try {
    const res = await fetch("/api/tracks?min_cameras=2&limit=50");
    if (!res.ok) {
      list.innerHTML = `<li class="empty">${(await res.json().catch(() => ({}))).detail || "Chưa có SQLite"}</li>`;
      return;
    }
    const data = await res.json();
    if (!data.tracks.length) {
      list.innerHTML = '<li class="empty">Chưa có Global ID nào đi qua nhiều camera.</li>';
      return;
    }
    list.innerHTML = "";
    for (const row of data.tracks) {
      const li = document.createElement("li");
      li.innerHTML = `
        <span class="dot" style="background:${colorOf(row.global_id)}"></span>
        <span class="gid">#${row.global_id}</span>
        <span class="meta">${row.n_cameras} camera · ${row.n_tracklets} lượt ·
          ${fmtTime(row.last_seen_ms)}</span>`;
      li.onclick = () => loadTrajectory(row.global_id);
      list.appendChild(li);
    }
  } catch (err) {
    list.innerHTML = `<li class="empty">Không tải được: ${err}</li>`;
  }
}

/* ------------------------------------------------------------------ WebSocket */

function applyTracks(items) {
  for (const track of items) state.tracks.set(track.global_id, track);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/live`);

  ws.onopen = () => {
    el("conn").textContent = "đang nối";
    el("conn").className = "badge on";
    // Ping định kỳ: server dùng `receive_text()` để biết client còn sống.
    ws.pingTimer = setInterval(() => ws.readyState === 1 && ws.send("ping"), 20000);
  };
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "snapshot") {
      state.tracks = new Map(msg.tracks.map((t) => [t.global_id, t]));
      el("n-updates").textContent = msg.n_updates;
      el("clock").textContent = fmtTime(msg.latest_ts_ms);
    } else if (msg.type === "update") {
      applyTracks(msg.tracks);
      el("n-updates").textContent = Number(el("n-updates").textContent) + msg.tracks.length;
      el("clock").textContent = fmtTime(Math.max(...msg.tracks.map((t) => t.ts_ms)));
    }
    renderLive();
    drawTracks();
  };
  ws.onclose = () => {
    clearInterval(ws.pingTimer);
    el("conn").textContent = "mất kết nối";
    el("conn").className = "badge off";
    // Tự nối lại: engine hoặc Redis restart là chuyện thường lúc phát triển, không nên
    // bắt người dùng F5.
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
}

/* ------------------------------------------------------------------ khởi động */

async function init() {
  try {
    state.layout = await (await fetch("/api/layout")).json();
    state.bounds = state.layout.bounds;
  } catch (err) {
    console.error("Không nạp được sơ đồ camera", err);
  }
  el("map-wrap").classList.toggle("no-map", !state.bounds);
  drawStatic();

  try {
    const snapshot = await (await fetch("/api/live")).json();
    state.tracks = new Map(snapshot.tracks.map((t) => [t.global_id, t]));
    el("n-updates").textContent = snapshot.n_updates;
  } catch (err) {
    console.warn("Không lấy được trạng thái hiện tại", err);
  }
  renderLive();
  drawTracks();
  loadHistory();
  // Danh sách lịch sử đọc từ SQLite nên không có kênh đẩy: làm mới định kỳ là đủ,
  // người mới đi xuyên camera xong sẽ hiện ra sau vài giây.
  setInterval(loadHistory, 15000);
  connect();

  addEventListener("resize", () => { drawStatic(); drawTracks(); });
  el("show-fov").onchange = drawFov;
  el("show-trails").onchange = drawTracks;
  el("lookup").onsubmit = (event) => {
    event.preventDefault();
    const gid = Number(el("gid").value);
    if (gid) loadTrajectory(gid);
  };
}

init();

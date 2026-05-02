// opus vps-live UI: WebSocket snapshot renderer plus small REST controls.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = { mode: "collect", lastSnapshot: null, settings: null, models: null };

function fmt(n, digits = 2) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "-";
    return Number(n).toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    });
}

function fmtPrice(n) {
    if (!n) return "-";
    const abs = Math.abs(Number(n));
    const digits = abs >= 100 ? 2 : abs >= 1 ? 4 : 6;
    return Number(n).toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    });
}

function pnlClass(n) {
    const v = Number(n || 0);
    return v > 0 ? "pos" : v < 0 ? "neg" : "";
}

function setConnBadge(el, ok, label, ts, now) {
    el.classList.remove("ok", "err");
    el.classList.add(ok ? "ok" : "err");
    const ageMs = ts ? Math.max(0, (now - ts) * 1000) : null;
    const ageStr = ageMs !== null ? ` (${ageMs.toFixed(0)}ms)` : "";
    el.innerHTML = `${label}: <b>${ok ? "ok" : "down"}</b><span class="muted-text">${ageStr}</span>`;
}

function renderModeButtons(currentMode) {
    $$(".mode-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.mode === currentMode);
    });
}

function renderModels(data) {
    if (!data) return;
    state.models = data;
    const select = $("#model-select");
    const models = data.models || [];
    const current = data.active_model || (data.status && data.status.model_dir) || "";
    select.innerHTML = "";
    if (models.length === 0) {
        const opt = document.createElement("option");
        opt.value = current;
        opt.textContent = current || "no models";
        select.appendChild(opt);
        select.disabled = true;
        $("#model-load").disabled = true;
        $("#model-horizons").textContent = "-";
        return;
    }
    select.disabled = false;
    $("#model-load").disabled = false;
    for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m.path;
        opt.textContent = m.active ? `${m.label} *` : m.label;
        opt.dataset.horizons = (m.horizons || []).join(",");
        select.appendChild(opt);
        if (m.active || m.path === current) opt.selected = true;
    }
    renderSelectedModelMeta();
}

function renderSelectedModelMeta() {
    const opt = $("#model-select").selectedOptions[0];
    $("#model-horizons").textContent = opt ? (opt.dataset.horizons || "-") : "-";
}

function writeInput(id, value) {
    const el = $(id);
    if (el && document.activeElement !== el && value !== undefined && value !== null) {
        el.value = value;
    }
}

function renderSettings(settings) {
    if (!settings) return;
    state.settings = settings;
    const caps = settings.hard_caps || {};
    $("#settings-caps").textContent = caps.hard_max_notional_usd === undefined
        ? ""
        : `env caps: leverage<=${caps.hard_max_leverage}, live symbols<=${caps.hard_max_live_symbols}, ` +
          `daily<=${caps.hard_daily_loss_usd}, 12h<=${caps.hard_12h_loss_usd}, ` +
          `symbol<=${caps.hard_symbol_loss_usd}, notional<=${caps.hard_notional_usd}`;
    writeInput("#s-leverage", settings.leverage);
    writeInput("#s-max-live", settings.max_live_symbols);
    writeInput("#s-daily-loss", settings.daily_loss_limit_usd);
    writeInput("#s-12h-loss", settings.loss_12h_limit_usd);
    writeInput("#s-symbol-loss", settings.symbol_loss_limit_usd);
    writeInput("#s-probation-notional", settings.probation_notional_usd);
    writeInput("#s-active-notional", settings.active_notional_usd);
    writeInput("#s-min-gross", settings.min_expected_gross_bp);
    writeInput("#s-global-giveback", Math.round((settings.global_profit_giveback_pct || 0) * 100));
    writeInput("#s-symbol-giveback", Math.round((settings.symbol_profit_giveback_pct || 0) * 100));
    writeInput("#s-loss-streak", settings.loss_streak_limit);
    writeInput("#s-rolling-trades", settings.rolling_guard_trades);
    writeInput("#s-rolling-wr", Math.round((settings.rolling_min_win_rate || 0) * 100));
    writeInput("#s-rolling-loss-bp", settings.rolling_min_loss_net_bp);
    writeInput("#s-rolling-dd", Math.round((settings.rolling_min_drawdown_pct || 0) * 100));
    writeInput("#s-probation-trades", settings.probation_trades);
    writeInput("#s-cooldown-hours", settings.cooldown_hours);
}

function renderGuards(guards) {
    if (!guards) return;
    writeInput("#g-dll", guards.daily_loss_limit_usd);
    writeInput("#g-12h", guards.loss_12h_limit_usd);
    writeInput("#g-symbol", guards.symbol_loss_limit_usd);
    writeInput("#g-mpu", guards.max_position_usd);
    writeInput("#g-mls", guards.max_live_symbols);
    writeInput("#g-mop", guards.max_orders_per_min);

    $("#daily-pnl").textContent = `$${fmt(guards.daily_pnl)}`;
    $("#daily-pnl").className = pnlClass(guards.daily_pnl);
    $("#pnl-12h").textContent = `$${fmt(guards.pnl_12h)}`;
    $("#pnl-12h").className = pnlClass(guards.pnl_12h);
    $("#daily-orders").textContent = guards.daily_orders || 0;

    const emergency = $("#emergency");
    if (guards.emergency_stopped) {
        emergency.classList.remove("hidden");
        $("#emergency-reason").textContent = guards.emergency_reason || "";
    } else {
        emergency.classList.add("hidden");
    }
}

function modeToggle(symbol, mode) {
    const checked = mode === "live" ? "checked" : "";
    return `
        <label class="switch" title="paper/live for ${symbol}">
            <input class="mode-toggle" data-symbol="${symbol}" type="checkbox" ${checked} />
            <span></span>
        </label>
    `;
}

function renderWatchlist(symbols) {
    const tbody = $("#watchlist tbody");
    tbody.innerHTML = "";
    if (!symbols || symbols.length === 0) {
        tbody.innerHTML = `<tr><td colspan="21" class="empty">empty - add a symbol above</td></tr>`;
        return;
    }
    for (const s of symbols) {
        const rowState = s.live_state || "paper";
        const isBlocked = rowState === "blocked" || rowState === "cooldown" || Boolean(s.block_reason);
        const tr = document.createElement("tr");
        tr.className = isBlocked ? "blocked-row" : "";
        tr.innerHTML = `
            <td class="sym">${s.symbol}</td>
            <td>${modeToggle(s.symbol, s.execution_mode || "paper")}</td>
            <td><span class="pill ${rowState}">${rowState}</span></td>
            <td>$${fmt(s.position_size_usd, 0)}</td>
            <td>$${fmt(s.current_notional_usd || 0, 0)}</td>
            <td>${fmt(s.expected_gross_bp || 0, 2)}</td>
            <td>${fmt((s.pnl_drawdown_pct || 0) * 100, 1)}</td>
            <td>${fmt((s.rolling_win_rate || 0) * 100, 1)}%</td>
            <td class="${pnlClass(s.rolling_sum_net_bp)}">${fmt(s.rolling_sum_net_bp || 0, 2)}</td>
            <td>${s.live_trade_count || 0}</td>
            <td>${s.live_wins || 0}/${s.live_losses || 0}</td>
            <td>${s.consecutive_losses || 0}</td>
            <td>${fmtPrice(s.best_bid)}</td>
            <td>${fmtPrice(s.best_ask)}</td>
            <td>${fmt(s.spread_bp, 2)}</td>
            <td>${s.snapshots_written || 0}</td>
            <td>${fmt(s.position_base, 4)}</td>
            <td class="${pnlClass(s.realized_pnl)}">$${fmt(s.realized_pnl || 0)}</td>
            <td class="${pnlClass(s.symbol_realized_pnl_12h)}">$${fmt(s.symbol_realized_pnl_12h || 0)}</td>
            <td class="block">${s.block_reason || ""}</td>
            <td class="actions">
                <button class="disable" data-symbol="${s.symbol}" title="watch only">watch</button>
                <button class="rm" data-symbol="${s.symbol}" title="remove">x</button>
            </td>
        `;
        tbody.appendChild(tr);
    }
    tbody.querySelectorAll("input.mode-toggle").forEach((b) => {
        b.addEventListener("change", () => setSymbolMode(b.dataset.symbol, b.checked ? "live" : "paper"));
    });
    tbody.querySelectorAll("button.disable").forEach((b) => {
        b.addEventListener("click", () => disableSymbol(b.dataset.symbol));
    });
    tbody.querySelectorAll("button.rm").forEach((b) => {
        b.addEventListener("click", () => removeSymbol(b.dataset.symbol));
    });
}

function renderSnapshot(snap) {
    state.lastSnapshot = snap;
    const now = snap.now || Date.now() / 1000;
    setConnBadge($("#conn-binance"), snap.binance_connected, "binance", snap.binance_last_msg_ts, now);
    setConnBadge(
        $("#conn-binance-private"),
        snap.binance_private_connected,
        "private",
        snap.binance_private_last_msg_ts,
        now
    );
    setConnBadge($("#conn-bybit"), snap.bybit_connected, "bybit", snap.bybit_last_msg_ts, now);

    if (snap.binance_last_msg_ts) {
        const ms = ((now - snap.binance_last_msg_ts) * 1000).toFixed(0);
        $("#ws-age").innerHTML = `ws age: <b>${ms}ms</b>`;
    }
    renderModeButtons(snap.mode);
    renderGuards(snap.guards);
    renderWatchlist(snap.symbols);
}

function renderTrader(trader, predictor) {
    if (!trader) return;
    const open = trader.open_positions || [];
    $("#trader-open").textContent = open.length;
    $("#trader-count").textContent = trader.daily_trades || 0;
    $("#trader-winrate").textContent = `${((trader.daily_win_rate || 0) * 100).toFixed(1)}%`;
    $("#trader-pnl").textContent = `$${fmt(trader.daily_pnl_usd || 0)}`;
    $("#trader-pnl").className = pnlClass(trader.daily_pnl_usd);
    if (predictor) {
        const horizons = (predictor.horizons || []).join(",") || "-";
        $("#trader-predictor").textContent = predictor.enabled ? `loaded (${horizons})` : "disabled";
    }
    const allowed = (trader.allowed_symbols || []).join(",") || "all";
    $("#trader-meta").textContent =
        `horizon=${trader.horizon} thr=${trader.conf_threshold} notional=$${trader.notional_usd} symbols=${allowed}`;

    const tbody = $("#trades tbody");
    tbody.innerHTML = "";
    if (open.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="empty">no open positions</td></tr>`;
        return;
    }
    const nowMs = Date.now();
    for (const p of open) {
        const ageS = Math.max(0, (nowMs - p.ts_open_ms) / 1000);
        const sideCls = p.side === "long" ? "pos" : "neg";
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${p.symbol}</td>
            <td class="${sideCls}">${p.side}</td>
            <td>${fmt(p.qty, 6)}</td>
            <td>${fmtPrice(p.entry_price)}</td>
            <td>${fmt(p.confidence, 3)}</td>
            <td>${ageS.toFixed(1)}</td>
            <td>${p.horizon}</td>
        `;
        tbody.appendChild(tr);
    }
}

async function apiJson(path, opts = {}) {
    const r = await fetch(path, {
        headers: { "content-type": "application/json" },
        ...opts,
    });
    if (!r.ok) {
        let detail = await r.text();
        try {
            const data = JSON.parse(detail);
            detail = data.detail || detail;
        } catch (_) {
            // keep raw text
        }
        throw new Error(`${r.status}: ${detail}`);
    }
    return r.json();
}

async function refreshTraderPanel() {
    try {
        const [trader, predictor] = await Promise.all([
            apiJson("/api/trader"),
            apiJson("/api/predictor"),
        ]);
        renderTrader(trader, predictor);
    } catch (e) {
        console.warn("trader refresh failed", e);
    }
}

async function refreshSettings() {
    try {
        renderSettings(await apiJson("/api/settings"));
    } catch (e) {
        console.warn("settings refresh failed", e);
    }
}

async function refreshModels() {
    try {
        renderModels(await apiJson("/api/models"));
    } catch (e) {
        console.warn("models refresh failed", e);
    }
}

async function setMode(mode) {
    if (mode === "live") {
        const ok = confirm(
            "Switch process to LIVE mode?\n\nThe bot can send real Binance Futures orders. Check API keys, IP whitelist, loss limits, and symbol paper/live switches first."
        );
        if (!ok) return;
    }
    try {
        await apiJson("/api/mode", { method: "POST", body: JSON.stringify({ mode }) });
    } catch (e) {
        alert(`mode change failed: ${e.message}`);
    }
}

async function selectModel() {
    const select = $("#model-select");
    const modelDir = select.value;
    if (!modelDir) return;
    const ok = confirm("Load selected model? Open positions must be closed before switching.");
    if (!ok) return;
    try {
        await apiJson("/api/models/select", {
            method: "POST",
            body: JSON.stringify({ model_dir: modelDir }),
        });
        await Promise.all([refreshModels(), refreshTraderPanel()]);
    } catch (e) {
        alert(`model switch failed: ${e.message}`);
        await refreshModels();
    }
}

async function addSymbol(symbol, sizeUsd) {
    await apiJson("/api/watchlist/add", {
        method: "POST",
        body: JSON.stringify({ symbol: symbol.toUpperCase(), position_size_usd: sizeUsd }),
    });
}

async function removeSymbol(symbol) {
    if (!confirm(`Remove ${symbol} from watchlist?`)) return;
    try {
        await apiJson("/api/watchlist/remove", {
            method: "POST",
            body: JSON.stringify({ symbol }),
        });
    } catch (e) {
        alert(`remove failed: ${e.message}`);
    }
}

async function setSymbolMode(symbol, executionMode) {
    if (executionMode === "live") {
        const ok = confirm(`Enable LIVE execution for ${symbol}?`);
        if (!ok) {
            renderWatchlist((state.lastSnapshot && state.lastSnapshot.symbols) || []);
            return;
        }
    }
    try {
        await apiJson("/api/watchlist/mode", {
            method: "POST",
            body: JSON.stringify({ symbol, execution_mode: executionMode }),
        });
    } catch (e) {
        alert(`symbol mode failed: ${e.message}`);
        renderWatchlist((state.lastSnapshot && state.lastSnapshot.symbols) || []);
    }
}

async function disableSymbol(symbol) {
    const reason = prompt(`Watch-only reason for ${symbol}`, "operator");
    if (reason === null) return;
    try {
        await apiJson("/api/watchlist/disable", {
            method: "POST",
            body: JSON.stringify({ symbol, reason }),
        });
    } catch (e) {
        alert(`disable failed: ${e.message}`);
    }
}

async function applyGuards(e) {
    e.preventDefault();
    const body = {
        daily_loss_limit_usd: parseFloat($("#g-dll").value),
        loss_12h_limit_usd: parseFloat($("#g-12h").value),
        symbol_loss_limit_usd: parseFloat($("#g-symbol").value),
        max_position_usd: parseFloat($("#g-mpu").value),
        max_live_symbols: parseInt($("#g-mls").value, 10),
        max_orders_per_min: parseInt($("#g-mop").value, 10),
    };
    try {
        await apiJson("/api/guards", { method: "POST", body: JSON.stringify(body) });
    } catch (e2) {
        alert(`guards update failed: ${e2.message}`);
    }
}

async function applySettings(e) {
    e.preventDefault();
    const body = {
        leverage: parseInt($("#s-leverage").value, 10),
        max_live_symbols: parseInt($("#s-max-live").value, 10),
        daily_loss_limit_usd: parseFloat($("#s-daily-loss").value),
        loss_12h_limit_usd: parseFloat($("#s-12h-loss").value),
        symbol_loss_limit_usd: parseFloat($("#s-symbol-loss").value),
        probation_notional_usd: parseFloat($("#s-probation-notional").value),
        active_notional_usd: parseFloat($("#s-active-notional").value),
        min_expected_gross_bp: parseFloat($("#s-min-gross").value),
        global_profit_giveback_pct: parseFloat($("#s-global-giveback").value) / 100,
        symbol_profit_giveback_pct: parseFloat($("#s-symbol-giveback").value) / 100,
        loss_streak_limit: parseInt($("#s-loss-streak").value, 10),
        rolling_guard_trades: parseInt($("#s-rolling-trades").value, 10),
        rolling_min_win_rate: parseFloat($("#s-rolling-wr").value) / 100,
        rolling_min_loss_net_bp: parseFloat($("#s-rolling-loss-bp").value),
        rolling_min_drawdown_pct: parseFloat($("#s-rolling-dd").value) / 100,
        probation_trades: parseInt($("#s-probation-trades").value, 10),
        cooldown_hours: parseInt($("#s-cooldown-hours").value, 10),
    };
    try {
        renderSettings(await apiJson("/api/settings", { method: "POST", body: JSON.stringify(body) }));
    } catch (e2) {
        alert(`settings update failed: ${e2.message}`);
    }
}

async function emergencyStop() {
    const reason = prompt("Emergency stop reason", "operator");
    if (reason === null) return;
    if (!confirm("Stop trading, cancel orders, and flatten managed live symbols?")) return;
    try {
        await apiJson("/api/emergency/stop", {
            method: "POST",
            body: JSON.stringify({ reason }),
        });
    } catch (e) {
        alert(`emergency stop failed: ${e.message}`);
    }
}

function wireEvents() {
    $$(".mode-btn").forEach((btn) => btn.addEventListener("click", () => setMode(btn.dataset.mode)));
    $("#add-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const sym = $("#add-symbol").value.trim();
        const sz = parseFloat($("#add-size").value);
        if (!sym || !sz) return;
        try {
            await addSymbol(sym, sz);
            $("#add-symbol").value = "";
            $("#add-size").value = "";
        } catch (err) {
            alert(`add failed: ${err.message}`);
        }
    });
    $("#guards-form").addEventListener("submit", applyGuards);
    $("#settings-form").addEventListener("submit", applySettings);
    $("#model-select").addEventListener("change", renderSelectedModelMeta);
    $("#model-load").addEventListener("click", selectModel);
    $("#emergency-stop").addEventListener("click", emergencyStop);
    $("#emergency-clear").addEventListener("click", async () => {
        try {
            await apiJson("/api/emergency/clear", { method: "POST" });
        } catch (e) {
            alert(`clear failed: ${e.message}`);
        }
    });
}

function connectWs() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (evt) => {
        try {
            renderSnapshot(JSON.parse(evt.data));
        } catch (e) {
            console.error("ws parse", e);
        }
    };
    ws.onclose = () => setTimeout(connectWs, 1000);
    ws.onerror = (e) => console.warn("ws err", e);
}

wireEvents();
connectWs();
refreshSettings();
refreshModels();
refreshTraderPanel();
setInterval(refreshSettings, 10000);
setInterval(refreshModels, 10000);
setInterval(refreshTraderPanel, 1500);

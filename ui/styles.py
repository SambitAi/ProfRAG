NAVBAR_HTML = (
    '<div class="profrag-navbar">'
    '<span class="profrag-navbar-title">📄 ProfRAG</span>'
    "</div>"
)

APP_CSS = """
<style>
/* ── Navbar ──────────────────────────────────────────────────────────────── */
.profrag-navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 52px;
    z-index: 999999;
    background: linear-gradient(90deg, #1a3558 0%, #2563a8 100%);
    display: flex;
    align-items: center;
    padding: 0 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
.profrag-navbar-title {
    color: #ffffff;
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

/* ── Main layout ─────────────────────────────────────────────────────────── */
[data-testid="stMainBlockContainer"],
.block-container {
    padding-left:  0 !important;
    padding-right: 0 !important;
    padding-top:   52px !important;
    max-width:     100% !important;
}
section[data-testid="stMain"] {
    background-color: #edf4fc !important;
}

/* ── Open pane columns — independently scrollable ────────────────────────── */
[data-testid="stColumn"]:has(#doc-col-marker) {
    background-color: #b8d0eb !important;
    border-right:     2px solid #6fa3d8 !important;
    height:           calc(100vh - 52px);
    overflow-y:       auto;
    overflow-x:       hidden;
}
[data-testid="stColumn"]:has(#summary-pane-marker) {
    background-color: #d4e8f8 !important;
    border-right:     2px solid #6fa3d8 !important;
    height:           calc(100vh - 52px);
    overflow-y:       auto;
    overflow-x:       hidden;
}
[data-testid="stColumn"]:has(#chat-col-marker) {
    height:           calc(100vh - 52px);
    overflow-y:       auto;
    overflow-x:       hidden;
}

/* ── Collapsed pane tabs (narrow expand strips) ──────────────────────────── */
[data-testid="stColumn"]:has(#doc-tab-marker),
[data-testid="stColumn"]:has(#sum-tab-marker) {
    height:   calc(100vh - 52px);
    overflow: hidden;
    cursor:   pointer;
}
[data-testid="stColumn"]:has(#doc-tab-marker) {
    background-color: #b8d0eb !important;
    border-right:     2px solid #6fa3d8 !important;
}
[data-testid="stColumn"]:has(#sum-tab-marker) {
    background-color: #d4e8f8 !important;
    border-right:     2px solid #6fa3d8 !important;
}
/* Strip default vertical-block padding so the arrow button fills the narrow column */
[data-testid="stColumn"]:has(#doc-tab-marker) [data-testid="stVerticalBlock"],
[data-testid="stColumn"]:has(#sum-tab-marker) [data-testid="stVerticalBlock"] {
    padding: 0 !important;
    gap:     0 !important;
}
/* Expand-arrow button style */
[data-testid="stColumn"]:has(#doc-tab-marker) button,
[data-testid="stColumn"]:has(#sum-tab-marker) button {
    background:  transparent !important;
    border:      none !important;
    box-shadow:  none !important;
    color:       #1a3558 !important;
    font-size:   1.2rem !important;
    font-weight: 900 !important;
    min-height:  64px !important;
    width:       100% !important;
    padding:     10px 0 !important;
}
[data-testid="stColumn"]:has(#doc-tab-marker) button:hover,
[data-testid="stColumn"]:has(#sum-tab-marker) button:hover {
    background: rgba(26, 53, 88, 0.14) !important;
    color:      #2563a8 !important;
}

/* ── Responsive — tablet (≤ 1024px) ─────────────────────────────────────── */
@media (max-width: 1024px) {
    .profrag-navbar-title { font-size: 1.25rem; }
}

/* ── Responsive — mobile (≤ 768px) ──────────────────────────────────────── */
@media (max-width: 768px) {
    .profrag-navbar-title { font-size: 1rem; letter-spacing: 0.05em; }
    [data-testid="stHorizontalBlock"]:has(#chat-col-marker) {
        flex-wrap: wrap !important;
    }
    [data-testid="stColumn"]:has(#doc-col-marker),
    [data-testid="stColumn"]:has(#doc-tab-marker),
    [data-testid="stColumn"]:has(#summary-pane-marker),
    [data-testid="stColumn"]:has(#sum-tab-marker),
    [data-testid="stColumn"]:has(#chat-col-marker) {
        min-width:     100% !important;
        width:         100% !important;
        height:        auto !important;
        max-height:    55vh;
        border-right:  none !important;
        border-bottom: 2px solid #6fa3d8;
    }
}
</style>
"""

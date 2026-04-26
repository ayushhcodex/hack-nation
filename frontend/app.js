/* ==========================================
   HealthBricks India — Main Application Logic
   ==========================================*/

const API = '';
let map = null;
let facilityMarkers = null;
let desertMarkers = null;
let statsData = null;

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupSearch();
    setupModal();
    loadStats();
    loadAuditFilters();
});

// ===== TAB NAVIGATION =====
function setupTabs() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`section-${target}`).classList.add('active');
            
            if (target === 'deserts' && !map) initMap();
            if (target === 'audit') loadAuditData();
            if (target === 'insights') loadInsights();
        });
    });
}

// ===== STATS BAR =====
async function loadStats() {
    try {
        const res = await fetch(`${API}/api/stats`);
        statsData = await res.json();
        
        document.getElementById('stat-total').textContent = statsData.total_facilities?.toLocaleString() || '—';
        document.getElementById('stat-trust').textContent = statsData.avg_trust_score?.toFixed(2) || '—';
        document.getElementById('stat-deserts').textContent = statsData.total_desert_regions?.toLocaleString() || '—';
        document.getElementById('stat-contradictions').textContent = statsData.total_contradictions?.toLocaleString() || '—';
    } catch (e) {
        console.error('Stats load failed:', e);
    }
}

// ===== SEARCH & QUERY =====
function setupSearch() {
    const btn = document.getElementById('search-btn');
    const input = document.getElementById('query-input');
    
    btn.addEventListener('click', runSearch);
    input.addEventListener('keypress', e => { if (e.key === 'Enter') runSearch(); });
}

async function runSearch() {
    const query = document.getElementById('query-input').value.trim();
    if (!query) return;
    
    const btn = document.getElementById('search-btn');
    btn.innerHTML = '<span class="loading-spinner"></span>';
    clearStrategy();
    
    try {
        const res = await fetch(`${API}/api/query`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query, top_k: 20 }),
        });
        const data = await res.json();
        
        document.getElementById('discover-results').style.display = 'block';
        renderResults(data.results || [], data.summary || {}, data.exact_match !== false);
        renderQueryEvaluation(data.evaluation || null);
        renderChainOfThought(data.chain_of_thought || []);

        // Trigger Genie Strategy in background
        loadGenieStrategy(query);
    } catch (e) {
        console.error('Query failed:', e);
        document.getElementById('facility-results').innerHTML = 
            '<div class="empty-state"><div class="empty-state-icon">⚠️</div><p>Query failed. Make sure the API is running.</p></div>';
        const evalContainer = document.getElementById('query-eval');
        if (evalContainer) evalContainer.innerHTML = '';
        document.getElementById('discover-results').style.display = 'block';
    }
    
    btn.innerHTML = '<span>Search</span><span class="btn-arrow">→</span>';
}

function clearStrategy() {
    const container = document.getElementById('strategy-container');
    container.innerHTML = '';
    container.style.display = 'none';
}

async function loadGenieStrategy(query) {
    const container = document.getElementById('strategy-container');
    container.style.display = 'block';
    container.innerHTML = `
        <div class="strategy-card">
            <div class="strategy-header">
                <div class="strategy-label">🧠 Agent Planning Intervention...</div>
            </div>
            <div class="loading-spinner" style="margin: 0 auto;"></div>
        </div>
    `;

    try {
        const res = await fetch(`${API}/api/genie/strategy`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query }),
        });
        const data = await res.json();
        if (data.status === 'ok' && data.data) {
            renderGenieStrategy(data.data);
        } else {
            container.style.display = 'none';
        }
    } catch (e) {
        console.error('Genie failed:', e);
        container.style.display = 'none';
    }
}

function renderGenieStrategy(data) {
    const container = document.getElementById('strategy-container');
    const priority = data.priority || 'medium';
    
    // Simple markdown-ish bolding
    let strategy = data.strategy || '';
    strategy = strategy.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    strategy = strategy.replace(/\n\n/g, '<br><br>');
    strategy = strategy.replace(/- (.*)/g, '<li>$1</li>');
    if (strategy.includes('<li>')) strategy = '<ul>' + strategy + '</ul>';

    container.innerHTML = `
        <div class="strategy-card">
            <div class="strategy-header">
                <div class="strategy-label">🚀 Agentic Crisis Strategy Recommended</div>
                <div class="strategy-priority ${priority}">${priority}</div>
            </div>
            <div class="strategy-body">
                ${strategy}
            </div>
        </div>
    `;
}

function renderQueryEvaluation(evaluation) {
    const container = document.getElementById('query-eval');
    if (!container) return;
    if (!evaluation) {
        container.innerHTML = '';
        return;
    }

    const rubric = evaluation.rubric || {};
    const diagnostics = evaluation.diagnostics || {};
    const overall = Number(evaluation.overall_score || 0);
    const label = evaluation.confidence_label || 'low';

    container.innerHTML = `
        <div class="facility-card" style="margin-bottom: 14px;">
            <div class="facility-header">
                <div>
                    <div class="facility-name">RAG Evaluation</div>
                    <div class="facility-location">Estimated answer quality for this query</div>
                </div>
                <div class="trust-badge trust-${escapeHtml(label)}">
                    <div class="trust-gauge" style="--pct: ${Math.round(overall)}"><span>${Math.round(overall)}%</span></div>
                    <span class="trust-label">${escapeHtml(label)}</span>
                </div>
            </div>
            <div class="facility-location">Discovery & Verification (35%): <strong>${rubric.discovery_verification ?? 0}%</strong></div>
            <div class="facility-location">IDP Innovation (30%): <strong>${rubric.idp_innovation ?? 0}%</strong></div>
            <div class="facility-location">Social Impact & Utility (25%): <strong>${rubric.social_impact_utility ?? 0}%</strong></div>
            <div class="facility-location">UX Transparency (10%): <strong>${rubric.ux_transparency ?? 0}%</strong></div>
            <div class="facility-location" style="margin-top: 8px;">Exact Match: <strong>${diagnostics.exact_match ? 'Yes' : 'No (relaxed fallback)'}</strong></div>
            <div class="facility-location">Evidence Coverage: <strong>${diagnostics.evidence_coverage ?? 0}%</strong> • Consistency: <strong>${diagnostics.consistency ?? 0}%</strong></div>
            <div class="facility-location">Grounded Evidence: <strong>${diagnostics.grounded_evidence ?? 0}%</strong> • Cross-pass Corroboration: <strong>${diagnostics.cross_pass_corroboration ?? 0}%</strong></div>
        </div>
    `;
}

function renderResults(results, summary = {}, exactMatch = true) {
    const container = document.getElementById('facility-results');
    const hasBedSummary = Number.isFinite(summary.total_beds) || Number.isFinite(summary.facilities_with_bed_data);
    
    if (!results.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🔍</div><p>No facilities found matching your query.</p></div>';
        return;
    }

    const fallbackNotice = !exactMatch ? `
        <div class="facility-card" style="border-left: 4px solid #e5a50a;">
            <div class="facility-name">No Exact Capability Match</div>
            <div class="facility-location">Showing best metadata-matched facilities (state/type). Try broader query or refresh 10k outputs.</div>
        </div>
    ` : '';

    const summaryHtml = hasBedSummary ? `
        <div class="facility-card" style="border-left: 4px solid var(--accent, #2ec27e);">
            <div class="facility-name">Bed Availability Summary</div>
            <div class="facility-location">Total Beds (matched set): <strong>${summary.total_beds ?? 0}</strong></div>
            <div class="facility-location">Facilities with bed data: <strong>${summary.facilities_with_bed_data ?? 0}</strong></div>
            ${Number.isFinite(summary.matched_facilities) ? `<div class="facility-location">Matched facilities: <strong>${summary.matched_facilities}</strong></div>` : ''}
            ${Number.isFinite(summary.bed_data_coverage_pct) ? `<div class="facility-location">Bed data coverage: <strong>${summary.bed_data_coverage_pct}%</strong></div>` : ''}
            ${Number.isFinite(summary.oxygen_facilities_matched) ? `<div class="facility-location">Oxygen-matched facilities: <strong>${summary.oxygen_facilities_matched}</strong></div>` : ''}
        </div>
    ` : '';

    container.innerHTML = fallbackNotice + summaryHtml + results.map((f, i) => {
        const trustPct = Math.round((f.trust_score || 0) * 100);
        const band = f.trust_band || 'low';
        const contradictions = parseJSON(f.contradiction_flags);
        const evidence = parseEvidence(f.citations || f.extraction_evidence);
        
        return `
        <div class="facility-card" onclick="openFacilityModal('${f.facility_id || ''}')" style="animation-delay: ${i * 0.05}s">
            <div class="facility-header">
                <div>
                    <div class="facility-name">${escapeHtml(f.name || 'Unknown')}</div>
                    <div class="facility-location">
                        📍 ${escapeHtml(f.address_city || '')}${f.address_stateOrRegion ? ', ' + escapeHtml(f.address_stateOrRegion) : ''}
                        ${f.distance_km && f.distance_km < 10000 ? ` • ${f.distance_km.toFixed(1)}km` : ''}
                    </div>
                </div>
                <div class="trust-badge trust-${band}">
                    <div class="trust-gauge" style="--pct: ${trustPct}"><span>${trustPct}%</span></div>
                    <span class="trust-label">${band}</span>
                </div>
            </div>
            
            <div class="capability-pills">
                ${(f.matched_capabilities || '').split(', ').filter(Boolean).map(c => 
                    `<span class="cap-pill matched">${c.replace('has_', '').replace(/_/g, ' ')}</span>`
                ).join('')}
            </div>
            
            ${contradictions.length ? contradictions.slice(0, 2).map(c => 
                `<div class="contradiction-alert">⚠️ ${escapeHtml(c)}</div>`
            ).join('') : ''}
            
            ${evidence.length ? `
            <div class="evidence-section">
                <button class="evidence-toggle" onclick="event.stopPropagation(); this.nextElementSibling.classList.toggle('show');">
                    ▸ ${evidence.length} evidence citation${evidence.length > 1 ? 's' : ''}
                </button>
                <div class="evidence-list">
                    ${evidence.slice(0, 5).map(e => `<div class="evidence-item">"${escapeHtml(e)}"</div>`).join('')}
                </div>
            </div>` : ''}
        </div>`;
    }).join('');
}

function renderChainOfThought(steps) {
    const container = document.getElementById('chain-of-thought');
    const icons = { QueryParser: '🔍', Filter: '📊', Verifier: '🛡️', Ranker: '⚖️', Extractor: '🧬', Consensus: '✅' };
    
    container.innerHTML = steps.map((s, i) => `
        <div class="cot-step" style="animation-delay: ${i * 0.15}s">
            <div class="cot-icon">${icons[s.agent] || '🤖'}</div>
            <div class="cot-body">
                <div class="cot-agent">${escapeHtml(s.agent || 'Agent')}</div>
                <div class="cot-action">${escapeHtml(s.action || '')}</div>
                <div class="cot-detail">${escapeHtml(s.detail || s.state_filter || '')}</div>
            </div>
        </div>
    `).join('');
}

// ===== DESERT MAP =====
function initMap() {
    map = L.map('india-map').setView([22.5, 82], 5);
    
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        maxZoom: 18,
    }).addTo(map);
    
    loadMapData();
}

async function loadMapData() {
    try {
        // Load facilities for markers
        const facRes = await fetch(`${API}/api/map/facilities`);
        const facData = await facRes.json();
        renderFacilityMarkers(facData.facilities || []);
        
        // Load deserts for overlay
        const desRes = await fetch(`${API}/api/deserts/geojson`);
        const desData = await desRes.json();
        renderDesertOverlay(desData.features || []);
        
        // Load desert sidebar
        const listRes = await fetch(`${API}/api/deserts?risk_tier=critical&limit=30`);
        const listData = await listRes.json();
        renderDesertSidebar(listData.results || []);
        
        // Load recommendations
        const recRes = await fetch(`${API}/api/recommendations`);
        const recData = await recRes.json();
        if (recData.recommendations?.length) {
            renderRecommendations(recData.recommendations);
        }
    } catch (e) {
        console.error('Map data load failed:', e);
    }
}

function renderFacilityMarkers(facilities) {
    if (facilityMarkers) {
        map.removeLayer(facilityMarkers);
    }
    
    facilityMarkers = L.markerClusterGroup({
        chunkedLoading: true,
        maxClusterRadius: 50,
        spiderfyOnMaxZoom: true,
        iconCreateFunction: function(cluster) {
            const count = cluster.getChildCount();
            let size = 'small';
            if (count > 50) size = 'medium';
            if (count > 200) size = 'large';
            return L.divIcon({
                html: `<div><span>${count}</span></div>`,
                className: `custom-cluster custom-cluster-${size}`,
                iconSize: L.point(40, 40)
            });
        }
    });
    
    const trustColors = { high: '#10b981', medium: '#f59e0b', low: '#ef4444' };
    
    facilities.forEach(f => {
        if (!f.latitude || !f.longitude) return;
        const color = trustColors[f.trust_band] || '#64748b';
        const radius = f.facilityTypeId === 'hospital' ? 6 : 4;
        
        const marker = L.circleMarker([f.latitude, f.longitude], {
            radius,
            fillColor: color,
            color: color,
            weight: 1,
            opacity: 0.9,
            fillOpacity: 0.7,
        });
        
        marker.bindPopup(`
            <div class="map-popup">
                <h4>${escapeHtml(f.name)}</h4>
                <p>📍 ${escapeHtml(f.address_city || '')}, ${escapeHtml(f.address_stateOrRegion || '')}</p>
                <p>Type: ${f.facilityTypeId || 'N/A'}</p>
                <p>Trust: <span class="trust-inline" style="color:${color}">${((f.trust_score || 0) * 100).toFixed(0)}% (${f.trust_band})</span></p>
                <p>Capabilities: ${f.capabilities_found || 0}</p>
                <button class="popup-detail-btn" onclick="openFacilityModal('${f.facility_id}')">View Full Audit Report</button>
                <a href="https://www.google.com/maps/dir/?api=1&destination=${f.latitude},${f.longitude}" target="_blank" class="popup-detail-btn" style="background:#3b82f6; margin-top: 6px; text-align: center; display: block; text-decoration: none; color: white;">Get Directions</a>
            </div>
        `);
        
        facilityMarkers.addLayer(marker);
    });
    
    map.addLayer(facilityMarkers);
}

function renderDesertOverlay(features) {
    desertMarkers = L.layerGroup();
    
    features.forEach(f => {
        const props = f.properties;
        const coords = f.geometry?.coordinates;
        if (!coords) return;
        
        const score = props.desert_score || 0;
        const color = score >= 0.8 ? '#ef4444' : score >= 0.6 ? '#f97316' : score >= 0.3 ? '#eab308' : '#22c55e';
        const radius = Math.max(8, score * 25);
        
        const marker = L.circleMarker([coords[1], coords[0]], {
            radius,
            fillColor: color,
            color: color,
            weight: 2,
            opacity: 0.4,
            fillOpacity: 0.15,
        });
        
        marker.bindPopup(`
            <div class="map-popup">
                <h4>🔴 ${escapeHtml(props.state)}, PIN ${props.pin_code}</h4>
                <p>Desert Score: <strong>${(score * 100).toFixed(0)}%</strong> (${props.risk_tier})</p>
                <p>Facilities: ${props.facilities_in_region} (${props.hospitals_in_region} hospitals)</p>
                <p>Missing: ${escapeHtml(props.missing_high_acuity || 'None')}</p>
            </div>
        `);
        
        desertMarkers.addLayer(marker);
    });
    
    desertMarkers.addTo(map);
}

function renderDesertSidebar(deserts) {
    const container = document.getElementById('desert-list');
    container.innerHTML = deserts.slice(0, 25).map(d => {
        const riskClass = `risk-${d.risk_tier || 'moderate'}`;
        return `
        <div class="desert-item" onclick="map.setView([${d.region_lat || 22}, ${d.region_lon || 82}], 10)">
            <div class="desert-item-header">
                <span class="desert-region">${escapeHtml(d.address_stateOrRegion || '')}</span>
                <span class="desert-score-badge ${riskClass}">${((d.desert_score || 0) * 100).toFixed(0)}%</span>
            </div>
            <div class="desert-meta">PIN ${d.address_zipOrPostcode || 'N/A'} • ${d.facilities_in_region || 0} facilities</div>
            ${d.missing_high_acuity && d.missing_high_acuity !== 'None' ? 
                `<div class="desert-missing">Missing: ${escapeHtml(d.missing_high_acuity)}</div>` : ''}
        </div>`;
    }).join('');
}

function renderRecommendations(recs) {
    const panel = document.getElementById('recommendations-panel');
    const list = document.getElementById('recommendations-list');
    panel.style.display = 'block';
    
    list.innerHTML = recs.slice(0, 10).map(r => `
        <div class="rec-card">
            <div class="rec-location">${escapeHtml(r.state)}, PIN ${r.pin_code}</div>
            <div class="rec-detail">${r.existing_facilities} existing facilities (${r.existing_hospitals} hospitals)</div>
            <div class="rec-detail">Missing: ${escapeHtml(r.missing_capabilities || 'N/A')}</div>
            ${(r.priority_deployments || []).slice(0, 2).map(p => 
                `<div class="rec-priority">🎯 Deploy ${p.capability}: nearest is ${p.nearest_km}km away (${p.urgency})</div>`
            ).join('')}
        </div>
    `).join('');
}

// ===== TRUST AUDIT =====
let auditPage = 0;
const AUDIT_PAGE_SIZE = 50;

async function loadAuditFilters() {
    try {
        const res = await fetch(`${API}/api/stats`);
        const data = await res.json();
        const select = document.getElementById('audit-state-filter');
        if (data.state_summary) {
            Object.keys(data.state_summary).sort().forEach(state => {
                const opt = document.createElement('option');
                opt.value = state;
                opt.textContent = state;
                select.appendChild(opt);
            });
        }
    } catch (e) {}
    
    ['audit-state-filter', 'audit-trust-filter', 'audit-type-filter'].forEach(id => {
        document.getElementById(id).addEventListener('change', () => { auditPage = 0; loadAuditData(); });
    });
}

async function loadAuditData() {
    const state = document.getElementById('audit-state-filter').value;
    const trust = document.getElementById('audit-trust-filter').value;
    const type = document.getElementById('audit-type-filter').value;
    
    const params = new URLSearchParams({ limit: AUDIT_PAGE_SIZE, offset: auditPage * AUDIT_PAGE_SIZE });
    if (state) params.set('state', state);
    if (trust) params.set('trust_band', trust);
    if (type) params.set('facility_type', type);
    
    try {
        const res = await fetch(`${API}/api/facilities?${params}`);
        const data = await res.json();
        renderAuditTable(data.results || [], data.total || 0);
        loadRedFlags();
    } catch (e) {
        console.error('Audit load failed:', e);
    }
}

function renderAuditTable(results, total) {
    const tbody = document.getElementById('audit-table-body');
    
    tbody.innerHTML = results.map(f => {
        const trustColor = f.trust_band === 'high' ? 'var(--trust-high)' : f.trust_band === 'medium' ? 'var(--trust-medium)' : 'var(--trust-low)';
        const trustIcon = f.trust_band === 'high' ? '✅' : f.trust_band === 'medium' ? '⚠️' : '🔴';
        
        return `
        <tr onclick="openFacilityModal('${f.facility_id || ''}')">
            <td><strong>${escapeHtml(f.name || 'Unknown')}</strong></td>
            <td>${escapeHtml(f.address_city || '')}, ${escapeHtml(f.address_stateOrRegion || '')}</td>
            <td>${f.facilityTypeId || 'N/A'}</td>
            <td style="color:${trustColor}">${trustIcon} ${((f.trust_score || 0) * 100).toFixed(0)}%</td>
            <td>${f.contradiction_count || 0}</td>
            <td>${f.capabilities_found || 0}</td>
        </tr>`;
    }).join('');
    
    // Pagination
    const totalPages = Math.ceil(total / AUDIT_PAGE_SIZE);
    const pagination = document.getElementById('audit-pagination');
    let paginationHtml = '';
    for (let i = 0; i < Math.min(totalPages, 10); i++) {
        paginationHtml += `<button class="page-btn ${i === auditPage ? 'active' : ''}" onclick="auditPage=${i}; loadAuditData();">${i + 1}</button>`;
    }
    if (totalPages > 10) paginationHtml += `<span style="color:var(--text-muted);padding:6px">... of ${totalPages}</span>`;
    pagination.innerHTML = paginationHtml;
}

async function loadRedFlags() {
    try {
        const res = await fetch(`${API}/api/facilities?trust_band=low&limit=10`);
        const data = await res.json();
        const container = document.getElementById('red-flag-list');
        
        container.innerHTML = (data.results || []).slice(0, 8).map(f => {
            const flags = parseJSON(f.contradiction_flags);
            return `
            <div class="red-flag-item">
                <span>${escapeHtml(f.name || 'Unknown')} — ${escapeHtml(f.address_stateOrRegion || '')}</span>
                <span style="color:var(--trust-low)">${flags[0] || 'Low trust'}</span>
            </div>`;
        }).join('');
    } catch (e) {}
}

// ===== INSIGHTS =====
async function loadInsights() {
    if (!statsData) {
        try {
            const res = await fetch(`${API}/api/stats`);
            statsData = await res.json();
        } catch (e) { return; }
    }
    
    renderTrustDistribution(statsData.trust_distribution || {});
    renderCapabilityChart(statsData.capability_counts || {});
    renderFacilityTypeChart(statsData.facility_type_distribution || {});
    renderStateChart(statsData.state_summary || {});
    renderDesertRiskChart(statsData.desert_risk_distribution || {});
    renderKeyFindings(statsData);
}

function renderTrustDistribution(dist) {
    const container = document.getElementById('chart-trust');
    const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
    const colors = { high: 'green', medium: 'orange', low: 'blue' };
    const labels = { high: '✅ High Trust', medium: '⚠️ Medium Trust', low: '🔴 Low Trust' };
    
    container.innerHTML = `<div class="bar-chart">
        ${Object.entries(dist).map(([band, count]) => `
            <div class="bar-row">
                <span class="bar-label">${labels[band] || band}</span>
                <div class="bar-track">
                    <div class="bar-fill ${colors[band] || 'blue'}" style="width:${(count/total*100).toFixed(1)}%">${count}</div>
                </div>
            </div>
        `).join('')}
    </div>`;
}

function renderCapabilityChart(caps) {
    const container = document.getElementById('chart-capabilities');
    const entries = Object.entries(caps)
        .filter(([k]) => k.startsWith('has_'))
        .sort((a, b) => b[1] - a[1])
        .slice(0, 15);
    const max = entries[0]?.[1] || 1;
    
    container.innerHTML = `<div class="bar-chart">
        ${entries.map(([cap, count]) => `
            <div class="bar-row">
                <span class="bar-label">${cap.replace('has_', '').replace(/_/g, ' ')}</span>
                <div class="bar-track">
                    <div class="bar-fill teal" style="width:${(count/max*100).toFixed(1)}%">${count}</div>
                </div>
            </div>
        `).join('')}
    </div>`;
}

function renderFacilityTypeChart(types) {
    const container = document.getElementById('chart-types');
    const total = Object.values(types).reduce((a, b) => a + b, 0) || 1;
    const colors = ['blue', 'teal', 'purple', 'orange', 'green'];
    
    container.innerHTML = `<div class="bar-chart">
        ${Object.entries(types).sort((a,b) => b[1]-a[1]).map(([type, count], i) => `
            <div class="bar-row">
                <span class="bar-label">${type}</span>
                <div class="bar-track">
                    <div class="bar-fill ${colors[i % colors.length]}" style="width:${(count/total*100).toFixed(1)}%">${count} (${(count/total*100).toFixed(0)}%)</div>
                </div>
            </div>
        `).join('')}
    </div>`;
}

function renderStateChart(states) {
    const container = document.getElementById('chart-states');
    const entries = Object.entries(states).sort((a, b) => b[1].count - a[1].count).slice(0, 15);
    const max = entries[0]?.[1]?.count || 1;
    
    container.innerHTML = `<div class="bar-chart">
        ${entries.map(([state, data]) => `
            <div class="bar-row">
                <span class="bar-label">${state}</span>
                <div class="bar-track">
                    <div class="bar-fill purple" style="width:${(data.count/max*100).toFixed(1)}%">${data.count} (avg trust: ${(data.avg_trust*100).toFixed(0)}%)</div>
                </div>
            </div>
        `).join('')}
    </div>`;
}

function renderDesertRiskChart(dist) {
    const container = document.getElementById('chart-desert-risk');
    const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
    const riskColors = { critical: 'var(--desert-critical)', high: 'var(--desert-high)', moderate: 'var(--desert-moderate)', low: 'var(--desert-low)' };
    
    container.innerHTML = `<div class="bar-chart">
        ${['critical', 'high', 'moderate', 'low'].map(tier => {
            const count = dist[tier] || 0;
            return `
            <div class="bar-row">
                <span class="bar-label">${tier.toUpperCase()}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width:${(count/total*100).toFixed(1)}%; background:${riskColors[tier]}">${count} regions</div>
                </div>
            </div>`;
        }).join('')}
    </div>`;
}

function renderKeyFindings(data) {
    const container = document.getElementById('key-findings');
    const caps = data.capability_counts || {};
    
    const findings = [
        { icon: '🏥', text: `<strong>${data.total_facilities?.toLocaleString()}</strong> medical facilities analyzed across <strong>${Object.keys(data.state_summary || {}).length}</strong> states` },
        { icon: '🛡️', text: `Average trust score: <strong>${(data.avg_trust_score * 100).toFixed(1)}%</strong> with <strong>${data.total_contradictions}</strong> contradictions detected` },
        { icon: '🔴', text: `<strong>${(data.desert_risk_distribution?.critical || 0) + (data.desert_risk_distribution?.high || 0)}</strong> regions classified as critical or high-risk medical deserts` },
        { icon: '💉', text: `Only <strong>${caps.has_dialysis || 0}</strong> facilities (${((caps.has_dialysis || 0) / data.total_facilities * 100).toFixed(1)}%) have dialysis capability` },
        { icon: '🫁', text: `Only <strong>${caps.has_oxygen || 0}</strong> facilities mention oxygen support — yet <strong>${caps.has_icu || 0}</strong> claim ICU` },
        { icon: '🏥', text: `<strong>${caps.has_24x7 || 0}</strong> facilities claim 24/7 operations` },
    ];
    
    container.innerHTML = findings.map(f => `
        <div class="finding-item">
            <div class="finding-icon">${f.icon}</div>
            <div class="finding-text">${f.text}</div>
        </div>
    `).join('');
}

// ===== FACILITY DETAIL MODAL =====
function setupModal() {
    document.getElementById('modal-close').addEventListener('click', closeModal);
    document.getElementById('facility-modal').addEventListener('click', e => {
        if (e.target.id === 'facility-modal') closeModal();
    });
}

async function openFacilityModal(facilityId) {
    if (!facilityId) return;
    const modal = document.getElementById('facility-modal');
    const body = document.getElementById('modal-body');
    
    modal.style.display = 'flex';
    body.innerHTML = '<div style="text-align:center;padding:40px"><span class="loading-spinner"></span></div>';
    
    try {
        const res = await fetch(`${API}/api/facilities/${facilityId}`);
        const f = await res.json();
        
        const trustPct = Math.round((f.trust_score || 0) * 100);
        const band = f.trust_band || 'low';
        const contradictions = parseJSON(f.contradiction_flags);
        const evidence = parseJSON(f.extraction_evidence);
        const trustReasoning = parseJSON(f.trust_reasoning);
        const validatorIssues = parseJSON(f.validator_issues);
        
        body.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px">
                <div>
                    <h2 style="font-size:20px;margin-bottom:4px">${escapeHtml(f.name || 'Unknown')}</h2>
                    <p style="color:var(--text-secondary);font-size:14px">
                        📍 ${escapeHtml(f.address_line1 || '')} ${escapeHtml(f.address_city || '')}, ${escapeHtml(f.address_stateOrRegion || '')} ${f.address_zipOrPostcode || ''}
                    </p>
                    <p style="color:var(--text-muted);font-size:12px;margin-top:4px">
                        Type: ${f.facilityTypeId || 'N/A'} • Operator: ${f.operatorTypeId || 'N/A'} • ID: ${f.facility_id || 'N/A'}
                    </p>
                </div>
                <div class="trust-badge trust-${band}">
                    <div class="trust-gauge" style="--pct:${trustPct}"><span>${trustPct}%</span></div>
                    <span class="trust-label">${band}</span>
                </div>
            </div>
            
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
                <div style="background:var(--bg-card);border-radius:8px;padding:12px">
                    <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase">Confidence Range</div>
                    <div style="font-size:18px;font-weight:700">${((f.confidence_low || 0) * 100).toFixed(0)}% — ${((f.confidence_high || 0) * 100).toFixed(0)}%</div>
                </div>
                <div style="background:var(--bg-card);border-radius:8px;padding:12px">
                    <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase">Capabilities Found</div>
                    <div style="font-size:18px;font-weight:700">${f.capabilities_found || 0} <span style="font-size:12px;color:var(--text-secondary)">(avg conf: ${((f.avg_confidence || 0) * 100).toFixed(0)}%)</span></div>
                </div>
            </div>
            
            ${contradictions.length ? `
            <h3 style="font-size:14px;color:#fca5a5;margin-bottom:8px">⚠️ Contradictions (${contradictions.length})</h3>
            ${contradictions.map(c => `<div class="contradiction-alert">${escapeHtml(c)}</div>`).join('')}
            ` : '<p style="color:var(--trust-high);margin-bottom:12px">✅ No contradictions detected</p>'}
            
            ${f.peer_comparison_flag ? `<div class="contradiction-alert" style="border-color:rgba(251,146,60,0.3);background:rgba(251,146,60,0.08);color:var(--accent-orange)">👥 ${escapeHtml(f.peer_comparison_flag)}</div>` : ''}
            
            <h3 style="font-size:14px;margin:16px 0 8px">📄 Extraction Evidence</h3>
            <div style="max-height:300px;overflow-y:auto">
            ${Object.entries(evidence).filter(([,v]) => v?.evidence?.length || (Array.isArray(v) && v.length)).map(([cap, data]) => {
                const items = data?.evidence || (Array.isArray(data) ? data : []);
                const conf = data?.confidence;
                const reasoning = data?.reasoning;
                return `
                <div style="margin-bottom:8px;padding:8px;background:rgba(255,255,255,0.03);border-radius:6px">
                    <div style="font-size:12px;font-weight:600;color:var(--accent-teal)">${cap.replace('has_', '').replace(/_/g, ' ')}
                        ${conf ? `<span style="color:var(--text-muted);font-weight:400"> — confidence: ${(conf * 100).toFixed(0)}%</span>` : ''}
                    </div>
                    ${reasoning ? `<div style="font-size:11px;color:var(--text-muted);margin-top:2px">${escapeHtml(reasoning)}</div>` : ''}
                    ${items.map(e => `<div class="evidence-item">"${escapeHtml(typeof e === 'string' ? e : JSON.stringify(e))}"</div>`).join('')}
                </div>`;
            }).join('')}
            </div>
            
            ${trustReasoning ? `
            <h3 style="font-size:14px;margin:16px 0 8px">🧠 Trust Scoring Reasoning</h3>
            <div style="background:var(--bg-card);border-radius:8px;padding:12px;font-size:12px;color:var(--text-secondary)">
                <p>Positive signals: <strong>${trustReasoning.positive_signals}</strong> | Negative signals: <strong>${trustReasoning.negative_signals}</strong></p>
                <p>Evidence count: ${trustReasoning.evidence_count} | Data completeness: ${((trustReasoning.data_completeness || 0) * 100).toFixed(0)}%</p>
                <p style="margin-top:4px;font-family:monospace;font-size:11px">${escapeHtml(trustReasoning.formula || '')}</p>
            </div>
            ` : ''}
        `;
    } catch (e) {
        body.innerHTML = '<div class="empty-state"><p>Failed to load facility details.</p></div>';
    }
}

function closeModal() {
    document.getElementById('facility-modal').style.display = 'none';
}

// ===== UTILITIES =====
function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function parseJSON(str) {
    if (!str) return [];
    if (Array.isArray(str)) return str;
    if (typeof str === 'object') return str;
    try { return JSON.parse(str); } catch { return []; }
}

function parseEvidence(data) {
    if (!data) return [];
    const parsed = parseJSON(data);
    if (Array.isArray(parsed)) return parsed;
    // It's an object with capability keys
    const all = [];
    Object.values(parsed).forEach(v => {
        if (v?.evidence) all.push(...v.evidence);
        else if (Array.isArray(v)) all.push(...v);
    });
    return all;
}

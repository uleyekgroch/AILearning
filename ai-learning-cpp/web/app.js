/**
 * AILearning Web Console — app.js
 * Vue 3 + ECharts single-file app logic
 * No build step required.
 */

const { createApp, ref, reactive, onMounted, onUnmounted, computed, watch, nextTick } = Vue;

/* ── API helper ─────────────────────────────────────────────── */
const API = '';
const WS_URL = `ws://${location.host}/ws/events`;
const WS_STATS_URL = `ws://${location.host}/ws/stats`;

async function apiGet(path) {
  const r = await fetch(API + path);
  return r.json();
}

async function apiPost(path, body = {}) {
  const r = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}

/* ── Vue app ────────────────────────────────────────────────── */
const app = createApp({
  setup() {
    // ── reactive state ──
    const connected = ref(false);
    const health = reactive({ status: '-', version: '-', stage: '-', total_steps: 0 });
    const stats = reactive({});
    const wsConnected = ref(false);

    // learning input
    const learnText = ref('');
    const learnBusy = ref(false);

    // Q&A
    const qaQuestion = ref('');
    const qaBusy = ref(false);
    const qaHistory = ref([]);

    // operation results log
    const resultLog = ref([]);

    // knowledge graph data
    const graphNodes = ref([]);
    const graphLinks = ref([]);

    // learning progress (time series)
    const progressData = ref([]);

    // emotion state
    const emotionState = reactive({ valence: 0, arousal: 0, dominance: 0, label: '-' });

    // chart refs (set in template via ref attr)
    const progressChartRef = ref(null);
    const emotionChartRef = ref(null);
    const graphChartRef = ref(null);

    let progressChart = null;
    let emotionChart = null;
    let graphChart = null;
    let wsEvents = null;
    let wsStats = null;
    let refreshTimer = null;

    // ── computed ──
    const knowledgeCount = computed(() => stats.knowledge_count || stats.triples_count || 0);
    const stageName = computed(() => health.stage || '-');

    // ── result logging ──
    function logResult(msg, type = 'info') {
      const ts = new Date().toLocaleTimeString();
      resultLog.value.unshift({ ts, msg, type });
      if (resultLog.value.length > 200) resultLog.value.length = 200;
    }

    // ── init charts ──
    function initCharts() {
      // Progress chart
      if (progressChartRef.value) {
        progressChart = echarts.init(progressChartRef.value, 'dark');
        progressChart.setOption({
          backgroundColor: 'transparent',
          tooltip: { trigger: 'axis' },
          grid: { top: 30, right: 20, bottom: 30, left: 50 },
          xAxis: { type: 'category', data: [], axisLabel: { color: '#9a9db5' } },
          yAxis: { type: 'value', axisLabel: { color: '#9a9db5' }, splitLine: { lineStyle: { color: '#2e3348' } } },
          series: [
            { name: 'Steps', type: 'line', smooth: true, data: [], lineStyle: { color: '#4a9eff', width: 2 }, itemStyle: { color: '#4a9eff' }, areaStyle: { color: 'rgba(74,158,255,0.15)' } },
          ],
        });
      }

      // Emotion radar
      if (emotionChartRef.value) {
        emotionChart = echarts.init(emotionChartRef.value, 'dark');
        emotionChart.setOption({
          backgroundColor: 'transparent',
          radar: {
            indicator: [
              { name: 'Valence', max: 1, min: -1 },
              { name: 'Arousal', max: 1, min: -1 },
              { name: 'Dominance', max: 1, min: -1 },
            ],
            shape: 'circle',
            splitNumber: 4,
            axisName: { color: '#9a9db5' },
            splitLine: { lineStyle: { color: '#2e3348' } },
            splitArea: { areaStyle: { color: ['transparent'] } },
          },
          series: [{
            type: 'radar',
            data: [{ value: [0, 0, 0], name: 'Emotion', areaStyle: { color: 'rgba(167,139,250,0.25)' }, lineStyle: { color: '#a78bfa' }, itemStyle: { color: '#a78bfa' } }],
          }],
        });
      }

      // Knowledge graph (force-directed)
      if (graphChartRef.value) {
        graphChart = echarts.init(graphChartRef.value, 'dark');
        updateGraphChart();
      }
    }

    function updateProgressChart() {
      if (!progressChart) return;
      progressChart.setOption({
        xAxis: { data: progressData.value.map((_, i) => i) },
        series: [{ data: progressData.value }],
      });
    }

    function updateEmotionChart() {
      if (!emotionChart) return;
      emotionChart.setOption({
        series: [{ data: [{ value: [emotionState.valence, emotionState.arousal, emotionState.dominance], name: 'Emotion', areaStyle: { color: 'rgba(167,139,250,0.25)' }, lineStyle: { color: '#a78bfa' }, itemStyle: { color: '#a78bfa' } }] }],
      });
    }

    function updateGraphChart() {
      if (!graphChart) return;
      const categories = [
        { name: 'Concept' }, { name: 'Relation' }, { name: 'Insight' },
      ];
      const nodes = graphNodes.value.map(n => ({
        id: String(n.id),
        name: n.name || n.id,
        symbolSize: Math.max(20, (n.weight || 1) * 15),
        category: n.category || 0,
        itemStyle: { color: n.category === 0 ? '#4a9eff' : n.category === 1 ? '#34d399' : '#f59e0b' },
        label: { show: true, fontSize: 10, color: '#e4e6f0' },
      }));
      const links = graphLinks.value.map(l => ({
        source: String(l.source),
        target: String(l.target),
        lineStyle: { color: '#2e3348', width: 1 },
      }));
      graphChart.setOption({
        backgroundColor: 'transparent',
        tooltip: {},
        legend: { data: categories.map(c => c.name), textStyle: { color: '#9a9db5' } },
        series: [{
          type: 'graph',
          layout: 'force',
          roam: true,
          draggable: true,
          force: { repulsion: 120, gravity: 0.1, edgeLength: 80 },
          categories,
          data: nodes,
          links,
          emphasis: { focus: 'adjacency', lineStyle: { width: 3 } },
        }],
      }, true);
    }

    // ── resize handler ──
    function handleResize() {
      progressChart && progressChart.resize();
      emotionChart && emotionChart.resize();
      graphChart && graphChart.resize();
    }

    // ── WebSocket connections ──
    function connectWS() {
      // Events WS
      try {
        wsEvents = new WebSocket(WS_URL);
        wsEvents.onopen = () => { wsConnected.value = true; logResult('WS /ws/events connected', 'success'); };
        wsEvents.onclose = () => { wsConnected.value = false; logResult('WS /ws/events disconnected', 'error'); };
        wsEvents.onerror = () => { wsConnected.value = false; };
        wsEvents.onmessage = (ev) => {
          try {
            const data = JSON.parse(ev.data);
            if (Array.isArray(data)) {
              data.forEach(handleWSEvent);
            } else {
              handleWSEvent(data);
            }
          } catch { /* ignore */ }
        };
      } catch (e) {
        logResult('WS events connect failed: ' + e.message, 'error');
      }

      // Stats WS
      try {
        wsStats = new WebSocket(WS_STATS_URL);
        wsStats.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === 'stats' && msg.data) {
              Object.assign(stats, msg.data);
              // push to progress
              const steps = msg.data.total_steps || 0;
              if (progressData.value.length === 0 || steps !== progressData.value[progressData.value.length - 1]) {
                progressData.value.push(steps);
                if (progressData.value.length > 60) progressData.value.shift();
                updateProgressChart();
              }
            }
          } catch { /* ignore */ }
        };
      } catch (e) {
        logResult('WS stats connect failed: ' + e.message, 'error');
      }
    }

    function handleWSEvent(event) {
      if (!event || !event.type) return;
      if (event.type === 'knowledge_added' || event.type === 'learned') {
        // Add to graph
        if (event.subject || event.data) {
          const d = event.data || event;
          if (d.subject && d.predicate && d.object) {
            const sId = String(d.subject);
            const oId = String(d.object);
            if (!graphNodes.value.find(n => n.id === sId)) {
              graphNodes.value.push({ id: sId, name: d.subject, category: 0, weight: 1 });
            }
            if (!graphNodes.value.find(n => n.id === oId)) {
              graphNodes.value.push({ id: oId, name: d.object, category: 0, weight: 1 });
            }
            graphLinks.value.push({ source: sId, target: oId });
            // Keep graph manageable
            if (graphNodes.value.length > 80) graphNodes.value.splice(0, graphNodes.value.length - 80);
            if (graphLinks.value.length > 120) graphLinks.value.splice(0, graphLinks.value.length - 120);
            updateGraphChart();
          }
        }
        logResult(`[${event.type}] ${JSON.stringify(event.data || event).substring(0, 120)}`, 'success');
      } else {
        logResult(`[${event.type}]`, 'info');
      }
    }

    // ── API actions ──
    async function fetchHealth() {
      try {
        const data = await apiGet('/api/health');
        Object.assign(health, data);
        connected.value = data.status === 'ok';
      } catch {
        connected.value = false;
      }
    }

    async function fetchStats() {
      try {
        const data = await apiGet('/api/stats');
        Object.assign(stats, data);
      } catch { /* ignore */ }
    }

    async function doLearn() {
      const text = learnText.value.trim();
      if (!text || learnBusy.value) return;
      learnBusy.value = true;
      logResult(`Learn: "${text.substring(0, 60)}..."`, 'info');
      try {
        const result = await apiPost('/api/learn/text', { text });
        logResult('Learn result: ' + JSON.stringify(result).substring(0, 200), 'success');
        // Update graph from result
        if (result.triples) {
          for (const t of result.triples) {
            const sId = String(t.subject);
            const oId = String(t.object);
            if (!graphNodes.value.find(n => n.id === sId)) {
              graphNodes.value.push({ id: sId, name: t.subject, category: 0, weight: 1 });
            }
            if (!graphNodes.value.find(n => n.id === oId)) {
              graphNodes.value.push({ id: oId, name: t.object, category: 0, weight: 1 });
            }
            graphLinks.value.push({ source: sId, target: oId });
          }
          if (graphNodes.value.length > 80) graphNodes.value.splice(0, graphNodes.value.length - 80);
          if (graphLinks.value.length > 120) graphLinks.value.splice(0, graphLinks.value.length - 120);
          updateGraphChart();
        }
        learnText.value = '';
        fetchStats();
      } catch (e) {
        logResult('Learn error: ' + e.message, 'error');
      } finally {
        learnBusy.value = false;
      }
    }

    async function doReason() {
      const q = qaQuestion.value.trim();
      if (!q || qaBusy.value) return;
      qaBusy.value = true;
      logResult(`Reason: "${q}"`, 'info');
      try {
        const results = await apiPost('/api/reason', { question: q });
        const answers = Array.isArray(results) ? results : [results];
        qaHistory.value.unshift({ question: q, answers });
        if (qaHistory.value.length > 50) qaHistory.value.length = 50;
        logResult(`Reason: ${answers.length} results`, 'success');
        qaQuestion.value = '';
      } catch (e) {
        logResult('Reason error: ' + e.message, 'error');
      } finally {
        qaBusy.value = false;
      }
    }

    async function doThink() {
      const q = qaQuestion.value.trim();
      if (!q) return;
      logResult(`Think: "${q}"`, 'info');
      try {
        const result = await apiPost('/api/think', { question: q });
        qaHistory.value.unshift({ question: q, answers: [{ answer: result.answer, confidence: 1.0 }] });
        logResult('Think: ' + (result.answer || '').substring(0, 120), 'success');
        qaQuestion.value = '';
      } catch (e) {
        logResult('Think error: ' + e.message, 'error');
      }
    }

    async function doInsight() {
      const ctx = qaQuestion.value.trim() || 'general';
      logResult('Insight attempt...', 'info');
      try {
        const result = await apiPost('/api/insight', { problem_context: ctx });
        if (result.found) {
          logResult('Insight: ' + JSON.stringify(result.insight).substring(0, 200), 'success');
        } else {
          logResult('No insight emerged', 'info');
        }
      } catch (e) {
        logResult('Insight error: ' + e.message, 'error');
      }
    }

    async function doAnalogize() {
      const text = learnText.value.trim();
      if (!text) {
        logResult('Enter text for analogize source/target', 'error');
        return;
      }
      logResult('Analogize...', 'info');
      try {
        const result = await apiPost('/api/analogize', {
          source_concepts: [{ name: text, domain: 'source' }],
          target_concepts: [{ name: text, domain: 'target' }],
        });
        logResult('Analogize result: ' + JSON.stringify(result).substring(0, 200), 'success');
      } catch (e) {
        logResult('Analogize error: ' + e.message, 'error');
      }
    }

    async function doConsolidate() {
      logResult('Consolidating memory...', 'info');
      try {
        const result = await apiPost('/api/consolidate');
        logResult('Consolidate: ' + JSON.stringify(result).substring(0, 200), 'success');
        fetchStats();
      } catch (e) {
        logResult('Consolidate error: ' + e.message, 'error');
      }
    }

    async function doPipeline() {
      const text = learnText.value.trim() || 'observation';
      logResult('Integrated pipeline...', 'info');
      try {
        const result = await apiPost('/api/integrated/pipeline', { observation: text, domain: 'general' });
        logResult('Pipeline done: ' + JSON.stringify(result).substring(0, 200), 'success');
        fetchStats();
      } catch (e) {
        logResult('Pipeline error: ' + e.message, 'error');
      }
    }

    async function doAutonomous() {
      logResult('Starting autonomous learning (10 iterations)...', 'info');
      try {
        const result = await apiPost('/api/autonomous', { iterations: 10 });
        logResult('Autonomous task: ' + result.task_id + ' ' + result.status, 'success');
      } catch (e) {
        logResult('Autonomous error: ' + e.message, 'error');
      }
    }

    async function fetchEmotionParams() {
      try {
        const result = await apiGet('/api/integrated/emotion-params');
        if (result.valence !== undefined) {
          emotionState.valence = result.valence;
          emotionState.arousal = result.arousal;
          emotionState.dominance = result.dominance;
          emotionState.label = result.primary_emotion || '-';
          updateEmotionChart();
        }
      } catch { /* ignore */ }
    }

    // ── lifecycle ──
    onMounted(async () => {
      await fetchHealth();
      await fetchStats();
      await fetchEmotionParams();

      nextTick(() => {
        initCharts();
        connectWS();
      });

      window.addEventListener('resize', handleResize);

      // periodic refresh
      refreshTimer = setInterval(() => {
        fetchHealth();
        fetchStats();
        fetchEmotionParams();
      }, 5000);
    });

    onUnmounted(() => {
      window.removeEventListener('resize', handleResize);
      if (refreshTimer) clearInterval(refreshTimer);
      if (wsEvents) wsEvents.close();
      if (wsStats) wsStats.close();
      if (progressChart) progressChart.dispose();
      if (emotionChart) emotionChart.dispose();
      if (graphChart) graphChart.dispose();
    });

    return {
      connected, health, stats, wsConnected,
      learnText, learnBusy,
      qaQuestion, qaBusy, qaHistory,
      resultLog,
      emotionState,
      progressChartRef, emotionChartRef, graphChartRef,
      knowledgeCount, stageName,
      doLearn, doReason, doThink, doInsight, doAnalogize,
      doConsolidate, doPipeline, doAutonomous,
    };
  },
});

app.mount('#app');

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

    // Chat
    const chatMessages = ref([]);
    const chatInput = ref('');
    const chatBusy = ref(false);
    const chatSessionId = ref('');
    const chatMessagesRef = ref(null);

    // Society
    const societyAgents = ref([]);
    const societyMetrics = ref(null);
    const newAgentPort = ref('');

    // Runtime
    const runtimeStatus = reactive({ running: false, uptime: '0s', iterations: 0, retention: '0%', pending_data: 0 });

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
            { name: '学习步数', type: 'line', smooth: true, data: [], lineStyle: { color: '#4a9eff', width: 2 }, itemStyle: { color: '#4a9eff' }, areaStyle: { color: 'rgba(74,158,255,0.15)' } },
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
              { name: '效价', max: 1, min: -1 },
              { name: '唤醒度', max: 1, min: -1 },
              { name: '支配度', max: 1, min: -1 },
            ],
            shape: 'circle',
            splitNumber: 4,
            axisName: { color: '#9a9db5' },
            splitLine: { lineStyle: { color: '#2e3348' } },
            splitArea: { areaStyle: { color: ['transparent'] } },
          },
          series: [{
            type: 'radar',
            data: [{ value: [0, 0, 0], name: '情感', areaStyle: { color: 'rgba(167,139,250,0.25)' }, lineStyle: { color: '#a78bfa' }, itemStyle: { color: '#a78bfa' } }],
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
        series: [{ data: [{ value: [emotionState.valence, emotionState.arousal, emotionState.dominance], name: '情感', areaStyle: { color: 'rgba(167,139,250,0.25)' }, lineStyle: { color: '#a78bfa' }, itemStyle: { color: '#a78bfa' } }] }],
      });
    }

    function updateGraphChart() {
      if (!graphChart) return;
      const categories = [
        { name: '概念' }, { name: '关系' }, { name: '顿悟' },
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
        wsEvents.onopen = () => { wsConnected.value = true; logResult('WS /ws/events 已连接', 'success'); };
        wsEvents.onclose = () => { wsConnected.value = false; logResult('WS /ws/events 已断开', 'error'); };
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
        logResult('WS 事件连接失败: ' + e.message, 'error');
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
        logResult('WS 统计连接失败: ' + e.message, 'error');
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
      logResult(`学习: "${text.substring(0, 60)}..."`, 'info');
      try {
        const result = await apiPost('/api/learn/text', { text });
        logResult('学习结果: ' + JSON.stringify(result).substring(0, 200), 'success');
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
        logResult('学习错误: ' + e.message, 'error');
      } finally {
        learnBusy.value = false;
      }
    }

    async function doReason() {
      const q = qaQuestion.value.trim();
      if (!q || qaBusy.value) return;
      qaBusy.value = true;
      logResult(`推理: "${q}"`, 'info');
      try {
        const results = await apiPost('/api/reason', { question: q });
        const answers = Array.isArray(results) ? results : [results];
        qaHistory.value.unshift({ question: q, answers });
        if (qaHistory.value.length > 50) qaHistory.value.length = 50;
        logResult(`推理完成: ${answers.length} 条结果`, 'success');
        qaQuestion.value = '';
      } catch (e) {
        logResult('推理错误: ' + e.message, 'error');
      } finally {
        qaBusy.value = false;
      }
    }

    async function doThink() {
      const q = qaQuestion.value.trim();
      if (!q) return;
      logResult(`思考: "${q}"`, 'info');
      try {
        const result = await apiPost('/api/think', { question: q });
        qaHistory.value.unshift({ question: q, answers: [{ answer: result.answer, confidence: 1.0 }] });
        logResult('思考结果: ' + (result.answer || '').substring(0, 120), 'success');
        qaQuestion.value = '';
      } catch (e) {
        logResult('思考错误: ' + e.message, 'error');
      }
    }

    async function doInsight() {
      const ctx = qaQuestion.value.trim() || 'general';
      logResult('尝试顿悟...', 'info');
      try {
        const result = await apiPost('/api/insight', { problem_context: ctx });
        if (result.found) {
          logResult('顿悟: ' + JSON.stringify(result.insight).substring(0, 200), 'success');
        } else {
          logResult('未产生顿悟', 'info');
        }
      } catch (e) {
        logResult('顿悟错误: ' + e.message, 'error');
      }
    }

    async function doAnalogize() {
      const text = learnText.value.trim();
      if (!text) {
        logResult('请输入类比迁移的源/目标文本', 'error');
        return;
      }
      logResult('类比迁移中...', 'info');
      try {
        const result = await apiPost('/api/analogize', {
          source_concepts: [{ name: text, domain: 'source' }],
          target_concepts: [{ name: text, domain: 'target' }],
        });
        logResult('类比迁移结果: ' + JSON.stringify(result).substring(0, 200), 'success');
      } catch (e) {
        logResult('类比迁移错误: ' + e.message, 'error');
      }
    }

    async function doConsolidate() {
      logResult('正在巩固记忆...', 'info');
      try {
        const result = await apiPost('/api/consolidate');
        logResult('巩固记忆完成: ' + JSON.stringify(result).substring(0, 200), 'success');
        fetchStats();
      } catch (e) {
        logResult('巩固记忆错误: ' + e.message, 'error');
      }
    }

    async function doPipeline() {
      const text = learnText.value.trim() || 'observation';
      logResult('执行整合流水线...', 'info');
      try {
        const result = await apiPost('/api/integrated/pipeline', { observation: text, domain: 'general' });
        logResult('整合流水线完成: ' + JSON.stringify(result).substring(0, 200), 'success');
        fetchStats();
      } catch (e) {
        logResult('整合流水线错误: ' + e.message, 'error');
      }
    }

    async function doAutonomous() {
      logResult('启动自主学习（10轮）...', 'info');
      try {
        const result = await apiPost('/api/autonomous', { iterations: 10 });
        logResult('自主任务: ' + result.task_id + ' ' + result.status, 'success');
      } catch (e) {
        logResult('自主学习错误: ' + e.message, 'error');
      }
    }

    // ── Chat API ──
    async function doChat() {
      const text = chatInput.value.trim();
      if (!text || chatBusy.value) return;
      chatBusy.value = true;
      chatMessages.value.push({ role: 'user', content: text });
      chatInput.value = '';
      scrollChatToBottom();
      logResult(`对话: "${text.substring(0, 60)}"`, 'info');
      try {
        const body = { message: text };
        if (chatSessionId.value) body.session_id = chatSessionId.value;
        const result = await apiPost('/api/chat', body);
        if (result.session_id) chatSessionId.value = result.session_id;
        const learned = result.learned_knowledge || [];
        chatMessages.value.push({ role: 'assistant', content: result.response || result.reply || JSON.stringify(result), learned });
        scrollChatToBottom();
        logResult('对话回复: ' + (result.response || '').substring(0, 100), 'success');
        fetchStats();
      } catch (e) {
        chatMessages.value.push({ role: 'assistant', content: '错误: ' + e.message, learned: [] });
        logResult('对话错误: ' + e.message, 'error');
      } finally {
        chatBusy.value = false;
      }
    }

    async function fetchChatHistory() {
      try {
        const result = await apiGet('/api/chat/history' + (chatSessionId.value ? '?session_id=' + chatSessionId.value : ''));
        if (Array.isArray(result)) {
          chatMessages.value = result.map(m => ({
            role: m.role,
            content: m.content || m.message,
            learned: m.learned_knowledge || [],
          }));
          scrollChatToBottom();
        }
      } catch { /* ignore */ }
    }

    function scrollChatToBottom() {
      nextTick(() => {
        const el = chatMessagesRef.value;
        if (el) el.scrollTop = el.scrollHeight;
      });
    }

    // ── Society API ──
    async function createAgent() {
      const port = parseInt(newAgentPort.value.trim());
      if (!port || port < 1 || port > 65535) {
        logResult('请输入有效端口号 (1-65535)', 'error');
        return;
      }
      logResult(`创建智能体 port=${port}...`, 'info');
      try {
        const result = await apiPost('/api/society/agents', { port });
        logResult('智能体创建: ' + JSON.stringify(result).substring(0, 120), 'success');
        newAgentPort.value = '';
        fetchSocietyMetrics();
      } catch (e) {
        logResult('创建智能体错误: ' + e.message, 'error');
      }
    }

    async function removeAgent(agent) {
      logResult(`移除智能体 ${agent.host}:${agent.port}...`, 'info');
      try {
        await apiPost('/api/society/agents/remove', { host: agent.host, port: agent.port });
        logResult('智能体已移除', 'success');
        fetchSocietyMetrics();
      } catch (e) {
        logResult('移除智能体错误: ' + e.message, 'error');
      }
    }

    async function fetchSocietyMetrics() {
      try {
        const result = await apiGet('/api/society/metrics');
        societyMetrics.value = result;
        if (Array.isArray(result.agents)) {
          societyAgents.value = result.agents;
        }
      } catch { /* ignore */ }
    }

    async function triggerObservation() {
      logResult('触发观察学习...', 'info');
      try {
        const result = await apiPost('/api/society/observe');
        logResult('观察学习结果: ' + JSON.stringify(result).substring(0, 200), 'success');
        fetchSocietyMetrics();
        fetchStats();
      } catch (e) {
        logResult('观察学习错误: ' + e.message, 'error');
      }
    }

    // ── Runtime API ──
    async function startRuntime() {
      logResult('启动持续学习...', 'info');
      try {
        const result = await apiPost('/api/runtime/start');
        runtimeStatus.running = true;
        logResult('持续学习已启动: ' + JSON.stringify(result).substring(0, 120), 'success');
      } catch (e) {
        logResult('启动错误: ' + e.message, 'error');
      }
    }

    async function stopRuntime() {
      logResult('停止持续学习...', 'info');
      try {
        const result = await apiPost('/api/runtime/stop');
        runtimeStatus.running = false;
        logResult('持续学习已停止', 'success');
      } catch (e) {
        logResult('停止错误: ' + e.message, 'error');
      }
    }

    async function doCheckpoint() {
      logResult('保存检查点...', 'info');
      try {
        const result = await apiPost('/api/runtime/checkpoint');
        logResult('检查点已保存: ' + JSON.stringify(result).substring(0, 120), 'success');
      } catch (e) {
        logResult('检查点错误: ' + e.message, 'error');
      }
    }

    async function fetchRuntimeStatus() {
      try {
        const data = await apiGet('/api/runtime/status');
        runtimeStatus.running = data.running || false;
        runtimeStatus.uptime = data.uptime || '0s';
        runtimeStatus.iterations = data.iterations || 0;
        runtimeStatus.retention = data.retention || '0%';
        runtimeStatus.pending_data = data.pending_data || 0;
      } catch { /* ignore */ }
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
      await fetchRuntimeStatus();
      await fetchSocietyMetrics();
      await fetchChatHistory();

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
        fetchRuntimeStatus();
        fetchSocietyMetrics();
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
      // Chat
      chatMessages, chatInput, chatBusy, chatSessionId, chatMessagesRef,
      doChat, fetchChatHistory,
      // Society
      societyAgents, societyMetrics, newAgentPort,
      createAgent, removeAgent, fetchSocietyMetrics, triggerObservation,
      // Runtime
      runtimeStatus,
      startRuntime, stopRuntime, doCheckpoint, fetchRuntimeStatus,
    };
  },
});

app.mount('#app');

<script setup lang="ts">
import type { EChartsOption } from 'echarts'
import { computed, reactive, ref } from 'vue'
import { api, errorMessage } from '../api'
import EChart from '../components/EChart.vue'
import MetricCard from '../components/MetricCard.vue'
import { appState } from '../state'

const form = reactive({ asset_a: '', asset_b: '', k_multiplier: 1.5, n_slots: 10, erp_step: 0.5 })
const result = ref<any>()
const loading = ref(false)
const error = ref('')

async function run() {
  loading.value = true; error.value = ''
  try { result.value = (await api.get('/api/rotation', { params: form })).data }
  catch (requestError) { error.value = errorMessage(requestError) } finally { loading.value = false }
}
async function refreshMacro() {
  loading.value = true; error.value = ''
  try { await api.post('/api/macro/refresh'); await run() } catch (requestError) { error.value = errorMessage(requestError); loading.value = false }
}
const chartOption = computed<EChartsOption>(() => {
  const series = result.value?.series || []
  return { tooltip: { trigger: 'axis' }, legend: { top: 4 }, grid: { left: 55, right: 22, top: 50, bottom: 42 }, xAxis: { type: 'category', data: series.map((item: any) => item.Date), boundaryGap: false, axisLabel: { hideOverlap: true } }, yAxis: { type: 'value', splitLine: { lineStyle: { color: '#e8edf3' } } }, dataZoom: [{ type: 'inside' }], series: [
    { name: '轮动策略', type: 'line', data: series.map((item: any) => item.Cum_Ret), showSymbol: false, lineStyle: { color: '#d97745', width: 2.5 } },
    { name: '持有资产 A', type: 'line', data: series.map((item: any) => item.BH_A_Cum), showSymbol: false, lineStyle: { color: '#64748b' } },
    { name: '持有资产 B', type: 'line', data: series.map((item: any) => item.BH_B_Cum), showSymbol: false, lineStyle: { color: '#4b8f77' } },
  ] }
})
</script>

<template>
  <section class="page-section">
    <div class="section-heading"><div><span class="eyebrow">ERP STRATEGY</span><h2>股债轮动实验</h2><p>使用沪深 300 盈利收益率与十年国债构建风险溢价</p></div></div>
    <article class="panel inline-form"><label>资产 A<select v-model="form.asset_a"><option value="">请选择</option><option v-for="target in appState.targets" :key="target.index_code" :value="target.index_code">{{ target.name }}</option></select></label><label>资产 B<select v-model="form.asset_b"><option value="">请选择</option><option v-for="target in appState.targets" :key="target.index_code" :value="target.index_code">{{ target.name }}</option></select></label><label>K 倍数<input v-model.number="form.k_multiplier" type="number" step="0.1" /></label><label>仓位格数<input v-model.number="form.n_slots" type="number" /></label><label>ERP 步长<input v-model.number="form.erp_step" type="number" step="0.1" /></label><button class="primary-button" :disabled="loading || !form.asset_a || !form.asset_b" @click="run">运行回测</button><button class="subtle-button" :disabled="loading" @click="refreshMacro">刷新宏观数据</button></article>
    <div v-if="error" class="error-banner">{{ error }}</div>
    <div v-if="result" class="metric-grid"><MetricCard label="策略 CAGR" :value="`${result.metrics.CAGR.toFixed(2)}%`" /><MetricCard label="最大回撤" :value="`${result.metrics.MDD.toFixed(2)}%`" tone="negative" /><MetricCard label="Calmar" :value="result.metrics.Calmar.toFixed(2)" /><MetricCard label="累计收益" :value="`${result.metrics.Cum_Ret.toFixed(2)}%`" /><MetricCard label="再平衡次数" :value="String(result.metrics.Rebalances)" /></div>
    <article v-if="result" class="panel"><div class="panel-heading"><div><h3>净值轨迹</h3><p>策略与两种单一资产持有结果</p></div></div><EChart :option="chartOption" height="460px" /></article>
    <div v-else-if="!error" class="empty-state">选择两项资产并运行回测。</div>
  </section>
</template>

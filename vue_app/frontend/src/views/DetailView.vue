<script setup lang="ts">
import type { EChartsOption } from 'echarts'
import { computed, ref, watch } from 'vue'
import { api, errorMessage } from '../api'
import EChart from '../components/EChart.vue'
import MetricCard from '../components/MetricCard.vue'
import { analysisParams, appState, selectedTarget } from '../state'
import type { AnalysisResponse } from '../types'

const analysis = ref<AnalysisResponse>()
const loading = ref(false)
const error = ref('')

const format = (value: number | null | undefined, digits = 2) => value == null ? '—' : value.toLocaleString('zh-CN', { maximumFractionDigits: digits })
const percent = (value: number | null | undefined) => value == null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
const tone = (value: number | null | undefined) => value == null ? 'neutral' : value > 0 ? 'negative' : 'positive'

async function loadAnalysis() {
  if (!appState.selectedIndex) return
  loading.value = true
  error.value = ''
  try {
    const { data } = await api.get<AnalysisResponse>(`/api/targets/${encodeURIComponent(appState.selectedIndex)}/analysis`, { params: analysisParams() })
    analysis.value = data
  } catch (requestError) {
    error.value = errorMessage(requestError)
  } finally {
    loading.value = false
  }
}

watch([() => appState.selectedIndex, () => ({ ...appState.filters })], loadAnalysis, { deep: true, immediate: true })

const priceOption = computed<EChartsOption>(() => {
  const series = analysis.value?.series || []
  return {
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: { top: 4, textStyle: { color: '#64748b' } },
    grid: { left: 64, right: 22, top: 52, bottom: 52 },
    xAxis: { type: 'category', data: series.map((point) => point.Date), boundaryGap: false, axisLabel: { hideOverlap: true } },
    yAxis: { type: 'log', axisLabel: { formatter: (value: number) => value.toLocaleString() }, splitLine: { lineStyle: { color: '#e8edf3' } } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8 }],
    series: [
      { name: '统一价格', type: 'line', data: series.map((point) => point.Close), showSymbol: false, lineStyle: { color: '#12263f', width: 2 } },
      { name: '传统回归', type: 'line', data: series.map((point) => point.Trad_Pred_Price), showSymbol: false, lineStyle: { color: '#d97745', type: 'dashed', width: 2 } },
      { name: '滚动回归', type: 'line', data: series.map((point) => point.Roll_Pred_Price), showSymbol: false, lineStyle: { color: '#446b8f', width: 1.5 } },
      { name: `MA${appState.filters.ma_window}`, type: 'line', data: series.map((point) => point.MA_Price), showSymbol: false, lineStyle: { color: '#4b8f77', width: 1.5 } },
    ],
  }
})

const zOption = computed<EChartsOption>(() => {
  const series = analysis.value?.series || []
  return {
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: { top: 4, textStyle: { color: '#64748b' } },
    grid: { left: 52, right: 22, top: 52, bottom: 42 },
    xAxis: { type: 'category', data: series.map((point) => point.Date), boundaryGap: false, axisLabel: { hideOverlap: true } },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: '#e8edf3' } } },
    series: [
      { name: '传统 Z', type: 'line', data: series.map((point) => point.Trad_Z_Score), showSymbol: false, lineStyle: { color: '#12263f', width: 2 }, markLine: { symbol: 'none', data: [{ yAxis: 2 }, { yAxis: 0 }, { yAxis: -2 }], lineStyle: { color: '#cbd5e1', type: 'dashed' } } },
      { name: '滚动 Z', type: 'line', data: series.map((point) => point.Roll_Z_Score), showSymbol: false, lineStyle: { color: '#446b8f' } },
      { name: 'MA Z', type: 'line', data: series.map((point) => point.MA_Z_Score), showSymbol: false, lineStyle: { color: '#4b8f77' } },
    ],
  }
})
</script>

<template>
  <section class="page-section">
    <div v-if="selectedTarget" class="section-heading">
      <div><span class="eyebrow">{{ selectedTarget.data_source }} · {{ selectedTarget.asset_currency }}</span><h2>{{ selectedTarget.name }}</h2><p>{{ selectedTarget.index_code }}<template v-if="selectedTarget.etf_code"> · 跟踪 {{ selectedTarget.etf_code }}</template></p></div>
      <div v-if="analysis" class="date-badge">数据至 {{ analysis.metrics.latest_date }}</div>
    </div>
    <div v-if="loading" class="empty-state">正在计算长期趋势…</div>
    <div v-else-if="error" class="error-banner">{{ error }}</div>
    <template v-else-if="analysis">
      <div class="metric-grid">
        <MetricCard label="当前统一价格" :value="format(analysis.metrics.latest_close)" :detail="selectedTarget?.report_currency || 'CNY'" />
        <MetricCard label="传统偏离度" :value="percent(analysis.metrics.dev_trad)" :tone="tone(analysis.metrics.dev_trad)" :detail="`${analysis.metrics.trad_range_start} 至 ${analysis.metrics.trad_range_end}`" />
        <MetricCard label="传统 Z 值" :value="format(analysis.metrics.z_trad_latest)" :tone="tone(analysis.metrics.z_trad_latest)" detail="残差标准差单位" />
        <MetricCard label="长期 CAGR" :value="percent(analysis.metrics.cagr_trad)" :detail="analysis.metrics.cagr_95ci" />
        <MetricCard label="滚动偏离度" :value="percent(analysis.metrics.dev_roll)" :tone="tone(analysis.metrics.dev_roll)" :detail="`${appState.filters.rolling_window} 个交易日`" />
        <MetricCard :label="`MA${appState.filters.ma_window} 偏离度`" :value="percent(analysis.metrics.dev_ma)" :tone="tone(analysis.metrics.dev_ma)" />
      </div>
      <article class="panel"><div class="panel-heading"><div><h3>长期价格轨迹</h3><p>对数坐标 · 支持缩放时间范围</p></div></div><EChart :option="priceOption" height="500px" /></article>
      <article class="panel"><div class="panel-heading"><div><h3>标准化偏离度</h3><p>正值代表高于基准，负值代表低于基准</p></div></div><EChart :option="zOption" height="320px" /></article>
    </template>
    <div v-else class="empty-state">请先在数据管理中添加标的。</div>
  </section>
</template>

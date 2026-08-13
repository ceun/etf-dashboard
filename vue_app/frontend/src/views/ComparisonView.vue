<script setup lang="ts">
import type { EChartsOption } from 'echarts'
import { computed, ref, watch } from 'vue'
import { api, errorMessage } from '../api'
import EChart from '../components/EChart.vue'
import { analysisParams, appState } from '../state'
import type { ComparisonRow } from '../types'

const rows = ref<ComparisonRow[]>([])
const loading = ref(false)
const error = ref('')
const maKey = computed(() => `ma_${appState.filters.ma_window}_deviation_pct`)
const number = (value: number | null) => value == null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(2)}`

async function loadComparison() {
  loading.value = true
  error.value = ''
  try {
    const { data } = await api.get<ComparisonRow[]>('/api/comparison', { params: analysisParams() })
    rows.value = data.sort((left, right) => (left.trad_deviation_pct ?? 999) - (right.trad_deviation_pct ?? 999))
  } catch (requestError) {
    error.value = errorMessage(requestError)
  } finally {
    loading.value = false
  }
}

watch(() => ({ ...appState.filters, count: appState.targets.length }), loadComparison, { deep: true, immediate: true })

const chartOption = computed<EChartsOption>(() => ({
  tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
  legend: { top: 4 },
  grid: { left: 110, right: 30, top: 54, bottom: 30 },
  xAxis: { type: 'value', axisLabel: { formatter: '{value}%' }, splitLine: { lineStyle: { color: '#e8edf3' } } },
  yAxis: { type: 'category', data: rows.value.map((row) => row.name) },
  series: [
    { name: '传统偏离', type: 'bar', data: rows.value.map((row) => row.trad_deviation_pct), itemStyle: { color: '#d99a72' } },
    { name: '滚动偏离', type: 'bar', data: rows.value.map((row) => row.roll_deviation_pct), itemStyle: { color: '#7698b5' } },
    { name: `MA${appState.filters.ma_window}`, type: 'bar', data: rows.value.map((row) => row[maKey.value] as number | null), itemStyle: { color: '#78a997' } },
  ],
}))
</script>

<template>
  <section class="page-section">
    <div class="section-heading"><div><span class="eyebrow">CROSS MARKET</span><h2>市场温度计</h2><p>同一套 CNY 口径下比较长期位置</p></div></div>
    <div v-if="loading" class="empty-state">正在计算全部标的…</div>
    <div v-else-if="error" class="error-banner">{{ error }}</div>
    <template v-else>
      <article class="panel table-panel">
        <div class="table-scroll"><table><thead><tr><th>标的</th><th>回归范围</th><th>传统偏离</th><th>传统 Z</th><th>滚动偏离</th><th>MA 偏离</th><th>长期 CAGR</th><th>95% CI</th></tr></thead>
        <tbody><tr v-for="row in rows" :key="row.name"><td><strong>{{ row.name }}</strong><small>{{ row.etf_code }}</small></td><td>{{ row.trad_cagr_range || '—' }}</td><td :class="['heat', (row.trad_deviation_pct ?? 0) > 0 ? 'hot' : 'cool']">{{ number(row.trad_deviation_pct) }}%</td><td>{{ number(row.trad_z_score) }}</td><td>{{ number(row.roll_deviation_pct) }}%</td><td>{{ number(row[maKey] as number | null) }}%</td><td>{{ number(row.trad_cagr_pct) }}%</td><td>{{ row.cagr_95ci || '—' }}</td></tr></tbody></table></div>
      </article>
      <article class="panel"><div class="panel-heading"><div><h3>偏离度横向比较</h3><p>越靠左代表越低于各自基准</p></div></div><EChart :option="chartOption" :height="`${Math.max(360, rows.length * 48)}px`" /></article>
    </template>
  </section>
</template>

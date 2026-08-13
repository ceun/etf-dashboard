<script setup lang="ts">
import { BarChart3, Database, GitCompareArrows, LineChart, RefreshCw, X } from 'lucide-vue-next'
import { useRoute } from 'vue-router'
import { api, errorMessage } from '../api'
import { appState, loadTargets } from '../state'
import { ref } from 'vue'

const route = useRoute()
const syncing = ref(false)
const syncMessage = ref('')

const links = [
  { to: '/', label: '单标的详情', icon: LineChart },
  { to: '/comparison', label: '全市场对比', icon: GitCompareArrows },
  { to: '/data', label: '数据管理', icon: Database },
  { to: '/rotation', label: '股债轮动', icon: BarChart3 },
]

async function syncAll() {
  syncing.value = true
  syncMessage.value = ''
  try {
    const { data } = await api.post('/api/sync-all')
    syncMessage.value = data.failures ? `${data.failures} 个标的失败` : '全部更新成功'
    await loadTargets()
  } catch (error) {
    syncMessage.value = errorMessage(error)
  } finally {
    syncing.value = false
  }
}
</script>

<template>
  <aside class="sidebar" :class="{ open: appState.sidebarOpen }">
    <div class="brand-row">
      <div class="brand-mark">ETF</div>
      <div><strong>今天买什么</strong><span>长期估值实验室</span></div>
      <button class="icon-button mobile-close" @click="appState.sidebarOpen = false"><X :size="18" /></button>
    </div>

    <nav class="nav-list">
      <RouterLink v-for="link in links" :key="link.to" :to="link.to" :class="{ active: route.path === link.to }" @click="appState.sidebarOpen = false">
        <component :is="link.icon" :size="18" />{{ link.label }}
      </RouterLink>
    </nav>

    <div class="sidebar-section">
      <label>分析标的</label>
      <select v-model="appState.selectedIndex" :disabled="appState.loadingTargets">
        <option v-for="target in appState.targets" :key="target.index_code" :value="target.index_code">{{ target.name }}</option>
      </select>
    </div>

    <div class="sidebar-section parameter-grid">
      <label>传统回归起始<input v-model="appState.filters.tradition_start" type="date" /></label>
      <label>传统回归结束<input v-model="appState.filters.tradition_end" type="date" /></label>
      <label>滚动窗口<input v-model.number="appState.filters.rolling_window" type="number" min="20" /></label>
      <label>MA 周期<input v-model.number="appState.filters.ma_window" type="number" min="2" /></label>
      <label>偏离阈值<input v-model.number="appState.filters.deviation_pct" type="number" min="1" max="100" /></label>
    </div>

    <div class="sidebar-footer">
      <button class="primary-button full" :disabled="syncing" @click="syncAll">
        <RefreshCw :size="16" :class="{ spin: syncing }" />{{ syncing ? '正在更新…' : '更新全部数据' }}
      </button>
      <p v-if="syncMessage" class="small-note">{{ syncMessage }}</p>
    </div>
  </aside>
</template>

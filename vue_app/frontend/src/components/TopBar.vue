<script setup lang="ts">
import { Menu, Search } from 'lucide-vue-next'
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { appState } from '../state'

const route = useRoute()
const titles: Record<string, [string, string]> = {
  detail: ['单标的详情', '长期趋势、偏离度与风险位置'],
  comparison: ['全市场对比', '跨市场观察估值与增长差异'],
  data: ['数据管理', '标的、历史数据与同步链路'],
  rotation: ['股债轮动', '基于股债风险溢价的配置实验'],
}
const title = computed(() => titles[String(route.name)] || titles.detail)
</script>

<template>
  <header class="topbar">
    <button class="icon-button mobile-menu" @click="appState.sidebarOpen = true"><Menu :size="20" /></button>
    <div><h1>{{ title[0] }}</h1><p>{{ title[1] }}</p></div>
    <div class="topbar-status"><Search :size="16" /><span>{{ appState.targets.length }} 个标的</span></div>
  </header>
</template>

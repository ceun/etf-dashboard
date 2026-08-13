import { computed, reactive } from 'vue'
import { api } from './api'
import type { Target } from './types'

const today = new Date().toISOString().slice(0, 10)

export const appState = reactive({
  targets: [] as Target[],
  selectedIndex: '',
  loadingTargets: false,
  sidebarOpen: false,
  filters: {
    deviation_pct: 15,
    ma_window: 250,
    rolling_window: 1250,
    tradition_start: '2008-10-31',
    tradition_end: today,
  },
})

export const selectedTarget = computed(() =>
  appState.targets.find((target) => target.index_code === appState.selectedIndex),
)

export async function loadTargets() {
  appState.loadingTargets = true
  try {
    const { data } = await api.get<Target[]>('/api/targets')
    appState.targets = data
    if (!data.some((target) => target.index_code === appState.selectedIndex)) {
      appState.selectedIndex = data[0]?.index_code || ''
    }
  } finally {
    appState.loadingTargets = false
  }
}

export function analysisParams() {
  return { ...appState.filters }
}

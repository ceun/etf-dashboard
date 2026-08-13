<script setup lang="ts">
import * as echarts from 'echarts'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useResizeObserver } from '@vueuse/core'

const props = defineProps<{ option: echarts.EChartsOption; height?: string }>()
const element = ref<HTMLElement>()
let chart: echarts.ECharts | undefined

onMounted(() => {
  chart = echarts.init(element.value!)
  chart.setOption(props.option)
})

watch(() => props.option, (option) => chart?.setOption(option, true), { deep: true })
useResizeObserver(element, () => chart?.resize())
onBeforeUnmount(() => chart?.dispose())
</script>

<template><div ref="element" class="chart" :style="{ height: height || '420px' }" /></template>
